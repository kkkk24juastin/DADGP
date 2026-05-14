# -*- coding: utf-8 -*-
"""
Uncertainty Weighting 基线。
"""

import torch
import tqdm

from common import IndexedTensorDataset, create_loss_function
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LR_GAMMA,
    MAX_GRAD_NORM,
)


def run_uncertainty_weighting_training_loop(
    model,
    datasets,
    num_epochs,
    task_ids,
    sample_weights,
):
    """执行原始同方差不确定性加权的 Uncertainty Weighting 训练。"""
    train_data = datasets["train"]
    val_data = datasets["val"]
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    device = next(model.parameters()).device
    log_vars = torch.zeros(len(task_ids), requires_grad=True, device=device)
    log_var_update_interval = 4

    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    log_var_optimizer = torch.optim.Adam([log_vars], lr=LEARNING_RATE)
    model_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        model_optimizer, gamma=LR_GAMMA
    )
    log_var_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        log_var_optimizer, gamma=LR_GAMMA
    )
    loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )
    history = {"weights": [], "losses": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        log_var_accumulated_batches = 0
        log_var_optimizer.zero_grad()
        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"UW Epoch {epoch + 1}/{num_epochs}", leave=False
        )

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            model_optimizer.zero_grad()

            output = model(train_x_batch)
            task_losses = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )
            task_losses_tensor = torch.stack(task_losses)
            precision = torch.exp(-log_vars)
            uw_data_loss = torch.sum(precision * task_losses_tensor + log_vars) / 2.0
            weighted_loss = uw_data_loss

            weighted_loss.backward()
            log_var_accumulated_batches += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            model_optimizer.step()

            if log_var_accumulated_batches >= log_var_update_interval:
                log_vars.grad.div_(log_var_accumulated_batches)
                torch.nn.utils.clip_grad_norm_([log_vars], MAX_GRAD_NORM)
                log_var_optimizer.step()
                log_var_optimizer.zero_grad()
                log_var_accumulated_batches = 0

            epoch_total_loss += weighted_loss.item()
            num_batches += 1

            with torch.no_grad():
                current_coefficients = 0.5 * torch.exp(-log_vars)
                history["weights"].append(
                    current_coefficients.detach().cpu().numpy().copy()
                )
                weight_list = [
                    round(weight.item(), 4) for weight in current_coefficients
                ]
            epochs_iter.set_postfix(loss=weighted_loss.item(), w=weight_list)

        if log_var_accumulated_batches > 0:
            log_vars.grad.div_(log_var_accumulated_batches)
            torch.nn.utils.clip_grad_norm_([log_vars], MAX_GRAD_NORM)
            log_var_optimizer.step()
            log_var_optimizer.zero_grad()

        model_scheduler.step()
        log_var_scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        with torch.no_grad():
            final_coefficients = 0.5 * torch.exp(-log_vars)
            weights_print = [round(weight.item(), 4) for weight in final_coefficients]
            variances_print = [
                round(torch.exp(log_var).item(), 4) for log_var in log_vars
            ]

        print(
            f"UW Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Loss Coefficients: {weights_print}, Variances: {variances_print}"
        )

    return history
