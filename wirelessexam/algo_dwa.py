# -*- coding: utf-8 -*-
"""
DWA 基线。
"""

import torch
import tqdm

from common import IndexedTensorDataset, create_loss_function
from config import (
    BATCH_SIZE,
    DWA_RATIO_EPS,
    DWA_TEMPERATURE,
    LEARNING_RATE,
    LR_GAMMA,
    MAX_GRAD_NORM,
)


def _compute_dwa_weights(prev_epoch_losses, num_tasks, device, temperature):
    """使用经典 DWA 比值计算任务权重。"""
    if len(prev_epoch_losses) < 2:
        return torch.ones(num_tasks, device=device)

    last_loss = torch.tensor(prev_epoch_losses[-1], device=device)
    prev_loss = torch.tensor(prev_epoch_losses[-2], device=device)
    if not torch.isfinite(last_loss).all() or not torch.isfinite(prev_loss).all():
        return torch.ones(num_tasks, device=device)

    ratio = last_loss / prev_loss.clamp_min(DWA_RATIO_EPS)
    if not torch.isfinite(ratio).all():
        return torch.ones(num_tasks, device=device)

    weights = torch.softmax(ratio / max(temperature, DWA_RATIO_EPS), dim=0) * num_tasks
    if not torch.isfinite(weights).all():
        return torch.ones(num_tasks, device=device)
    return weights


def _check_finite_named_tensors(named_tensors, error_prefix):
    """校验张量是否为有限值，并在失败时报告来源。"""
    for tensor_name, tensor in named_tensors:
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"{error_prefix}: detected non-finite values in {tensor_name}."
            )


def run_dwa_baseline_training_loop(
    model,
    datasets,
    num_epochs,
    task_ids,
    sample_weights,
    temperature=DWA_TEMPERATURE,
):
    """执行 DWA 训练。"""
    train_data = datasets["train"]
    val_data = datasets["val"]
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )
    device = next(model.parameters()).device
    prev_epoch_losses = []
    history = {"weights": [], "losses": []}

    for epoch in range(num_epochs):
        model.train()
        weights = _compute_dwa_weights(
            prev_epoch_losses,
            len(task_ids),
            device,
            temperature,
        )
        normalized_weights = weights / weights.sum().clamp_min(1e-12)

        epoch_total_loss = 0.0
        num_batches = 0
        epoch_task_loss_totals = [0.0 for _ in task_ids]
        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"DWA Epoch {epoch + 1}/{num_epochs}", leave=False
        )

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            _check_finite_named_tensors(
                model.named_parameters(),
                f"DWA epoch {epoch + 1} before forward",
            )
            output = model(train_x_batch)
            task_losses = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )
            weights_tensor = weights.to(train_x_batch.device)
            total_loss = loss_fn.combine_task_losses(
                model,
                task_losses,
                task_weights=weights_tensor,
                is_val=False,
            )
            _check_finite_named_tensors(
                [
                    (f"task_loss_{task_id}", task_loss)
                    for task_id, task_loss in zip(task_ids, task_losses)
                ],
                f"DWA epoch {epoch + 1} after loss computation",
            )
            _check_finite_named_tensors(
                [("total_loss", total_loss), ("weights", weights_tensor)],
                f"DWA epoch {epoch + 1} before backward",
            )

            total_loss.backward()
            _check_finite_named_tensors(
                [
                    (f"{param_name}.grad", param.grad)
                    for param_name, param in model.named_parameters()
                ],
                f"DWA epoch {epoch + 1} after backward",
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            _check_finite_named_tensors(
                model.named_parameters(),
                f"DWA epoch {epoch + 1} after optimizer step",
            )

            epoch_total_loss += total_loss.item()
            num_batches += 1
            for idx, loss_value in enumerate(task_losses):
                epoch_task_loss_totals[idx] += loss_value.item()
            history["weights"].append(normalized_weights.detach().cpu().numpy().copy())

            weight_list = [
                round(weight, 4) for weight in weights_tensor.detach().cpu().tolist()
            ]
            epochs_iter.set_postfix(loss=total_loss.item(), w=weight_list)

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        avg_task_losses = [
            total / num_batches if num_batches > 0 else 0.0
            for total in epoch_task_loss_totals
        ]
        prev_epoch_losses.append(avg_task_losses)

        weights_print = [
            round(weight, 4) for weight in normalized_weights.detach().cpu().tolist()
        ]
        avg_task_losses_print = [round(loss, 4) for loss in avg_task_losses]
        print(
            f"DWA Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Normalized Weights: {weights_print}, "
            f"Avg Task Losses: {avg_task_losses_print}"
        )

    return history
