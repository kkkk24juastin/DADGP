# -*- coding: utf-8 -*-
"""
消融实验：去除样本加权，仅保留 DADGP 任务调权。
"""

import torch
import tqdm

from common import IndexedTensorDataset
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LR_GAMMA,
    MAX_GRAD_NORM,
    META_LEARNING_RATE,
    META_LR_GAMMA,
)


def run_dadgp_no_sample_weights(model, dadgp, datasets, num_epochs):
    """执行无样本加权的 DADGP 训练。"""
    train_data, val_data = datasets["train"], datasets["val"]

    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataset = IndexedTensorDataset(val_data["x"], val_data["y"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    meta_optimizer = torch.optim.Adam([dadgp.meta_weights], lr=META_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        meta_optimizer, gamma=META_LR_GAMMA
    )

    history = {"weights": [], "losses": []}
    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0
        epochs_iter = tqdm.tqdm(
            train_loader,
            desc=f"Ablation(-SampleAttn) Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = dadgp.model_fit(
                model, output, train_y_batch, train_indices, is_val=False
            )
            normalized_weights = dadgp.get_normalized_weights()
            total_loss = dadgp.loss_fn.combine_task_losses(
                model,
                task_losses,
                normalized_weights,
                is_val=False,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            meta_optimizer.zero_grad()
            dadgp.unrolled_backward(
                train_x_batch,
                train_y_batch,
                train_indices,
                val_x_full,
                val_y_full,
                val_indices_full,
                optimizer.param_groups[0]["lr"],
                optimizer,
            )
            meta_optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            history["weights"].append(
                dadgp.get_normalized_weights().detach().cpu().numpy().copy()
            )
            epochs_iter.set_postfix(loss=total_loss.item())

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        scheduler.step()
        meta_scheduler.step()

        weights_str = ", ".join(
            f"{weight:.4f}"
            for weight in dadgp.get_normalized_weights().detach().cpu().numpy()
        )
        print(
            f"Ablation(-SampleAttn) Epoch {epoch + 1} finished. "
            f"Loss: {avg_epoch_loss:.4f}, Weights: [{weights_str}]"
        )

    return history
