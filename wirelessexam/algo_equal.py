# -*- coding: utf-8 -*-
"""
等权重基线。
"""

import torch
import tqdm

from common import IndexedTensorDataset, create_loss_function
from config import BATCH_SIZE, LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM


def run_baseline_training_loop(
    model, datasets, num_epochs, task_ids, sample_weights=None
):
    """执行固定等权任务的基线训练。"""
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
    equal_weights = torch.ones(len(task_ids), device=device) / len(task_ids)
    history = {"weights": [], "losses": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        num_batches = 0

        epochs_iter = tqdm.tqdm(
            train_loader,
            desc=f"Baseline Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )
            total_loss = loss_fn.combine_task_losses(
                model,
                task_losses,
                task_weights=None,
                is_val=False,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            history["weights"].append(equal_weights.detach().cpu().numpy().copy())
            epochs_iter.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        history["losses"].append(avg_epoch_loss)
        print(
            f"Baseline Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}"
        )

    return history
