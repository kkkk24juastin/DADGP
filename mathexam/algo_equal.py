# -*- coding: utf-8 -*-
"""
等权重基线算法：使用固定任务权重（1/n）进行训练

该算法是最简单的多任务学习基线方法：
- 每个任务权重相等（1/n）
- 不使用动态权重调整
"""

import torch
import tqdm

from config import LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM, BATCH_SIZE
from common import IndexedTensorDataset, create_loss_function


# ==========================================================================================
# 等权重基线训练循环
# ==========================================================================================

def run_baseline_training_loop(
    model,
    datasets,
    num_epochs,
    task_ids,
    sample_weights=None
):
    """执行固定任务权重的基线训练循环（等权重）。

    Args:
        model: 多任务DGP模型
        datasets: 数据集字典，包含"train"
        num_epochs: 训练轮数
        task_ids: 任务标识符列表
        sample_weights: 样本权重字典（可选）

    Returns:
        无返回值，模型在训练过程中被更新
    """
    train_data = datasets["train"]
    val_data = datasets["val"]

    # 设置数据加载器
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # 设置优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    # 创建损失函数
    loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"Baseline Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)

            # 备份版语义：每个任务损失自身包含完整ELBO
            task_losses = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )
            total_loss = torch.mean(torch.stack(task_losses))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            epochs_iter.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Baseline Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}")
