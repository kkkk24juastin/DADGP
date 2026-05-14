# -*- coding: utf-8 -*-
"""
动态权重平均（DWA）算法：基于历史损失动态调整任务权重

DWA算法基于 Liu et al. (2019) 的方法，通过任务损失的变化率
来动态调整各任务的权重，使损失下降较慢的任务获得更高权重。

参考论文：
"End-to-End Multi-Task Learning with Attention"
"""

import torch
import tqdm

from config import LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM, BATCH_SIZE, DWA_TEMPERATURE
from common import IndexedTensorDataset, create_loss_function


# ==========================================================================================
# DWA训练循环
# ==========================================================================================

def run_dwa_baseline_training_loop(
    model,
    datasets,
    num_epochs,
    task_ids,
    sample_weights,
    temperature=DWA_TEMPERATURE
):
    """执行动态权重平均（DWA）基线训练。

    DWA的核心思想：
    1. 记录每个任务的损失历史
    2. 计算损失变化率 ratio = L(t) / L(t-1)
    3. 通过softmax归一化权重：w = softmax(ratio / T) * n

    温度参数T控制权重调整的平滑程度：
    - 较大的T：权重调整更平滑
    - 较小的T：权重调整更激进

    Args:
        model: 多任务DGP模型
        datasets: 数据集字典，包含"train"
        num_epochs: 训练轮数
        task_ids: 任务标识符列表
        sample_weights: 样本权重字典
        temperature: DWA温度参数（默认2.0）

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

    device = next(model.parameters()).device
    loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )

    # 记录每个任务的历史平均损失
    prev_epoch_losses = []

    for epoch in range(num_epochs):
        model.train()

        # 计算当前epoch的任务权重
        if len(prev_epoch_losses) < 2:
            # 前2个epoch使用等权重
            weights = torch.ones(len(task_ids), device=device)
        else:
            # 计算损失变化率
            last_loss = torch.tensor(prev_epoch_losses[-1], device=device)
            prev_loss = torch.tensor(prev_epoch_losses[-2], device=device)
            ratio = last_loss / (prev_loss + 1e-8)
            # 通过softmax归一化权重
            weights = torch.softmax(ratio / temperature, dim=0) * len(task_ids)

        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"DWA Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        epoch_total_loss, num_batches = 0.0, 0
        epoch_task_loss_totals = [0.0 for _ in task_ids]

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
            weights_tensor = weights.to(train_x_batch.device)
            total_loss = sum(
                weights_tensor[i] * task_losses[i]
                for i in range(len(task_ids))
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1

            # 累积各任务损失
            for i, loss_val in enumerate(task_losses):
                epoch_task_loss_totals[i] += loss_val.item()

            # 显示当前权重
            weight_list = [round(w, 4) for w in weights_tensor.detach().cpu().tolist()]
            epochs_iter.set_postfix(loss=total_loss.item(), w=weight_list)

        scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_task_losses = [
            tot / num_batches if num_batches > 0 else 0.0
            for tot in epoch_task_loss_totals
        ]
        prev_epoch_losses.append(avg_task_losses)

        # 打印epoch统计
        weights_print = [round(w, 4) for w in weights.detach().cpu().tolist()]
        print(
            f"DWA Epoch {epoch + 1} finished. "
            f"Average Loss: {avg_epoch_loss:.4f}, Weights: {weights_print}"
        )
