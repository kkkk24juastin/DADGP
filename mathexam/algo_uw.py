# -*- coding: utf-8 -*-
"""
不确定性加权（Uncertainty Weighting）算法：通过学习任务不确定性来调整权重

UW算法基于 Kendall et al. (2018) 的方法，通过学习每个任务的log方差
来自动调整任务权重。损失函数形式为：
    L = sum_i [1/(2*sigma_i^2) * L_i + log(sigma_i)]

参考论文：
"Multi-Task Learning Using Uncertainty to Weigh Losses"
"""

import torch
import tqdm

from config import LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM, BATCH_SIZE
from common import IndexedTensorDataset, create_loss_function


# ==========================================================================================
# 不确定性加权训练循环
# ==========================================================================================

def run_uncertainty_weighting_training_loop(
    model,
    datasets,
    num_epochs,
    task_ids,
    sample_weights
):
    """执行不确定性加权（UW）基线训练。

    核心思想：
    - 为每个任务引入一个可学习的log方差参数 log_var_i
    - 损失函数自动平衡精度和方差：
      L = sum_i [1/(2*exp(log_var_i)) * L_i + log_var_i/2]
    - 不确定性大的任务（sigma大）获得更小的权重
    - 模型会自动调整各任务的不确定性

    Args:
        model: 多任务DGP模型
        datasets: 数据集字典，包含"train"
        num_epochs: 训练轮数
        task_ids: 任务标识符列表
        sample_weights: 样本权重字典

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

    device = next(model.parameters()).device

    # 初始化log方差参数（每个任务一个）
    # 初始化为0，对应sigma=1的初始不确定性
    log_vars = torch.zeros(len(task_ids), requires_grad=True, device=device)

    # 分别创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    log_var_optimizer = torch.optim.Adam([log_vars], lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    log_var_scheduler = torch.optim.lr_scheduler.ExponentialLR(log_var_optimizer, gamma=LR_GAMMA)

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
            train_loader, desc=f"UW Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            # 清空梯度
            optimizer.zero_grad()
            log_var_optimizer.zero_grad()

            # 前向传播
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
            task_losses_tensor = torch.stack(task_losses)

            # 计算基于不确定性的加权损失
            # L = sum_i [exp(-log_var_i) * L_i + log_var_i] / 2
            precision = torch.exp(-log_vars)  # 精度 = 1/sigma^2
            weighted_loss = torch.sum(precision * task_losses_tensor + log_vars) / 2.0

            # 反向传播
            weighted_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            torch.nn.utils.clip_grad_norm_([log_vars], MAX_GRAD_NORM)

            # 更新参数
            optimizer.step()
            log_var_optimizer.step()

            epoch_total_loss += weighted_loss.item()
            num_batches += 1

            # 计算当前归一化权重用于显示
            with torch.no_grad():
                current_weights = precision / precision.sum()
                weight_list = [round(w.item(), 4) for w in current_weights]

            epochs_iter.set_postfix(loss=weighted_loss.item(), w=weight_list)

        # 更新学习率
        scheduler.step()
        log_var_scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0

        # 计算并显示最终权重和sigma
        with torch.no_grad():
            final_precision = torch.exp(-log_vars)
            final_weights = final_precision / final_precision.sum()
            weights_print = [round(w.item(), 4) for w in final_weights]
            sigmas_print = [round(torch.exp(lv).item(), 4) for lv in log_vars]

        print(
            f"UW Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Weights: {weights_print}, Sigmas: {sigmas_print}"
        )
