# -*- coding: utf-8 -*-
"""
MGDA（Multiple Gradient Descent Algorithm）算法：多目标优化方法

MGDA算法基于 Sener & Koltun (2018) 的方法，通过求解凸优化问题
找到帕累托最优的梯度方向。使用Frank-Wolfe算法求解最优权重。

参考论文：
"Multi-Task Learning as Multi-Objective Optimization"
"""

import torch
import tqdm

from config import (
    LEARNING_RATE, LR_GAMMA, MAX_GRAD_NORM, BATCH_SIZE,
    MGDA_MAX_ITER, MGDA_CONVERGENCE_THRESHOLD
)
from common import IndexedTensorDataset, create_loss_function


# ==========================================================================================
# MGDA求解器
# ==========================================================================================

def mgda_solver(grads, device):
    """使用Frank-Wolfe算法求解MGDA的最优权重。

    求解凸优化问题：
        min_w ||sum_i w_i * g_i||^2
        s.t. sum_i w_i = 1, w_i >= 0

    Args:
        grads: 列表，每个元素是一个任务的梯度张量列表
        device: 计算设备

    Returns:
        最优权重张量，形状为[num_tasks]
    """
    num_tasks = len(grads)
    if num_tasks == 0:
        return torch.tensor([], device=device)

    # 将每个任务的梯度展平并堆叠
    flat_grads = []
    for task_grads in grads:
        flat_grad = torch.cat(
            [
                g.flatten() if g is not None else torch.zeros(1, device=device)
                for g in task_grads
            ]
        )
        flat_grads.append(flat_grad)

    # 计算Gram矩阵 G_ij = <g_i, g_j>
    gram_matrix = torch.zeros((num_tasks, num_tasks), device=device)
    for i in range(num_tasks):
        for j in range(num_tasks):
            gram_matrix[i, j] = torch.dot(flat_grads[i], flat_grads[j])

    # 使用Frank-Wolfe算法求解
    # 初始化：等权重
    weights = torch.ones(num_tasks, device=device) / num_tasks

    for _ in range(MGDA_MAX_ITER):
        # 计算当前梯度: grad_w = 2 * G * w
        grad_w = 2.0 * torch.mv(gram_matrix, weights)

        # Frank-Wolfe: 找到使<grad_w, e_i>最小的单纯形顶点
        min_idx = torch.argmin(grad_w)

        # 创建目标方向（单纯形顶点）
        direction = torch.zeros(num_tasks, device=device)
        direction[min_idx] = 1.0

        # 线搜索: 找到最优步长
        d_minus_w = direction - weights

        # 二次函数系数: a*gamma^2 + b*gamma + c
        a = torch.dot(d_minus_w, torch.mv(gram_matrix, d_minus_w))
        b = 2.0 * torch.dot(weights, torch.mv(gram_matrix, d_minus_w))

        # 最优步长
        if a > 1e-8:
            gamma = -b / (2.0 * a)
            gamma = torch.clamp(gamma, 0.0, 1.0)
        else:
            gamma = 0.0

        # 更新权重
        new_weights = weights + gamma * d_minus_w

        # 检查收敛
        if torch.norm(new_weights - weights) < MGDA_CONVERGENCE_THRESHOLD:
            break

        weights = new_weights

    return weights


# ==========================================================================================
# MGDA训练循环
# ==========================================================================================

def run_mgda_training_loop(model, datasets, num_epochs, task_ids, sample_weights):
    """执行MGDA多目标优化训练。

    MGDA的核心思想：
    1. 为每个任务单独计算梯度
    2. 使用Frank-Wolfe算法找到最优权重组合
    3. 使用聚合梯度更新模型参数

    该方法能够在帕累托前沿上找到最优解，避免任务间的梯度冲突。

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

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(
            train_loader, desc=f"MGDA Epoch {epoch + 1}/{num_epochs}", leave=False
        )
        epoch_total_loss, num_batches = 0.0, 0
        epoch_weights_sum = None

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()

            # 为每个任务单独计算梯度
            task_gradients = []
            task_losses = []
            output = model(train_x_batch)
            task_loss_values = loss_fn.task_losses(
                model,
                output,
                train_y_batch,
                task_ids,
                train_indices,
                is_val=False,
            )
            params = list(model.parameters())

            for task_loss in task_loss_values:
                task_losses.append(task_loss.item())

                # 计算当前任务的梯度
                grads = torch.autograd.grad(
                    task_loss, params, retain_graph=True, allow_unused=True
                )
                task_gradients.append(
                    [g.clone() if g is not None else None for g in grads]
                )

            # 使用MGDA求解器计算最优权重
            optimal_weights = mgda_solver(task_gradients, device)

            # 使用最优权重聚合梯度
            aggregated_grads = []
            for param_idx in range(len(params)):
                grad_sum = None
                for task_idx, task_grads in enumerate(task_gradients):
                    if task_grads[param_idx] is not None:
                        weighted_grad = optimal_weights[task_idx] * task_grads[param_idx]
                        if grad_sum is None:
                            grad_sum = weighted_grad.clone()
                        else:
                            grad_sum += weighted_grad

                aggregated_grads.append(grad_sum)

            # 将聚合梯度赋值给模型参数
            for param, grad in zip(params, aggregated_grads):
                if grad is not None:
                    param.grad = grad.clone()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # 更新参数
            optimizer.step()

            # 累积权重用于显示
            if epoch_weights_sum is None:
                epoch_weights_sum = optimal_weights.detach().cpu()
            else:
                epoch_weights_sum += optimal_weights.detach().cpu()

            avg_task_loss = sum(task_losses) / len(task_losses)
            epoch_total_loss += avg_task_loss
            num_batches += 1

            # 显示当前权重
            weight_list = [round(w.item(), 4) for w in optimal_weights]
            epochs_iter.set_postfix(loss=avg_task_loss, w=weight_list)

        scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_weights = (
            epoch_weights_sum / num_batches if num_batches > 0 else epoch_weights_sum
        )
        weights_print = [round(w, 4) for w in avg_weights.tolist()]

        print(
            f"MGDA Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Average Weights: {weights_print}"
        )
