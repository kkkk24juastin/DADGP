# -*- coding: utf-8 -*-
"""
消融实验1：DA-DGP无样本加权（-Sample Attn）

该消融实验验证样本加权机制的作用：
- 使用DA-DGP任务调权
- 但不使用样本加权机制

用于验证样本注意力对最终性能的贡献。
"""

import torch
import tqdm
import numpy as np

from config import (
    LEARNING_RATE, META_LEARNING_RATE, LR_GAMMA, META_LR_GAMMA,
    MAX_GRAD_NORM, BATCH_SIZE, WEIGHT_INIT, NUM_HIDDEN_DGP_DIMS,
    TRAIN_TASKS, PRI_TASKS, EXPERIMENT_SAMPLES
)
from common import IndexedTensorDataset, create_loss_function
from models_dgp import MultitaskDeepGP
from data_generation import setup_experiment_data
from algo_da_dgp import DADGP


# ==========================================================================================
# 消融实验：DA-DGP无样本加权训练循环
# ==========================================================================================

def run_da_dgp_no_sample_weights(model, dadgp, datasets, num_epochs):
    """执行DA-DGP训练，但不使用样本加权（消融样本注意力）。

    该函数与标准DA-DGP训练的区别：
    - 样本权重始终为None（不使用高斯加权）
    - 其他流程（双层优化）保持一致

    Args:
        model: 多任务DGP模型
        dadgp: DADGP实例（已设置sample_weights=None）
        datasets: 数据集字典
        num_epochs: 训练轮数

    Returns:
        history: 训练历史字典
    """
    train_data, val_data = datasets["train"], datasets["val"]

    # 设置数据加载器
    train_dataset = IndexedTensorDataset(train_data["x"], train_data["y"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # 使用完整训练集进行虚拟步骤
    full_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    train_x_full, train_y_full, train_indices_full = next(iter(full_train_loader))

    # 验证集
    val_dataset = IndexedTensorDataset(val_data["x"], val_data["y"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=False
    )
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    meta_optimizer = torch.optim.Adam([dadgp.meta_weights], lr=META_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=META_LR_GAMMA)

    history = {"weights": [], "losses": []}

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(
            train_loader,
            desc=f"Ablation(-SampleAttn) Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            # 模型参数更新
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = dadgp.model_fit(
                model, output, train_y_batch, train_indices, is_val=False
            )
            normalized_weights = dadgp.get_normalized_weights()
            total_loss = sum(
                weight * task_loss
                for weight, task_loss in zip(normalized_weights, task_losses)
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # 元更新步骤
            meta_optimizer.zero_grad()
            dadgp.unrolled_backward(
                train_x_full,
                train_y_full,
                train_indices_full,
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

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        history["losses"].append(avg_epoch_loss)
        scheduler.step()
        meta_scheduler.step()

        weights_str = ", ".join(
            f"{w:.4f}" for w in dadgp.get_normalized_weights().detach().cpu().numpy()
        )
        print(
            f"Ablation(-SampleAttn) Epoch {epoch + 1} finished. "
            f"Loss: {avg_epoch_loss:.4f}, Weights: [{weights_str}]"
        )

    return history


def run_ablation_no_sample_attn(device, num_epochs, datasets=None, seed=None):
    """运行消融实验：DA-DGP无样本加权。

    Args:
        device: 计算设备
        num_epochs: 训练轮数
        datasets: 可复用的训练/验证/测试数据集；为空时按 seed 重新生成
        seed: 随机种子，用于确保可复现性（可选）

    Returns:
        model: 训练后的模型
    """
    print("\n=== Running Ablation: -Sample Attn (DA-DGP, no sample weights) ===")

    # 设置随机种子（确保可复现性）
    if seed is not None:
        # Python内置随机模块
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 优先复用当前实验已生成的数据，确保与其他方法使用完全相同的数据集。
    if datasets is None:
        datasets = setup_experiment_data(device, samples=EXPERIMENT_SAMPLES, seed=seed)

    train_data = datasets["train"]
    num_tasks = train_data["y"].size(-1)

    # 创建模型
    model = MultitaskDeepGP(
        train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
    ).to(device)

    # 关键：sample_weights=None（不使用样本加权）
    dadgp = DADGP(
        model, device, TRAIN_TASKS, PRI_TASKS,
        weight_init=WEIGHT_INIT, sample_weights=None
    )
    dadgp.loss_fn = create_loss_function(
        sample_weights=None,
        split_sizes={
            "train": datasets["train"]["y"].size(0),
            "val": datasets["val"]["y"].size(0),
        },
    )

    # 训练
    history = run_da_dgp_no_sample_weights(model, dadgp, datasets, num_epochs)

    return model
