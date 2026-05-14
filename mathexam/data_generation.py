# -*- coding: utf-8 -*-
"""
数据生成模块：包含测试函数、采样方法和实验数据准备

本文件提供：
- three_task_function：三输出测试函数（用于生成模拟数据）
- generate_candidates：拉丁超立方采样方法
- setup_experiment_data：实验数据生成与准备
- setup_sample_weights：样本权重设置
"""

import torch
import numpy as np
from typing import Union, Sequence, Tuple

from pyDOE2 import lhs

from config import (
    BOUNDS,
    DIMENSIONS,
    DEFAULT_SAMPLES,
    TARGET_VALUES,
    SIGMA_VALUES,
)


# 类型别名：ArrayLike可以是浮点数序列或numpy数组
ArrayLike = Union[Sequence[float], np.ndarray]


# ==========================================================================================
# 测试函数定义
# ==========================================================================================

def three_task_function(x: ArrayLike) -> np.ndarray:
    """用于优化实验的三输出测试函数。

    设计原则：
    - y1: 非线性组合（sin, 平方, exp, log）
    - y2: 乘积和三角函数（相关性中等）
    - y3: 指数、对数和幂函数组合（新增任务）

    Args:
        x: 输入数据，可以是：
            - 一维向量（长度必须为5）
            - 二维数组（形状必须为(N, 5)）

    Returns:
        输出数组，形状为(3,)或(N, 3)，取决于输入形状

    Raises:
        ValueError: 如果输入维度不正确
    """
    x = np.asarray(x, dtype=float)
    is_vector = x.ndim == 1

    if is_vector:
        if x.size != 5:
            raise ValueError("一维输入必须长度为5。")
        X = x.reshape(1, 5)
    else:
        if x.shape[1] != 5:
            raise ValueError("二维输入必须形状为(N, 5)。")
        X = x

    # 提取每一维的数据
    x1, x2, x3, x4, x5 = [X[:, i] for i in range(5)]

    # Task 1: 非线性组合
    y1 = np.sin(x1) + 0.5 * x2**2 - x3 + np.exp(-x4) + np.log1p(x5**2)

    # Task 2: 乘积和三角函数
    y2 = x1 * x5 + np.cos(x2) + np.sqrt(np.abs(x3) + 1.0) - np.tanh(x4)

    # Task 3: 指数、对数和幂函数组合
    y3 = (
        np.exp(-0.5 * x1**2)
        + np.log1p(np.abs(x2))
        - x3 * x4
        + np.power(np.abs(x5) + 0.1, 0.5)
    )

    # 将三个输出堆叠成一个数组
    Y = np.stack([y1, y2, y3], axis=1)
    return Y[0] if is_vector else Y


# ==========================================================================================
# 拉丁超立方采样
# ==========================================================================================

def generate_candidates(
    n_samples: int,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    seed: int = None
) -> np.ndarray:
    """在指定的边界内生成拉丁超立方采样（LHS）样本。

    拉丁超立方采样是一种分层采样方法，能够在多维空间中
    生成均匀分布的样本点，比随机采样更高效。

    Args:
        n_samples: 要生成的样本数量
        lower_bounds: 各维度的下界列表
        upper_bounds: 各维度的上界列表
        seed: 随机种子，用于确保可复现性（可选）

    Returns:
        样本数组，形状为[n_samples, dimensions]
    """
    lower = np.array(lower_bounds)
    upper = np.array(upper_bounds)
    dim = len(lower)

    # 在[0, 1]^dim单位超立方体内生成LHS样本
    # 如果提供了种子，使用它来确保可复现性
    unit_lhs = lhs(dim, samples=n_samples, random_state=seed)

    # 将样本线性缩放到指定的[lower, upper]范围内
    return unit_lhs * (upper - lower) + lower


# ==========================================================================================
# 实验数据生成
# ==========================================================================================

def setup_experiment_data(
    device: torch.device,
    bounds: Tuple[int, int] = BOUNDS,
    dimensions: int = DIMENSIONS,
    samples: Tuple[int, int, int] = DEFAULT_SAMPLES,
    seed: int = None
) -> dict:
    """生成并准备实验数据，包含训练集、验证集和测试集。

    Args:
        device: 计算设备（如torch.device("cuda:0")或torch.device("cpu")）
        bounds: 输入变量的取值范围，格式为(下界, 上界)
        dimensions: 输入变量的维度
        samples: 各数据集的样本数量，格式为(训练集, 验证集, 测试集)
        seed: 随机种子，用于确保可复现性（可选）

    Returns:
        数据集字典，包含：
            - "train": {"x": tensor, "y": tensor}
            - "val": {"x": tensor, "y": tensor}
            - "test": {"x": tensor, "y": tensor}
    """
    n_train, n_val, n_test = samples
    lower_bounds, upper_bounds = [bounds[0]] * dimensions, [bounds[1]] * dimensions

    # 为每个数据集使用不同的种子（基于基础种子）
    train_seed = seed if seed is None else seed
    val_seed = seed + 1 if seed is not None else None
    test_seed = seed + 2 if seed is not None else None

    # 生成训练数据
    x_train_np = generate_candidates(n_train, lower_bounds, upper_bounds, seed=train_seed)
    y_train_np = three_task_function(x_train_np)

    datasets = {
        "train": {
            "x": torch.from_numpy(x_train_np).float().to(device),
            "y": torch.from_numpy(y_train_np).float().to(device),
        }
    }

    # 生成独立的验证集和测试集
    for split, n_samples, split_seed in zip(["val", "test"], [n_val, n_test], [val_seed, test_seed]):
        x_np = generate_candidates(n_samples, lower_bounds, upper_bounds, seed=split_seed)
        y_np = three_task_function(x_np)
        datasets[split] = {
            "x": torch.from_numpy(x_np).float().to(device),
            "y": torch.from_numpy(y_np).float().to(device),
        }

    return datasets


# ==========================================================================================
# 样本权重计算
# ==========================================================================================

def compute_sample_weights(
    targets: torch.Tensor,
    target_values: Tuple[float, float, float],
    sigma_values: Tuple[float, float, float],
    n_samples: int
) -> dict:
    """为局部优化计算高斯加权的样本权重。

    使用高斯核函数为每个样本计算权重，使得靠近目标值的样本获得更高权重。
    这有助于模型在局部区域内获得更好的预测性能。

    Args:
        targets: 目标张量，形状为[n_samples, num_tasks]
        target_values: 各任务的目标值列表
        sigma_values: 各任务的高斯核标准差列表
        n_samples: 样本数量（用于权重归一化）

    Returns:
        权重字典，格式为{"local_A": tensor, "local_B": tensor, "local_C": tensor}
        每个张量的形状为[n_samples]
    """
    weights = {}

    for i, (target_val, sigma_val) in enumerate(zip(target_values, sigma_values)):
        # 计算每个样本到目标值的平方马氏距离
        dist = (targets[:, i] - target_val) ** 2 / (2 * sigma_val**2)

        # 使用高斯核函数计算权重
        weight = torch.exp(-dist)

        # 对权重进行归一化，使其总和等于样本数（保持数值稳定性）
        weight_sum = torch.sum(weight)
        if weight_sum > 1e-8:
            weight = (weight / weight_sum) * n_samples
        else:
            # 如果所有权重都接近零，则分配均匀权重
            weight = torch.ones_like(weight) * (n_samples / weight.numel())

        # 使用字母标记局部任务（A, B, C）
        weights[f"local_{chr(65 + i)}"] = weight

    return weights


def setup_sample_weights(
    train_data: dict,
    val_data: dict,
    target_values: Tuple[float, float, float] = TARGET_VALUES,
    sigma_values: Tuple[float, float, float] = SIGMA_VALUES
) -> dict:
    """为局部优化任务设置样本权重。

    为训练集和验证集分别计算样本权重，用于后续的加权损失函数。

    Args:
        train_data: 训练数据字典，包含"x"和"y"
        val_data: 验证数据字典，包含"x"和"y"
        target_values: 各任务的目标值（默认使用config中的值）
        sigma_values: 各任务的高斯核标准差（默认使用config中的值）

    Returns:
        样本权重字典，格式为：
            {"train": {"local_A": tensor, ...}, "val": {"local_A": tensor, ...}}
    """
    # 为训练集计算样本权重
    sample_weights_train = compute_sample_weights(
        train_data["y"], target_values, sigma_values, train_data["y"].size(0)
    )

    # 为验证集计算样本权重
    sample_weights_val = compute_sample_weights(
        val_data["y"], target_values, sigma_values, val_data["y"].size(0)
    )

    return {"train": sample_weights_train, "val": sample_weights_val}