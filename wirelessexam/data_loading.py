# -*- coding: utf-8 -*-
"""
真实数据加载与样本权重计算。
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from config import (
    DATA_DIR,
    INPUT_DIMENSIONS,
    NUM_TASKS,
    TASK_IDS,
    TRAIN_SPLIT_NAME,
    VAL_SPLIT_NAME,
)


def _candidate_split_paths(data_dir: Path, split_name: str) -> Iterable[Path]:
    suffixes = [
        f"{split_name}.csv",
        f"{split_name}.xlsx",
        f"{split_name}_data.xlsx",
    ]
    for suffix in suffixes:
        yield data_dir / suffix


def resolve_split_path(split_name: str, data_dir: Path = DATA_DIR) -> Path:
    candidates = list(_candidate_split_paths(data_dir, split_name))
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"未找到 {split_name} 数据文件。已检查路径: "
        + ", ".join(str(path) for path in candidates)
    )


def load_tabular_data(file_path: Path):
    """加载单个数据文件，默认前 4 列为输入，后 3 列为输出。"""
    if file_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(file_path)
    elif file_path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的数据文件格式: {file_path}")

    x_data = dataframe.iloc[:, :INPUT_DIMENSIONS].values
    y_data = dataframe.iloc[:, INPUT_DIMENSIONS : INPUT_DIMENSIONS + NUM_TASKS].values

    print(f"从 {file_path} 加载数据:")
    print(f"  输入维度: {x_data.shape}, 输出维度: {y_data.shape}")
    print(f"  输入范围: X[{x_data.min(axis=0)}, {x_data.max(axis=0)}]")
    print(f"  输出范围: Y[{y_data.min(axis=0)}, {y_data.max(axis=0)}]")

    return x_data, y_data


def _build_uniform_weights(targets: torch.Tensor, n_samples: int) -> torch.Tensor:
    """构造均匀样本权重，作为数值异常时的回退。"""
    num_samples = targets.size(0)
    if num_samples == 0:
        return torch.empty(0, device=targets.device, dtype=targets.dtype)
    return torch.ones(num_samples, device=targets.device, dtype=targets.dtype) * (
        n_samples / num_samples
    )


def compute_sample_weights(
    targets,
    target_values,
    sigma_values,
    n_samples,
    task_ids=None,
):
    """按任务分别计算高斯样本权重。"""
    task_ids = list(task_ids or TASK_IDS)
    target_tensor = torch.as_tensor(
        target_values, device=targets.device, dtype=targets.dtype
    )
    sigma_tensor = torch.as_tensor(
        sigma_values, device=targets.device, dtype=targets.dtype
    ).clamp_min(1e-8)

    if targets.size(-1) < len(task_ids):
        raise ValueError(
            f"targets 的任务维度不足: {targets.size(-1)} < {len(task_ids)}"
        )
    if target_tensor.numel() < len(task_ids):
        raise ValueError(
            f"target_values 长度不足: {target_tensor.numel()} < {len(task_ids)}"
        )
    if sigma_tensor.numel() < len(task_ids):
        raise ValueError(
            f"sigma_values 长度不足: {sigma_tensor.numel()} < {len(task_ids)}"
        )

    weights = {}
    for task_idx, task_id in enumerate(task_ids):
        distance = (targets[:, task_idx] - target_tensor[task_idx]).square()
        raw_weights = torch.exp(-distance / (2.0 * sigma_tensor[task_idx].square()))
        weight_sum = raw_weights.sum()

        if not torch.isfinite(raw_weights).all() or not torch.isfinite(weight_sum):
            weights[task_id] = _build_uniform_weights(targets, n_samples)
        elif weight_sum <= 1e-8:
            weights[task_id] = _build_uniform_weights(targets, n_samples)
        else:
            weights[task_id] = (raw_weights / weight_sum) * n_samples

    return weights


def _effective_sample_size(weights: torch.Tensor) -> float:
    if weights.numel() == 0:
        return 0.0
    return (
        (weights.sum() ** 2) / weights.square().sum().clamp_min(1e-8)
    ).item()


def _print_sample_weight_stats(split_name: str, sample_weights, split_size: int):
    print(f"  {split_name}:")
    for task_id, weights in sample_weights.items():
        ess = _effective_sample_size(weights)
        print(
            f"    {task_id} ESS: {ess:.2f} / {split_size}, "
            f"max: {weights.max().item():.4f}, min: {weights.min().item():.4f}"
        )


def setup_experiment_data(
    device: torch.device,
    normalize: bool = True,
):
    """加载并准备单次实验所需的训练/验证数据。"""
    print(f"\n=== 从目录 {DATA_DIR} 加载训练/验证数据 ===")
    x_train_np, y_train_np = load_tabular_data(resolve_split_path(TRAIN_SPLIT_NAME))
    x_val_np, y_val_np = load_tabular_data(resolve_split_path(VAL_SPLIT_NAME))

    x_train = torch.from_numpy(x_train_np).float().to(device)
    y_train = torch.from_numpy(y_train_np).float().to(device)
    x_val = torch.from_numpy(x_val_np).float().to(device)
    y_val = torch.from_numpy(y_val_np).float().to(device)

    normalization_params = None
    if normalize:
        x_min = x_train.amin(dim=0, keepdim=True)
        x_max = x_train.amax(dim=0, keepdim=True)
        y_mean = y_train.mean(dim=0, keepdim=True)
        y_std = y_train.std(dim=0, keepdim=True, unbiased=False)

        x_range = torch.where(x_max - x_min < 1e-8, torch.ones_like(x_max), x_max - x_min)
        y_std = torch.where(y_std < 1e-8, torch.ones_like(y_std), y_std)

        normalization_params = {
            "x_min": x_min,
            "x_max": x_max,
            "x_range": x_range,
            "y_mean": y_mean,
            "y_std": y_std,
        }

        x_train = (x_train - x_min) / x_range
        y_train = (y_train - y_mean) / y_std
        x_val = (x_val - x_min) / x_range
        y_val = (y_val - y_mean) / y_std

        print("\n数据预处理完成（X 使用最小最大归一化，Y 使用标准化）:")
        print(f"  X - 最小值: {x_min.cpu().numpy().flatten()}")
        print(f"  X - 最大值: {x_max.cpu().numpy().flatten()}")
        print(f"  Y - 均值: {y_mean.cpu().numpy().flatten()}")
        print(f"  Y - 标准差: {y_std.cpu().numpy().flatten()}")

    datasets = {
        "train": {"x": x_train, "y": y_train},
        "val": {"x": x_val, "y": y_val},
    }
    return datasets, normalization_params


def setup_sample_weights(
    train_data,
    val_data,
    target_values,
    sigma_values,
    normalization_params=None,
):
    """为训练集和验证集准备样本权重。"""
    if normalization_params is not None:
        y_mean = normalization_params["y_mean"].squeeze()
        y_std = normalization_params["y_std"].squeeze()

        normalized_target_values = tuple(
            (target_values[idx] - y_mean[idx].item()) / y_std[idx].item()
            for idx in range(len(target_values))
        )
        normalized_sigma_values = tuple(
            sigma_values[idx] / y_std[idx].item()
            for idx in range(len(sigma_values))
        )

        print("目标值标准化:")
        print(f"  原始目标值: {target_values}")
        print(f"  标准化后目标值: {normalized_target_values}")
        print(f"  原始 sigma 值: {sigma_values}")
        print(f"  标准化后 sigma 值: {normalized_sigma_values}")

        target_values = normalized_target_values
        sigma_values = normalized_sigma_values

    sample_weights_train = compute_sample_weights(
        train_data["y"],
        target_values,
        sigma_values,
        train_data["y"].size(0),
    )
    sample_weights_val = compute_sample_weights(
        val_data["y"],
        target_values,
        sigma_values,
        val_data["y"].size(0),
    )

    print("按任务高斯样本注意力配置:")
    print(f"  target_values: {target_values}")
    print(f"  sigma_values: {sigma_values}")
    print("  验证/Meta 使用样本注意力: True")
    _print_sample_weight_stats("训练集", sample_weights_train, train_data["y"].size(0))
    _print_sample_weight_stats("验证集", sample_weights_val, val_data["y"].size(0))

    return {
        "train": sample_weights_train,
        "val": sample_weights_val,
    }
