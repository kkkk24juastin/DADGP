# -*- coding: utf-8 -*-
"""
评估指标模块：包含模型性能评估相关的函数

本文件提供：
- evaluate_metrics：评估局部区域的RMSE、NLPD、质量损失指标
"""

import torch
import numpy as np
from typing import Dict, Tuple

from config import LOCAL_THRESHOLD


# ==========================================================================================
# 模型评估指标
# ==========================================================================================

def build_local_mask(
    test_y: torch.Tensor,
    target_values: Tuple[float, float, float],
    local_threshold: float = LOCAL_THRESHOLD,
) -> torch.Tensor:
    """根据真实输出筛选局部测试区域。"""
    target_y1, target_y2, target_y3 = target_values
    return (
        (torch.abs(test_y[:, 0] - target_y1) <= local_threshold)
        & (torch.abs(test_y[:, 1] - target_y2) <= local_threshold)
        & (torch.abs(test_y[:, 2] - target_y3) <= local_threshold)
    )


def select_local_data(
    data: Dict[str, torch.Tensor],
    target_values: Tuple[float, float, float],
    local_threshold: float = LOCAL_THRESHOLD,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """先筛选局部测试点，避免对全量测试集做无用预测。"""
    local_mask = build_local_mask(data["y"], target_values, local_threshold)
    return {
        "x": data["x"][local_mask],
        "y": data["y"][local_mask],
    }, local_mask


def evaluate_metrics(
    mean_pred: torch.Tensor,
    var_pred: torch.Tensor,
    test_y: torch.Tensor,
    target_values: Tuple[float, float, float],
    local_threshold: float = LOCAL_THRESHOLD
) -> Dict[str, float]:
    """评估局部区域的三个核心指标：RMSE、NLPD、质量损失。

    该函数只计算多任务模型在局部区域的性能指标：
    - RMSE（均方根误差）：衡量预测精度
    - NLPD（负对数预测密度）：衡量预测不确定性
    - 质量损失（Quality Loss）：综合误差和不确定性的指标

    Args:
        mean_pred: 预测均值，形状为[n_samples, num_tasks]
        var_pred: 预测方差，形状为[n_samples, num_tasks]
        test_y: 测试集真实值，形状为[n_samples, num_tasks]
        target_values: 目标值元组 (target_y1, target_y2, target_y3)
        local_threshold: 局部区域阈值，定义"局部"的范围

    Returns:
        结果字典，包含：
            - 局部指标: "local_rmse_task1", "local_nlpd_task1", "local_quality_loss_task1" 等

    数学公式:
        RMSE = sqrt(mean((y_pred - y_true)^2))
        NLPD = 0.5 * log(2π * σ^2) + (y - μ)^2 / (2σ^2)
        Quality Loss = (y_true - y_pred)^2 + var_pred
    """
    num_tasks = test_y.size(-1)  # 获取任务数量
    results = {}  # 初始化结果字典

    # 创建局部区域掩码。若调用方已预筛选局部点，这里会保持等价。
    local_mask = build_local_mask(test_y, target_values, local_threshold)

    # 对每个任务计算指标
    for d in range(num_tasks):
        task_num = d + 1

        # 提取当前任务的预测和真实值
        y_true = test_y[:, d]
        y_pred = mean_pred[:, d]
        var = var_pred[:, d]

        if local_mask.sum() > 0:
            y_true_local = y_true[local_mask]
            y_pred_local = y_pred[local_mask]
            var_local = var[local_mask]

            # 1. RMSE
            rmse_local = torch.sqrt(torch.mean((y_pred_local - y_true_local) ** 2))
            results[f"local_rmse_task{task_num}"] = rmse_local.item()

            # 2. NLPD
            nlpd_local = 0.5 * torch.log(2 * np.pi * var_local) + (
                y_true_local - y_pred_local
            ) ** 2 / (2 * var_local)
            results[f"local_nlpd_task{task_num}"] = torch.mean(nlpd_local).item()

            # 3. 质量损失
            quality_loss_local = (y_true_local - y_pred_local) ** 2 + var_local
            results[f"local_quality_loss_task{task_num}"] = torch.mean(
                quality_loss_local
            ).item()
        else:
            # 如果局部区域内没有样本，设为NaN
            results[f"local_rmse_task{task_num}"] = float("nan")
            results[f"local_nlpd_task{task_num}"] = float("nan")
            results[f"local_quality_loss_task{task_num}"] = float("nan")

    return results
