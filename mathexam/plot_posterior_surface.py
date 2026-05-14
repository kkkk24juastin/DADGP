# -*- coding: utf-8 -*-
"""
绘制 DA-DGP 训练完成后的多任务 DGP 模型在选定二维输入平面上的后验均值与标准差响应曲面。

使用重构后的模块结构。
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import (
    BASE_DIR, CN_FONTS, TASK_LABELS, TRAIN_TASKS, PRI_TASKS,
    TARGET_VALUES, SIGMA_VALUES, NUM_EPOCHS, NUM_HIDDEN_DGP_DIMS,
    GRID_POINTS_DEFAULT, VARY_DIMS_DEFAULT, PREDICT_BATCH_SIZE_SURFACE,
    VAR_MIN, WEIGHT_INIT
)
from models_dgp import MultitaskDeepGP
from data_generation import setup_experiment_data, setup_sample_weights
from common import create_loss_function
from algo_da_dgp import DADGP, run_training_loop


# ==========================================================================================
# 参数解析
# ==========================================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="训练 AutoLambda 并绘制多任务 DGP 的后验均值/标准差响应曲面（二维输入平面）。"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=NUM_EPOCHS,
        help="训练轮数。"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs=3,
        default=(400, 400, 5000),
        metavar=("N_TRAIN", "N_VAL", "N_TEST"),
        help="训练/验证/测试的样本数量。",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=GRID_POINTS_DEFAULT,
        help="每个维度的网格离散点数。",
    )
    parser.add_argument(
        "--vary-dims",
        type=int,
        nargs=2,
        default=VARY_DIMS_DEFAULT,
        metavar=("DIM_I", "DIM_J"),
        help="设置响应曲面沿哪两个输入维度展开。",
    )
    parser.add_argument(
        "--fixed-values",
        type=float,
        nargs="*",
        default=None,
        help="为未被绘制的维度提供固定值。",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        default=(-1.0, 1.0),
        metavar=("LOW", "HIGH"),
        help="输入变量的全局取值范围。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fig/posterior_surface",
        help="图像输出目录。",
    )
    parser.add_argument(
        "--predict-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "train"],
        help="响应曲面推理阶段使用的设备。",
    )
    parser.add_argument(
        "--predict-batch",
        type=int,
        default=PREDICT_BATCH_SIZE_SURFACE,
        help="预测阶段的批大小。",
    )
    return parser.parse_args()


# ==========================================================================================
# 辅助函数
# ==========================================================================================

def configure_font():
    """配置中文字体。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in CN_FONTS:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def ensure_fixed_values(raw_values: Sequence[float], dimensions: int) -> np.ndarray:
    """生成长度与输入维度一致的固定取值向量。"""
    if raw_values is None or len(raw_values) == 0:
        base = np.zeros(dimensions, dtype=np.float32)
    else:
        base = np.zeros(dimensions, dtype=np.float32)
        usable = list(raw_values[:dimensions])
        usable.extend([0.0] * (dimensions - len(usable)))
        base[:] = np.array(usable, dtype=np.float32)
    return base


def train_da_dgp_model(
    device: torch.device,
    samples: Tuple[int, int, int],
    num_epochs: int,
    target_values: Tuple[float, float, float],
    sigma_values: Tuple[float, float, float],
):
    """训练DA-DGP模型。"""
    print(f"正在生成训练/验证/测试数据（样本量: {samples}）...")
    datasets = setup_experiment_data(device, samples=samples)
    train_data, val_data = datasets["train"], datasets["val"]
    num_tasks = train_data["y"].size(-1)

    print("正在构建样本权重...")
    sample_weights = setup_sample_weights(train_data, val_data, target_values, sigma_values)

    print("启动 DA-DGP 训练循环...")
    model = MultitaskDeepGP(
        train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
    ).to(device)
    dadgp = DADGP(
        model, device, TRAIN_TASKS, PRI_TASKS,
        weight_init=WEIGHT_INIT, sample_weights=sample_weights
    )
    dadgp.loss_fn = create_loss_function(
        sample_weights=sample_weights,
        split_sizes={
            "train": train_data["y"].size(0),
            "val": val_data["y"].size(0),
        },
    )
    history = run_training_loop(model, dadgp, datasets, num_epochs)

    final_weights = dadgp.get_normalized_weights().detach().cpu().numpy()
    print(f"训练完成，最终任务权重（softmax）: {final_weights}")
    return model, datasets, history


def resolve_predict_device(option: str, train_device: torch.device) -> torch.device:
    """选择预测阶段的设备。"""
    option = option.lower()
    if option == "train":
        return train_device
    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print("未检测到可用 CUDA，预测阶段回退至 CPU。")
        return torch.device("cpu")
    return torch.device("cpu")


def build_surface_grid(
    vary_dims: Tuple[int, int],
    grid_points: int,
    dimensions: int,
    bounds: Tuple[float, float],
    fixed_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建二维网格采样点。"""
    lo, hi = bounds
    dim_i, dim_j = vary_dims
    axis = np.linspace(lo, hi, grid_points, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis)

    flat = np.tile(fixed_values, (grid_points * grid_points, 1))
    flat[:, dim_i] = grid_x.ravel()
    flat[:, dim_j] = grid_y.ravel()
    return flat.astype(np.float32), grid_x, grid_y


def evaluate_posterior_surfaces(model, grid_points, device, batch_size):
    """在网格上执行预测。"""
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid_points).to(device)
        mean_pred, var_pred = model.predict(grid_tensor, batch_size=batch_size)
        mean_np = mean_pred.cpu().numpy()
        std_np = torch.sqrt(var_pred.clamp_min(VAR_MIN)).cpu().numpy()
    return mean_np, std_np


def plot_task_surfaces(
    mesh_x, mesh_y, mean_surface, std_surface,
    task_label, vary_dims, save_path
):
    """绘制单个任务的后验曲面。"""
    fig = plt.figure(figsize=(12, 5))
    dim_i, dim_j = vary_dims

    ax_mean = fig.add_subplot(1, 2, 1, projection="3d")
    surf_mean = ax_mean.plot_surface(
        mesh_x, mesh_y, mean_surface, cmap="viridis",
        linewidth=0, antialiased=True
    )
    ax_mean.set_title(f"{task_label} 后验均值")
    ax_mean.set_xlabel(f"X{dim_i + 1}")
    ax_mean.set_ylabel(f"X{dim_j + 1}")
    ax_mean.set_zlabel("μ")
    fig.colorbar(surf_mean, ax=ax_mean, shrink=0.6, aspect=12)

    ax_std = fig.add_subplot(1, 2, 2, projection="3d")
    surf_std = ax_std.plot_surface(
        mesh_x, mesh_y, std_surface, cmap="magma",
        linewidth=0, antialiased=True
    )
    ax_std.set_title(f"{task_label} 后验标准差")
    ax_std.set_xlabel(f"X{dim_i + 1}")
    ax_std.set_ylabel(f"X{dim_j + 1}")
    ax_std.set_zlabel("σ")
    fig.colorbar(surf_std, ax=ax_std, shrink=0.6, aspect=12)

    fig.suptitle(f"{task_label} Posterior Response Surfaces", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================================
# 主函数
# ==========================================================================================

def main():
    args = parse_args()
    configure_font()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备：{device}")

    samples = tuple(args.samples)
    bounds = tuple(args.bounds)
    vary_dims = tuple(args.vary_dims)

    model, datasets, _ = train_da_dgp_model(
        device=device,
        samples=samples,
        num_epochs=args.num_epochs,
        target_values=TARGET_VALUES,
        sigma_values=SIGMA_VALUES,
    )

    dimensions = datasets["train"]["x"].shape[-1]
    fixed_values = ensure_fixed_values(args.fixed_values, dimensions)
    grid_flat, grid_x, grid_y = build_surface_grid(
        vary_dims, args.grid_points, dimensions, bounds, fixed_values
    )

    predict_device = resolve_predict_device(args.predict_device, device)
    if predict_device != device:
        print(f"正在将模型从 {device} 迁移到 {predict_device} 以执行推理...")
        model = model.to(predict_device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("正在计算响应曲面...")
    try:
        mean_np, std_np = evaluate_posterior_surfaces(
            model, grid_flat, predict_device, args.predict_batch
        )
    except RuntimeError as err:
        if predict_device.type == "cuda" and "out of memory" in str(err).lower():
            print("GPU 推理显存不足，自动切换至 CPU 重新计算。")
            torch.cuda.empty_cache()
            predict_device = torch.device("cpu")
            model = model.to(predict_device)
            mean_np, std_np = evaluate_posterior_surfaces(
                model, grid_flat, predict_device, args.predict_batch
            )
        else:
            raise

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_idx, task_label in enumerate(TASK_LABELS):
        mean_surface = mean_np[:, task_idx].reshape(args.grid_points, args.grid_points)
        std_surface = std_np[:, task_idx].reshape(args.grid_points, args.grid_points)
        save_path = out_dir / f"task{task_idx + 1}_posterior_surface.png"
        plot_task_surfaces(
            grid_x, grid_y, mean_surface, std_surface,
            task_label, vary_dims, save_path
        )
        print(f"已保存 {task_label} 曲面图 -> {save_path}")

    print("全部曲面绘制完成。")


if __name__ == "__main__":
    main()
