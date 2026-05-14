# -*- coding: utf-8 -*-
"""
绘制DA-DGP训练过程中的任务权重变化轨迹

从 Achievements 目录读取权重历史数据。
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    BASE_DIR,
    PLOT_FONT_SIZE, PLOT_LEGEND_FONT_SIZE
)


# ==========================================================================================
# 字体配置
# ==========================================================================================

plt.rcParams.update({"font.size": PLOT_FONT_SIZE})


# ==========================================================================================
# 输出目录设置
# ==========================================================================================

out_dir = BASE_DIR / "fig" / "weight"
out_dir.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 数据目录
# ==========================================================================================

achievements_dir = BASE_DIR / "Achievements"


# ==========================================================================================
# 目标运行配置
# ==========================================================================================

target_runs = [10, 11]  # 指定要绘制的运行编号


# ==========================================================================================
# 数据读取与汇总
# ==========================================================================================

def load_weight_history(run_id):
    """从指定运行目录读取权重历史数据。"""
    run_dir = achievements_dir / f"{run_id:02d}"
    weight_path = run_dir / "weight_history_da_dgp.xlsx"

    if not weight_path.exists():
        print(f"警告：run={run_id:02d} 的权重历史文件不存在")
        return None

    df = pd.read_excel(weight_path)

    # 检查必要列
    required_cols = {"step", "epoch", "batch", "weight_local_A", "weight_local_B", "weight_local_C"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"警告：run={run_id:02d} 缺少列: {missing}")
        return None

    return df


# ==========================================================================================
# 绘制权重轨迹
# ==========================================================================================

for run_id in target_runs:
    df = load_weight_history(run_id)
    if df is None:
        continue

    # 排序
    df = df.sort_values("step", kind="mergesort")

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(df["step"], df["weight_local_A"], label="Task1", linewidth=1.5)
    ax.plot(df["step"], df["weight_local_B"], label="Task2", linewidth=1.5)
    ax.plot(df["step"], df["weight_local_C"], label="Task3", linewidth=1.5)

    ax.grid(False)

    ax.set_xlabel("batch")
    ax.set_ylabel("weight")

    ax.legend(loc="upper right", frameon=True, fontsize=PLOT_LEGEND_FONT_SIZE)

    save_path = out_dir / f"run_{run_id:02d}.pdf"
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

print(f"完成：已将指定 run 的 PDF 图片保存到 {out_dir.resolve()}")