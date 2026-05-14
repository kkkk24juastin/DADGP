# -*- coding: utf-8 -*-
"""
绘制三个任务的指标均值误差棒图。

按任务分组，每个任务显示三个指标（rmse, nlpd, quality_loss）。
横轴为方法，纵轴为跨 run 的均值，误差棒默认表示 95% 置信区间。
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager

from config import (
    BASE_DIR, CN_FONTS,
    FONT_SIZE_LABEL, FONT_SIZE_TICK_X, FONT_SIZE_TICK_Y, FONT_SIZE_LEGEND
)


# ==========================================================================================
# 输出目录设置
# ==========================================================================================

outdir = BASE_DIR / "fig" / "metrics"
outdir.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 数据目录
# ==========================================================================================

achievements_dir = BASE_DIR / "Achievements"


# ==========================================================================================
# 字体配置
# ==========================================================================================

available = [f.name for f in font_manager.fontManager.ttflist]
for f in CN_FONTS:
    if f in available:
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

sns.set_theme(style="whitegrid", font_scale=1.4)


# ==========================================================================================
# 方法配置（包含消融实验）
# ==========================================================================================

methods = [
    "da_dgp",
    "baseline_equal",
    "baseline_pure_dgp",
    "baseline_dwa",
    "baseline_uw",
    "baseline_mgda",
    "baseline_indep_dgp",
    "baseline_indep_hetgp",
    "baseline_lmc_dgp",
    "ablation_no_sample_attn",
]
palette = [
    "#F8766D",  # da_dgp (DA-DGP) - 红色
    "#C49A00",  # EQUAL - 金黄色
    "#A58AFF",  # Pure DGP - 紫色
    "#53B400",  # DWA - 绿色
    "#00C1E4",  # UW - 青色
    "#FB61D7",  # MGDA - 粉色
    "#1F77B4",  # Indep-DGP - 蓝色
    "#FF7F0E",  # Indep-HetGP - 橙色
    "#2CA02C",  # LMC-DGP - 深绿色
    "#7F7F7F",  # T-Atten - 灰色
]
palette_map = dict(zip(methods, palette))

method_display_map = {
    "da_dgp": "DA-DGP",
    "baseline_equal": "EQUAL",
    "baseline_pure_dgp": "Pure DGP",
    "baseline_dwa": "DWA",
    "baseline_uw": "UW",
    "baseline_mgda": "MGDA",
    "baseline_indep_dgp": "Indep-DGP",
    "baseline_indep_hetgp": "Indep-HetGP",
    "baseline_lmc_dgp": "LMC-DGP",
    "ablation_no_sample_attn": "T-Atten",
}

metrics = ["rmse", "nlpd", "quality_loss"]
CI_Z = 1.96

# 误差棒类型：可选 "std" 或 "ci95"。
ERROR_BAR_MODE = "ci95"
ERROR_BAR_LABEL = {
    "std": "Mean ± SD",
    "ci95": "Mean ± 95% CI",
}

# 标记样式
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# 指标名称映射
METRIC_YLABEL_MAP = {
    "rmse": "RMSE",
    "nlpd": "NLPD",
    "quality_loss": "QL",
}


def summarize_values(values):
    """计算均值、样本标准差和95%置信区间半宽。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    count = int(arr.size)
    if count == 0:
        return {"mean": np.nan, "std": np.nan, "ci95": np.nan, "n": 0}

    mean_value = float(np.mean(arr))
    std_value = float(np.std(arr, ddof=1)) if count > 1 else np.nan
    ci95 = float(CI_Z * std_value / np.sqrt(count)) if count > 1 else np.nan
    return {"mean": mean_value, "std": std_value, "ci95": ci95, "n": count}


# ==========================================================================================
# 数据读取与汇总
# ==========================================================================================

def load_all_local_results():
    """从 Achievements 目录加载所有运行的 local_metrics 结果并汇总。"""
    all_data = []

    run_dirs = sorted(achievements_dir.glob("*"))
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        local_path = run_dir / "local_metrics.xlsx"

        if not run_id.isdigit() or not local_path.exists():
            continue

        for metric in metrics:
            try:
                df_local = pd.read_excel(local_path, sheet_name=metric)
                for method in methods:
                    if method not in df_local.columns:
                        continue
                    for task_idx in range(3):
                        task_name = f"task{task_idx + 1}"
                        row = df_local[df_local["Unnamed: 0"] == task_name]
                        if not row.empty:
                            value = row[method].values[0]
                            all_data.append({
                                "run": int(run_id),
                                "method": method,
                                "metric": metric,
                                "task": task_idx + 1,
                                "value": value,
                            })
            except Exception as e:
                print(f"Warning: Failed to read {local_path} sheet {metric}: {e}")

    return pd.DataFrame(all_data)


df_all = load_all_local_results()
if df_all.empty:
    print("未找到可绘制的局部指标数据。")
else:
    print(f"已加载 {len(df_all)} 条记录，来自 {df_all['run'].nunique()} 次运行")


# ==========================================================================================
# 计算统计值
# ==========================================================================================

def compute_statistics(df):
    """计算每个方法和任务在每个指标上的均值、标准差和置信区间。"""
    if df.empty or not {"method", "task", "metric", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["method", "task", "metric", "mean", "std", "ci95", "n"])

    stats_data = []

    for method in methods:
        for task in [1, 2, 3]:
            for metric in metrics:
                sub_df = df[
                    (df["method"] == method) &
                    (df["task"] == task) &
                    (df["metric"] == metric)
                ]
                if sub_df.empty:
                    continue

                summary = summarize_values(sub_df["value"].values)
                stats_data.append({
                    "method": method,
                    "task": task,
                    "metric": metric,
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "ci95": summary["ci95"],
                    "n": summary["n"],
                })

    return pd.DataFrame(stats_data)


df_stats = compute_statistics(df_all)


# ==========================================================================================
# 绘制图表
# ==========================================================================================

pdf_count = 0
if not df_stats.empty:
    for task_id in [1, 2, 3]:
        fig, axes = plt.subplots(1, 3, figsize=(24, 5))

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]

            # 筛选当前任务和指标的数据
            task_metric_stats = df_stats[
                (df_stats["task"] == task_id) &
                (df_stats["metric"] == metric)
            ]

            x_positions = np.arange(len(methods))
            for method_idx, method in enumerate(methods):
                method_stats = task_metric_stats[
                    task_metric_stats["method"] == method
                ]

                if not method_stats.empty:
                    row = method_stats.iloc[0]
                    error_value = row[ERROR_BAR_MODE]
                    if not np.isfinite(error_value):
                        error_value = 0.0

                    ax.errorbar(
                        method_idx,
                        row["mean"],
                        yerr=error_value,
                        fmt=MARKERS[method_idx] if method_idx < len(MARKERS) else "o",
                        color=palette[method_idx],
                        ecolor=palette[method_idx],
                        label=method_display_map[method],
                        elinewidth=2.2,
                        capsize=4,
                        capthick=1.8,
                        markersize=9,
                        alpha=0.9,
                    )

            ax.set_xlabel(ERROR_BAR_LABEL[ERROR_BAR_MODE], fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel(METRIC_YLABEL_MAP[metric], fontsize=FONT_SIZE_LABEL)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [method_display_map[method] for method in methods],
                rotation=35,
                ha="right",
                fontsize=FONT_SIZE_TICK_X,
            )
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_Y)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_linewidth(2.0)
            ax.spines["bottom"].set_linewidth(2.0)

            ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.5)

        axes[-1].legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=FONT_SIZE_LEGEND,
            framealpha=0.95,
            edgecolor="black",
            fancybox=True,
        )

        plt.tight_layout()

        output_filename = f"task{task_id}_metrics.pdf"
        output_file = outdir / output_filename
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        pdf_count += 1
        print(f"[{pdf_count}/3] 图片已保存: {output_file}")

        plt.close(fig)

print(f"\n完成！共生成 {pdf_count} 个PDF文件，保存在 '{outdir}' 目录中。")
