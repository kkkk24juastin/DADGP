# -*- coding: utf-8 -*-
"""
绘制三个任务的局部指标箱线图（按方法和指标分组）

从 Achievements 目录读取多次运行的结果并汇总。
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from config import (
    BASE_DIR, CN_FONTS,
    FONT_SIZE_LABEL, FONT_SIZE_TICK_X, FONT_SIZE_TICK_Y
)


# ==========================================================================================
# 输出目录设置
# ==========================================================================================

outdir = BASE_DIR / "fig" / "box"
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

# Seaborn 风格
sns.set(style="whitegrid", font_scale=1.4)


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
    "#F8766D",  # da_dgp (PM) - 红色
    "#C49A00",  # EQUAL - 金黄色
    "#A58AFF",  # Pure DGP - 紫色
    "#53B400",  # DWA - 绿色
    "#00C1E4",  # UW - 青色
    "#FB61D7",  # MGDA - 粉色
    "#1F77B4",  # Indep-DGP - 蓝色
    "#FF7F0E",  # Indep-HetGP - 橙色
    "#2CA02C",  # LMC-DGP - 深绿色
    "#7F7F7F",  # ablation_no_sample_attn (T-Atten) - 灰色
]
palette_map = dict(zip(methods, palette))

method_display_map = {
    "da_dgp": "PM",
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

# 散点显示开关
SHOW_SCATTER = False


# ==========================================================================================
# 数据读取与汇总
# ==========================================================================================

def load_all_results():
    """从 Achievements 目录加载所有运行的局部指标结果并汇总。"""
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


df = load_all_results()
if df.empty:
    print("未找到可绘制的局部指标数据。")
else:
    print(f"已加载 {len(df)} 条记录，来自 {df['run'].nunique()} 次运行")


# ==========================================================================================
# 绘图函数
# ==========================================================================================

METRIC_YLABEL_MAP = {
    "rmse": "RMSE",
    "nlpd": "NLPD",
    "quality_loss": "QL",
}


def make_box_pdf(dataframe, metric: str, task_id: int, outfile: str):
    """绘制箱线图。

    Args:
        dataframe: 数据DataFrame (包含 run, method, metric, task, value 列)
        metric: 指标名称（rmse, nlpd, quality_loss）
        task_id: 任务编号（1, 2, 3）
        outfile: 输出文件路径
    """
    # 筛选数据
    sub_df = dataframe[
        (dataframe["metric"] == metric) &
        (dataframe["task"] == task_id) &
        (dataframe["method"].isin(methods))
    ].copy()

    if sub_df.empty:
        print(f"Warning: No data for local_{metric}_task{task_id}, skipping...")
        return

    # 映射方法显示名称
    sub_df["method_display"] = sub_df["method"].map(method_display_map)
    display_methods = [method_display_map[m] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))

    # 箱线图
    sns.boxplot(
        x="method_display",
        y="value",
        data=sub_df,
        order=display_methods,
        palette=[palette_map[m] for m in methods],
        width=0.6,
        fliersize=0,
        ax=ax,
    )

    # 叠加散点
    if SHOW_SCATTER:
        sns.stripplot(
            x="method_display",
            y="value",
            data=sub_df,
            order=display_methods,
            palette=[palette_map[m] for m in methods],
            size=5.5,
            alpha=0.6,
            jitter=0.15,
            ax=ax,
        )

    ax.set_xlabel("Method", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(METRIC_YLABEL_MAP[metric], fontsize=FONT_SIZE_LABEL)

    ax.tick_params(axis="x", labelrotation=30, labelsize=FONT_SIZE_TICK_X)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_Y)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================================
# 生成图表
# ==========================================================================================

pdf_count = 0
if not df.empty:
    for task_id in range(1, 4):
        for metric in metrics:
            fname = f"task{task_id}_local_{metric}_box.pdf"
            make_box_pdf(df, metric, task_id, outdir / fname)
            pdf_count += 1

print(f"Done: {pdf_count} local PDF files have been generated in the ./{outdir} directory.")
