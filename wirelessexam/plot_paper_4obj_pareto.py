# -*- coding: utf-8 -*-
"""
为稳健四目标 Pareto 分析生成论文用独立图件。

本脚本有意避免雷达图、柱状图、棒棒糖图以及基于折线的平行坐标图。输出重点
放在 Pareto 几何结构、目标空间嵌入、attainment map、设计空间映射和不确定性
结构上。

输入:
    result/robust_simulation_<method>.xlsx, sheet all_candidates
    result/robust_pareto_13methods_indicators.xlsx

输出:
    fig/paper_4obj_pareto/*.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp") / "wireless-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
FIG_DIR = BASE_DIR / "fig" / "paper_4obj_pareto"

INPUT_PREFIX = "robust_simulation_"
INPUT_SHEET_NAME = "all_candidates"
INDICATOR_FILE = RESULT_DIR / "robust_pareto_13methods_indicators.xlsx"
INDICATOR_SHEET_NAME = "indicators"
TOPSIS_SHEET_NAME = "topsis"

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 128.0
FIGURE_SIZE = (FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
EXPORT_DPI = 600
PREVIEW_DPI = 300
OUTPUT_FORMAT = ".pdf"
CLEANUP_FORMATS = (".svg", ".pdf", ".tiff", ".png")
GENERATED_FIGURE_STEMS = (
    "01_metric_manifold_embedding",
    "02_objective_manifold_embedding",
    "03_global_front_3d_projection",
    "04_attainment_probability_hexbin",
    "05_design_space_pareto_map",
    "06_uncertainty_risk_map",
    "07_pareto_projection_density",
    "08_four_objective_bubble",
)

FONT_SIZE = 7.0
TITLE_SIZE = 8.2
LABEL_SIZE = 7.2
TICK_SIZE = 6.2
LEGEND_SIZE = 6.0
COLORBAR_SIZE = 6.4

AXIS_COLOR = "#2B2B2B"
GRID_COLOR = "#D6D6D6"
BACKGROUND_POINT_COLOR = "#C9C9C9"
GLOBAL_FRONT_EDGE = "#272727"
CONTEXT_FRONT_COLOR = "#A9A9A9"
DADGP_EDGE_COLOR = "#1F1F1F"

POWER_MODEL_P_BB = 0.2
POWER_MODEL_P_RF = 0.8
POWER_MODEL_ETA_PA = 0.35
MIN_THROUGHPUT_MBPS = 1e-6

FOUR_OBJECTIVES = [
    ("throughput_mbps", "Throughput", "max"),
    ("ber", "BER", "min"),
    ("papr_db", "PAPR", "min"),
    ("energy_efficiency", "Energy Efficiency", "max"),
]
OBJECTIVE_COLUMNS = [column_name for column_name, _, _ in FOUR_OBJECTIVES]

METHOD_LABELS = {
    "dadgp": "DADGP",
    "baseline_equal": "Equal",
    "baseline_pure_dgp": "Pure DGP",
    "baseline_dwa": "DWA",
    "baseline_uw": "UW",
    "baseline_mgda": "MGDA",
    "baseline_indep_dgp": "Indep-DGP",
    "baseline_indep_hetgp": "Indep-HetGP",
    "baseline_lmc_dgp": "LMC-DGP",
    "ablation_no_sample_attn": "No Sample Attn",
    "bo_qehvi": "BO-qEHVI",
    "bo_qnehvi": "BO-qNEHVI",
    "bo_qparego": "BO-qParEGO",
}

METHOD_ORDER = [
    "dadgp",
    "baseline_equal",
    "baseline_pure_dgp",
    "baseline_dwa",
    "baseline_uw",
    "baseline_mgda",
    "baseline_indep_dgp",
    "baseline_indep_hetgp",
    "baseline_lmc_dgp",
    "ablation_no_sample_attn",
    "bo_qehvi",
    "bo_qnehvi",
    "bo_qparego",
]

REPRESENTATIVE_METHODS = [
    "dadgp",
    "baseline_equal",
    "bo_qnehvi",
    "baseline_mgda",
    "baseline_dwa",
]

METHOD_COLORS = {
    "dadgp": "#B64342",
    "baseline_equal": "#484878",
    "baseline_pure_dgp": "#A8A8A8",
    "baseline_dwa": "#7884B4",
    "baseline_uw": "#9A7FA8",
    "baseline_mgda": "#42949E",
    "baseline_indep_dgp": "#7DA7A1",
    "baseline_indep_hetgp": "#8C7A6B",
    "baseline_lmc_dgp": "#B8A15B",
    "ablation_no_sample_attn": "#606060",
    "bo_qehvi": "#7C6CCF",
    "bo_qnehvi": "#5B8FD6",
    "bo_qparego": "#D08A55",
}

REQUIRED_CANDIDATE_COLUMNS = [
    "method",
    "moo_run",
    "solution_idx",
    "x1",
    "x2",
    "x3",
    "x4",
    "robust_mean_throughput_mbps",
    "robust_mean_ber",
    "robust_mean_papr_db",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "robust_var_throughput_mbps",
    "robust_var_ber",
    "robust_var_papr_db",
    "pred_var_task1",
    "pred_var_task2",
    "pred_var_task3",
    "quality_loss_task1",
    "quality_loss_task2",
    "quality_loss_task3",
    "moo_objective_task1",
    "moo_objective_task2",
    "moo_objective_task3",
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.linewidth": 0.55,
            "axes.edgecolor": AXIS_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": PREVIEW_DPI,
            "savefig.dpi": EXPORT_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label, fontsize=COLORBAR_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_SIZE, width=0.45, length=2.4)
    colorbar.outline.set_linewidth(0.45)


def add_compact_legend(
    ax: plt.Axes,
    loc: str = "best",
    ncol: int = 1,
    title: str | None = None,
):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        title=title,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.28,
        borderaxespad=0.25,
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    return legend


def format_millions(value: float) -> str:
    return f"{value / 1e6:.2f} x 10^6"


configure_matplotlib()


def method_label(method_name: str) -> str:
    return METHOD_LABELS.get(method_name, method_name)


def method_from_label(label: str) -> str:
    for method_name, method_label_value in METHOD_LABELS.items():
        if method_label_value == label:
            return method_name
    return label


def method_sort_key(method_name: str) -> int:
    if method_name in METHOD_ORDER:
        return METHOD_ORDER.index(method_name)
    return len(METHOD_ORDER)


def discover_methods() -> list[str]:
    discovered = [
        path.stem.replace(INPUT_PREFIX, "")
        for path in sorted(RESULT_DIR.glob(f"{INPUT_PREFIX}*.xlsx"))
    ]
    available = set(discovered)
    return [method for method in METHOD_ORDER if method in available]


def representative_methods(methods: list[str]) -> list[str]:
    available = set(methods)
    return [method for method in REPRESENTATIVE_METHODS if method in available]


def color_for_method(method_name: str, default: str = "#8F8F8F") -> str:
    return METHOD_COLORS.get(method_name, default)


def focus_methods(methods: list[str]) -> list[str]:
    available = set(methods)
    return [method for method in REPRESENTATIVE_METHODS if method in available]


def add_panel_note(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    if not text:
        return

    positions = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, horizontal_alignment, vertical_alignment = positions[loc]
    text_kwargs = {
        "transform": ax.transAxes,
        "ha": horizontal_alignment,
        "va": vertical_alignment,
        "fontsize": 6.3,
        "color": AXIS_COLOR,
        "bbox": {
            "facecolor": "white",
            "edgecolor": "#CFCFCF",
            "linewidth": 0.35,
            "alpha": 0.9,
            "pad": 2.2,
        },
        "zorder": 30,
    }
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, text, **text_kwargs)
    else:
        ax.text(x, y, text, **text_kwargs)


def build_dadgp_summary_note(indicators: pd.DataFrame, topsis: pd.DataFrame) -> str:
    indicator_row = indicators[indicators["method_key"] == "dadgp"]
    topsis_row = topsis[topsis["method_key"] == "dadgp"]
    if indicator_row.empty or topsis_row.empty:
        return ""

    indicator_values = indicator_row.iloc[0]
    topsis_values = topsis_row.iloc[0]
    return (
        f"DADGP: TOPSIS rank {int(topsis_values['Rank'])}, "
        f"Ci={topsis_values['Ci']:.3f}\n"
        f"HV ratio={indicator_values['HV Ratio']:.3f}; "
        f"global PF={indicator_values['#Global Pareto']:.1f}/run"
    )


def ensure_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def read_indicator_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INDICATOR_FILE.exists():
        raise FileNotFoundError(f"Missing indicator workbook: {INDICATOR_FILE}")

    indicators = pd.read_excel(INDICATOR_FILE, sheet_name=INDICATOR_SHEET_NAME)
    topsis = pd.read_excel(INDICATOR_FILE, sheet_name=TOPSIS_SHEET_NAME)

    indicator_numeric_columns = [
        "#Pareto",
        "#Global Pareto",
        "Global Contrib.",
        "HV (norm)",
        "HV Ratio",
        "IGD",
        "IGD+",
        "GD",
        "GD+",
        "Spacing",
        "Closest Ideal",
    ]
    topsis_numeric_columns = ["D+", "D-", "Ci", "Rank"]

    indicators = ensure_numeric(indicators, indicator_numeric_columns)
    topsis = ensure_numeric(topsis, topsis_numeric_columns)
    indicators["method_key"] = indicators["Method"].map(method_from_label)
    topsis["method_key"] = topsis["Method"].map(method_from_label)
    return indicators, topsis


def load_method_candidates(method_name: str) -> pd.DataFrame:
    input_file = RESULT_DIR / f"{INPUT_PREFIX}{method_name}.xlsx"
    if not input_file.exists():
        raise FileNotFoundError(f"Missing robust simulation workbook: {input_file}")

    frame = pd.read_excel(input_file, sheet_name=INPUT_SHEET_NAME)
    missing_columns = [
        column for column in REQUIRED_CANDIDATE_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{input_file} is missing required columns: {', '.join(missing_columns)}"
        )

    numeric_columns = [
        "moo_run",
        "solution_idx",
        "x1",
        "x2",
        "x3",
        "x4",
        "robust_mean_throughput_mbps",
        "robust_mean_ber",
        "robust_mean_papr_db",
        *OPTIONAL_NUMERIC_COLUMNS,
    ]
    frame = ensure_numeric(frame, numeric_columns)
    frame["method"] = method_name
    frame = frame.dropna(
        subset=[
            "x1",
            "x2",
            "x3",
            "x4",
            "robust_mean_throughput_mbps",
            "robust_mean_ber",
            "robust_mean_papr_db",
        ]
    )
    return frame


def prepare_four_objective(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["throughput_mbps"] = result["robust_mean_throughput_mbps"]
    result["ber"] = result["robust_mean_ber"]
    result["papr_db"] = result["robust_mean_papr_db"]
    result["log10_ber"] = np.log10(np.maximum(result["ber"], np.finfo(float).tiny))

    ptx_dbm = result["x1"].to_numpy(dtype=float)
    throughput_mbps = result["throughput_mbps"].to_numpy(dtype=float)
    pout_w = 10.0 ** ((ptx_dbm - 30.0) / 10.0)
    total_power_w = POWER_MODEL_P_BB + POWER_MODEL_P_RF + pout_w / POWER_MODEL_ETA_PA
    bitrate_bps = np.maximum(throughput_mbps, MIN_THROUGHPUT_MBPS) * 1e6
    energy_per_bit = total_power_w / bitrate_bps

    result["energy_per_bit"] = energy_per_bit
    result["energy_efficiency"] = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)
    result["label"] = result["method"].map(method_label)
    result = result.dropna(subset=OBJECTIVE_COLUMNS)
    return result


def build_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for column_name, _, direction in FOUR_OBJECTIVES:
        values = frame[column_name].to_numpy(dtype=float)
        columns.append(-values if direction == "max" else values)
    return np.column_stack(columns)


def compute_pareto_mask(objectives: np.ndarray) -> np.ndarray:
    if objectives.ndim != 2 or objectives.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    pareto_mask = np.ones(objectives.shape[0], dtype=bool)
    for index in range(objectives.shape[0]):
        dominated = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated[index] = False
        if dominated.any():
            pareto_mask[index] = False
    return pareto_mask


def load_all_candidates(methods: list[str]) -> pd.DataFrame:
    frames = []
    for method_name in methods:
        frame = load_method_candidates(method_name)
        frames.append(prepare_four_objective(frame))
    if not frames:
        raise RuntimeError("No robust simulation workbooks were found.")
    return pd.concat(frames, ignore_index=True)


def add_pareto_flags(candidates: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    tagged_runs = []
    for _, run_data in candidates.groupby("moo_run", sort=True):
        run_data = run_data.copy()
        run_data["is_run_global_pareto"] = compute_pareto_mask(
            build_objective_matrix(run_data)
        )
        run_data["is_run_method_pareto"] = False

        for method_name in methods:
            method_rows = run_data[run_data["method"] == method_name]
            if method_rows.empty:
                continue
            method_mask = compute_pareto_mask(build_objective_matrix(method_rows))
            run_data.loc[method_rows.index, "is_run_method_pareto"] = method_mask

        tagged_runs.append(run_data)

    return pd.concat(tagged_runs, ignore_index=True)


def clean_output_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for stem in GENERATED_FIGURE_STEMS:
        for suffix in CLEANUP_FORMATS:
            output_file = FIG_DIR / f"{stem}{suffix}"
            if output_file.exists():
                output_file.unlink()


def save_figure(fig: plt.Figure, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    target = output_file.with_suffix(OUTPUT_FORMAT)
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {target}")


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.35, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_SIZE,
        width=0.45,
        length=2.6,
        color=AXIS_COLOR,
    )


def normalize_metric(
    values: pd.Series,
    higher_is_better: bool,
    reference: pd.Series | None = None,
) -> pd.Series:
    reference_values = values if reference is None else reference
    minimum = float(reference_values.min())
    maximum = float(reference_values.max())
    span = maximum - minimum
    if span <= 0:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)

    normalized = (values - minimum) / span
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized.clip(0.0, 1.0)


def normalized_objective_costs(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> np.ndarray:
    ref = frame if reference is None else reference
    columns = []
    for column_name, _, direction in FOUR_OBJECTIVES:
        values = frame[column_name].to_numpy(dtype=float)
        ref_values = ref[column_name].to_numpy(dtype=float)
        minimum = float(np.nanmin(ref_values))
        maximum = float(np.nanmax(ref_values))
        span = maximum - minimum
        if span <= 0:
            normalized = np.zeros(len(frame), dtype=float)
        elif direction == "max":
            normalized = (maximum - values) / span
        else:
            normalized = (values - minimum) / span
        columns.append(np.clip(normalized, 0.0, 1.0))
    return np.column_stack(columns)


def normalized_indicator_desirability(indicators: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("#Pareto", True),
        ("#Global Pareto", True),
        ("Global Contrib.", True),
        ("HV (norm)", True),
        ("HV Ratio", True),
        ("IGD+", False),
        ("Spacing", False),
        ("Closest Ideal", False),
    ]
    result = pd.DataFrame(index=indicators.index)
    for column_name, higher_is_better in specs:
        result[column_name] = normalize_metric(
            indicators[column_name], higher_is_better=higher_is_better
        )
    return result


def pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    denominator = np.sum(singular_values ** 2)
    explained = (
        (singular_values[:2] ** 2) / denominator
        if denominator > 0
        else np.zeros(2, dtype=float)
    )
    return coords, explained


def scaled_marker_sizes(
    values: pd.Series,
    min_size: float = 22.0,
    max_size: float = 180.0,
) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if span <= 0:
        return np.full(len(values), (min_size + max_size) / 2.0)
    scaled = (values.to_numpy(dtype=float) - minimum) / span
    return min_size + np.sqrt(np.clip(scaled, 0.0, 1.0)) * (max_size - min_size)


def add_risk_scores(tagged: pd.DataFrame) -> pd.DataFrame:
    result = tagged.copy()
    variance_columns = [
        "robust_var_throughput_mbps",
        "robust_var_ber",
        "robust_var_papr_db",
    ]
    available = [column for column in variance_columns if column in result.columns]
    if available:
        risk_parts = []
        for column in available:
            values = result[column].fillna(result[column].median())
            risk_parts.append(normalize_metric(values, higher_is_better=True))
        result["robust_variance_risk"] = np.column_stack(risk_parts).mean(axis=1)
    else:
        objective_cost = normalized_objective_costs(result)
        result["robust_variance_risk"] = objective_cost.std(axis=1)

    result["objective_cost_score"] = normalized_objective_costs(result).mean(axis=1)
    return result


def sample_background(frame: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(n=max_points, random_state=123)


def plot_metric_manifold_embedding(indicators: pd.DataFrame, topsis: pd.DataFrame) -> None:
    merged = indicators.merge(
        topsis[["method_key", "Ci", "Rank"]],
        on="method_key",
        how="left",
        suffixes=("", "_topsis"),
    )
    desirability = normalized_indicator_desirability(merged)

    order = merged.sort_values("Rank").index
    merged = merged.loc[order].reset_index(drop=True)
    desirability = desirability.loc[order].reset_index(drop=True)

    metric_labels = [
        "Pareto\nsize",
        "Global\nhits",
        "Global\nshare",
        "HV",
        "HV\nratio",
        "IGD+",
        "Spacing",
        "Ideal\ndist.",
    ]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    image = ax.imshow(
        desirability.to_numpy(dtype=float),
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    style_colorbar(colorbar, "Normalized desirability")

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticks(np.arange(len(merged)))
    ax.set_yticklabels(
        [
            f"{int(row.Rank)}. {row.Method}  Ci={row.Ci:.3f}"
            for row in merged.itertuples(index=False)
        ]
    )
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    ax.set_xticks(np.arange(-0.5, len(metric_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(merged), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    dadgp_index = merged.index[merged["method_key"] == "dadgp"]
    if len(dadgp_index):
        row_index = int(dadgp_index[0])
        ax.add_patch(
            patches.Rectangle(
                (-0.5, row_index - 0.5),
                len(metric_labels),
                1.0,
                fill=False,
                edgecolor=color_for_method("dadgp"),
                linewidth=1.45,
                clip_on=False,
            )
        )
        ax.get_yticklabels()[row_index].set_fontweight("bold")
        ax.get_yticklabels()[row_index].set_color(color_for_method("dadgp"))
        for column_index, value in enumerate(desirability.iloc[row_index].to_numpy()):
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                fontweight="bold",
                color="#1B1B1B",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor(AXIS_COLOR)
    ax.set_title("DADGP ranks first by balanced Pareto-quality score")
    ax.set_xlabel("Metric desirability after direction normalization")
    save_figure(fig, FIG_DIR / "01_metric_manifold_embedding.png")


def plot_objective_manifold_embedding(
    tagged: pd.DataFrame,
    methods: list[str],
    summary_note: str,
) -> None:
    costs = normalized_objective_costs(tagged)
    coords, explained = pca_2d(costs)
    embedded = tagged.copy()
    embedded["pc1"] = coords[:, 0]
    embedded["pc2"] = coords[:, 1]

    background = sample_background(embedded[~embedded["is_run_method_pareto"]])
    method_front = embedded[embedded["is_run_method_pareto"]]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.scatter(
        background["pc1"],
        background["pc2"],
        s=8,
        color=BACKGROUND_POINT_COLOR,
        alpha=0.22,
        edgecolors="none",
        rasterized=True,
    )

    context_front = method_front[~method_front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["pc1"],
            context_front["pc2"],
            s=10,
            color=CONTEXT_FRONT_COLOR,
            alpha=0.24,
            edgecolors="none",
            label="Other Pareto fronts",
            rasterized=True,
            zorder=2,
        )

    for method_name in focus_methods(methods):
        front = method_front[method_front["method"] == method_name]
        if front.empty:
            continue
        ax.scatter(
            front["pc1"],
            front["pc2"],
            s=34 if method_name == "dadgp" else 18,
            color=color_for_method(method_name),
            alpha=0.9 if method_name == "dadgp" else 0.5,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.45 if method_name == "dadgp" else 0.25,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 4,
        )

    global_front = embedded[embedded["is_run_global_pareto"]]
    ax.scatter(
        global_front["pc1"],
        global_front["pc2"],
        s=46,
        facecolors="none",
        edgecolors=GLOBAL_FRONT_EDGE,
        linewidths=0.45,
        alpha=0.32,
        zorder=3,
    )

    ax.set_xlabel(f"Objective manifold PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"Objective manifold PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title("Objective-space Pareto manifold with DADGP highlighted")
    style_axes(ax)
    add_compact_legend(ax, loc="best", ncol=2)
    add_panel_note(ax, summary_note, loc="upper left")
    save_figure(fig, FIG_DIR / "02_objective_manifold_embedding.png")


def plot_global_front_3d_projection(
    tagged: pd.DataFrame,
    methods: list[str],
    summary_note: str,
) -> None:
    global_front = tagged[tagged["is_run_global_pareto"]].copy()

    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    context_front = global_front[~global_front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["log10_ber"],
            context_front["papr_db"],
            context_front["throughput_mbps"],
            s=scaled_marker_sizes(context_front["energy_efficiency"], 8, 45),
            color=CONTEXT_FRONT_COLOR,
            alpha=0.16,
            edgecolors="none",
            depthshade=False,
            label="Other global PF hits",
            zorder=1,
        )

    for method_name in focus_methods(methods):
        hits = global_front[global_front["method"] == method_name]
        if hits.empty:
            continue
        ax.scatter(
            hits["log10_ber"],
            hits["papr_db"],
            hits["throughput_mbps"],
            s=scaled_marker_sizes(
                hits["energy_efficiency"],
                34 if method_name == "dadgp" else 16,
                105 if method_name == "dadgp" else 64,
            ),
            color=color_for_method(method_name),
            alpha=0.9 if method_name == "dadgp" else 0.48,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.5 if method_name == "dadgp" else 0.22,
            depthshade=False,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 3,
        )

    quantiles = global_front["energy_efficiency"].quantile([0.25, 0.5, 0.75])
    handles = [
        ax.scatter([], [], [], s=size, color="#BDBDBD", alpha=0.45, edgecolors="none")
        for size in scaled_marker_sizes(quantiles, 12, 70)
    ]
    labels = [f"EE {format_millions(value)}" for value in quantiles]
    size_legend = ax.legend(
        handles,
        labels,
        title="Marker size",
        loc="upper right",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        frameon=False,
        handletextpad=0.35,
        labelspacing=0.28,
    )
    ax.add_artist(size_legend)
    add_compact_legend(ax, loc="lower left", ncol=2)
    ax.set_xlabel("log10(BER)", labelpad=7)
    ax.set_ylabel("PAPR (dB)", labelpad=7)
    ax.set_zlabel("Throughput (Mbps)", labelpad=7)
    ax.set_title("Global Pareto front: DADGP vs key challengers")
    ax.view_init(elev=23, azim=-138)
    ax.tick_params(labelsize=TICK_SIZE, width=0.45)
    add_panel_note(ax, summary_note, loc="upper left")
    save_figure(fig, FIG_DIR / "03_global_front_3d_projection.png")


def plot_attainment_probability_hexbin(tagged: pd.DataFrame, methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    attainment = ax.hexbin(
        tagged["log10_ber"],
        tagged["throughput_mbps"],
        C=tagged["is_run_global_pareto"].astype(float),
        reduce_C_function=np.mean,
        gridsize=34,
        mincnt=1,
        cmap="cividis",
        linewidths=0.0,
        alpha=0.9,
    )
    colorbar = fig.colorbar(attainment, ax=ax)
    style_colorbar(colorbar, "Global PF hit rate per bin")

    method_front = tagged[tagged["is_run_method_pareto"]]
    context_front = method_front[~method_front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["log10_ber"],
            context_front["throughput_mbps"],
            s=7,
            color=CONTEXT_FRONT_COLOR,
            alpha=0.2,
            edgecolors="none",
            label="Other Pareto fronts",
            rasterized=True,
            zorder=2,
        )

    for method_name in focus_methods(methods):
        front = method_front[method_front["method"] == method_name]
        if front.empty:
            continue
        ax.scatter(
            front["log10_ber"],
            front["throughput_mbps"],
            s=26 if method_name == "dadgp" else 12,
            color=color_for_method(method_name),
            alpha=0.88 if method_name == "dadgp" else 0.46,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.42 if method_name == "dadgp" else 0.18,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 4,
        )

    ax.set_xlabel("log10(BER)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("BER-throughput attainment with DADGP highlighted")
    style_axes(ax)
    add_compact_legend(ax, loc="lower right", ncol=2)
    save_figure(fig, FIG_DIR / "04_attainment_probability_hexbin.png")


def plot_design_space_pareto_map(tagged: pd.DataFrame, methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    design_map = ax.hexbin(
        tagged["x1"],
        tagged["x4"],
        C=tagged["is_run_global_pareto"].astype(float),
        reduce_C_function=np.mean,
        gridsize=30,
        mincnt=1,
        cmap="YlGnBu",
        linewidths=0.0,
        alpha=0.94,
    )
    colorbar = fig.colorbar(design_map, ax=ax)
    style_colorbar(colorbar, "Global PF density")

    global_front = tagged[tagged["is_run_global_pareto"]]
    context_front = global_front[~global_front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["x1"],
            context_front["x4"],
            s=10,
            color=CONTEXT_FRONT_COLOR,
            alpha=0.24,
            edgecolors="none",
            label="Other global PF hits",
            rasterized=True,
            zorder=2,
        )

    for method_name in focus_methods(methods):
        hits = global_front[global_front["method"] == method_name]
        if hits.empty:
            continue
        ax.scatter(
            hits["x1"],
            hits["x4"],
            s=42 if method_name == "dadgp" else 18,
            color=color_for_method(method_name),
            alpha=0.9 if method_name == "dadgp" else 0.52,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.45 if method_name == "dadgp" else 0.22,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 4,
        )

    ax.set_xlabel("x1: transmit power control")
    ax.set_ylabel("x4: OFDM design variable")
    ax.set_title("Design-space location of global Pareto solutions")
    style_axes(ax)
    add_compact_legend(ax, loc="best", ncol=2)
    save_figure(fig, FIG_DIR / "05_design_space_pareto_map.png")


def plot_uncertainty_risk_map(tagged: pd.DataFrame, methods: list[str]) -> None:
    scored = add_risk_scores(tagged)
    method_front = scored[scored["is_run_method_pareto"]].copy()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    background = sample_background(scored[~scored["is_run_method_pareto"]])
    ax.scatter(
        background["objective_cost_score"],
        background["robust_variance_risk"],
        s=7,
        color="#D6D6D6",
        alpha=0.18,
        edgecolors="none",
        rasterized=True,
    )

    context_front = method_front[~method_front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["objective_cost_score"],
            context_front["robust_variance_risk"],
            s=scaled_marker_sizes(context_front["energy_efficiency"], 10, 54),
            color=CONTEXT_FRONT_COLOR,
            alpha=0.18,
            edgecolors="none",
            label="Other Pareto fronts",
            rasterized=True,
            zorder=2,
        )

    for method_name in focus_methods(methods):
        front = method_front[method_front["method"] == method_name]
        if front.empty:
            continue
        ax.scatter(
            front["objective_cost_score"],
            front["robust_variance_risk"],
            s=scaled_marker_sizes(
                front["energy_efficiency"],
                34 if method_name == "dadgp" else 16,
                130 if method_name == "dadgp" else 82,
            ),
            color=color_for_method(method_name),
            alpha=0.9 if method_name == "dadgp" else 0.46,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.45 if method_name == "dadgp" else 0.2,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 4,
        )

    ax.set_xlabel("Normalized objective cost (lower is better)")
    ax.set_ylabel("Normalized robust variance risk")
    ax.annotate(
        "better",
        xy=(0.05, 0.06),
        xycoords="axes fraction",
        xytext=(0.22, 0.23),
        textcoords="axes fraction",
        fontsize=6.2,
        color=AXIS_COLOR,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.55,
            "color": AXIS_COLOR,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.set_title("Robust Pareto risk-return structure")
    style_axes(ax)
    add_compact_legend(ax, loc="upper right", ncol=2)
    save_figure(fig, FIG_DIR / "06_uncertainty_risk_map.png")


def plot_projection_density(tagged: pd.DataFrame, methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    density = ax.hexbin(
        tagged["log10_ber"],
        tagged["papr_db"],
        C=tagged["throughput_mbps"],
        reduce_C_function=np.mean,
        gridsize=34,
        mincnt=1,
        cmap="cividis",
        linewidths=0.0,
        alpha=0.9,
    )
    colorbar = fig.colorbar(density, ax=ax)
    style_colorbar(colorbar, "Mean throughput (Mbps)")

    front = tagged[tagged["is_run_method_pareto"]]
    context_front = front[~front["method"].isin(focus_methods(methods))]
    if not context_front.empty:
        ax.scatter(
            context_front["log10_ber"],
            context_front["papr_db"],
            s=8,
            color=CONTEXT_FRONT_COLOR,
            alpha=0.2,
            edgecolors="none",
            label="Other Pareto fronts",
            rasterized=True,
            zorder=2,
        )

    for method_name in focus_methods(methods):
        method_front = front[front["method"] == method_name]
        if method_front.empty:
            continue
        ax.scatter(
            method_front["log10_ber"],
            method_front["papr_db"],
            s=30 if method_name == "dadgp" else 14,
            color=color_for_method(method_name),
            alpha=0.9 if method_name == "dadgp" else 0.48,
            edgecolors=DADGP_EDGE_COLOR if method_name == "dadgp" else "white",
            linewidths=0.42 if method_name == "dadgp" else 0.2,
            label=method_label(method_name),
            zorder=5 if method_name == "dadgp" else 4,
        )

    ax.set_xlabel("log10(BER)")
    ax.set_ylabel("PAPR (dB)")
    ax.set_title("BER-PAPR projection with throughput field")
    style_axes(ax)
    add_compact_legend(ax, loc="upper right", ncol=2)
    save_figure(fig, FIG_DIR / "07_pareto_projection_density.png")


def plot_four_objective_bubble(tagged: pd.DataFrame, summary_note: str) -> None:
    global_front = tagged[tagged["is_run_global_pareto"]].copy()
    sizes = scaled_marker_sizes(global_front["energy_efficiency"])

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    scatter = ax.scatter(
        global_front["throughput_mbps"],
        global_front["log10_ber"],
        c=global_front["papr_db"],
        s=sizes,
        cmap="cividis_r",
        alpha=0.38,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
        zorder=1,
    )

    dadgp_front = global_front[global_front["method"] == "dadgp"]
    if not dadgp_front.empty:
        ax.scatter(
            dadgp_front["throughput_mbps"],
            dadgp_front["log10_ber"],
            s=scaled_marker_sizes(dadgp_front["energy_efficiency"], 34.0, 130.0),
            color=color_for_method("dadgp"),
            alpha=0.9,
            edgecolors=DADGP_EDGE_COLOR,
            linewidths=0.45,
            label="DADGP global PF hits",
            zorder=5,
        )

    colorbar = fig.colorbar(scatter, ax=ax)
    style_colorbar(colorbar, "PAPR (dB)")

    quantiles = global_front["energy_efficiency"].quantile([0.25, 0.5, 0.75])
    handles = [
        ax.scatter([], [], s=size, facecolors="#C8C8C8", edgecolors="#666666", alpha=0.7)
        for size in scaled_marker_sizes(quantiles)
    ]
    labels = [f"EE {format_millions(value)}" for value in quantiles]
    size_legend = ax.legend(
        handles,
        labels,
        title="Bubble size",
        loc="upper right",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
        frameon=False,
        handletextpad=0.35,
        labelspacing=0.28,
    )
    ax.add_artist(size_legend)
    if not dadgp_front.empty:
        add_compact_legend(ax, loc="lower left")

    ax.set_xlabel("Throughput (Mbps)")
    ax.set_ylabel("log10(BER)")
    ax.set_title("Four-objective global Pareto bubble map")
    style_axes(ax)
    add_panel_note(ax, summary_note, loc="upper left")
    save_figure(fig, FIG_DIR / "08_four_objective_bubble.png")


def main() -> None:
    methods = discover_methods()
    if not methods:
        raise RuntimeError(f"No input workbooks found in {RESULT_DIR}")

    indicators, topsis = read_indicator_tables()
    indicators = indicators[indicators["method_key"].isin(methods)].copy()
    topsis = topsis[topsis["method_key"].isin(methods)].copy()
    indicators = indicators.sort_values(
        "method_key", key=lambda values: values.map(method_sort_key)
    ).reset_index(drop=True)
    topsis = topsis.sort_values("Rank").reset_index(drop=True)

    candidates = load_all_candidates(methods)
    tagged = add_pareto_flags(candidates, methods)
    summary_note = build_dadgp_summary_note(indicators, topsis)

    clean_output_dir()
    plot_metric_manifold_embedding(indicators, topsis)
    plot_objective_manifold_embedding(tagged, methods, summary_note)
    plot_global_front_3d_projection(tagged, methods, summary_note)
    plot_attainment_probability_hexbin(tagged, methods)
    plot_design_space_pareto_map(tagged, methods)
    plot_uncertainty_risk_map(tagged, methods)
    plot_projection_density(tagged, methods)
    plot_four_objective_bubble(tagged, summary_note)


if __name__ == "__main__":
    main()
