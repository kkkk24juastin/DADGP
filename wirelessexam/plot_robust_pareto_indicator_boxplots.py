# -*- coding: utf-8 -*-
"""
Plot per-run robust Pareto indicator boxplots across methods.

Inputs:
    result/robust_simulation_<method>.xlsx, sheet all_candidates

Outputs:
    fig/robust_pareto_indicator_boxplots.png
    fig/robust_pareto_indicator_boxplots.pdf
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
import numpy as np
import pandas as pd

import calc_robust_pareto_14methods as metrics


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "fig"
OUTPUT_PNG = FIG_DIR / "robust_pareto_indicator_boxplots.png"
OUTPUT_PDF = FIG_DIR / "robust_pareto_indicator_boxplots.pdf"

MM_PER_INCH = 25.4
FIGURE_SIZE = (260 / MM_PER_INCH, 190 / MM_PER_INCH)
EXPORT_DPI = 600

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

AXIS_COLOR = "#2B2B2B"
GRID_COLOR = "#D6D6D6"
DADGP_EDGE_COLOR = "#1F1F1F"

METRIC_LABELS = {
    "method_pareto_points": "#Pareto",
    "global_pareto_points": "#Global Pareto",
    "global_contribution": "Global Contrib.",
    "hv_norm": "HV (norm)",
    "hv_ratio": "HV Ratio",
    "igd": "IGD",
    "igd_plus": "IGD+",
    "gd": "GD",
    "gd_plus": "GD+",
    "spacing": "Spacing",
    "closest_ideal": "Closest Ideal",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 6.4,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.55,
            "axes.edgecolor": AXIS_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "savefig.dpi": EXPORT_DPI,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def compute_per_run_indicators() -> tuple[pd.DataFrame, list[str]]:
    methods = metrics.discover_methods()
    all_frames = []
    for method in methods:
        frame = metrics.prepare_four_objective(metrics.load_method(method))
        all_frames.append(frame)

    all_data = pd.concat(all_frames, ignore_index=True)
    run_results = []
    for run_id, run_data in all_data.groupby("moo_run", sort=True):
        minima = {
            column: float(run_data[column].min())
            for column, _, _ in metrics.FOUR_OBJECTIVES
        }
        maxima = {
            column: float(run_data[column].max())
            for column, _, _ in metrics.FOUR_OBJECTIVES
        }

        global_mask = metrics.compute_pareto_mask(metrics.build_objective_matrix(run_data))
        run_data_with_global = run_data.copy()
        run_data_with_global["_is_global_pareto"] = global_mask
        global_front = run_data_with_global[global_mask].copy()

        for method in methods:
            method_data = run_data_with_global[
                run_data_with_global["method"] == method
            ].copy()
            if method_data.empty:
                continue

            method_mask = metrics.compute_pareto_mask(
                metrics.build_objective_matrix(method_data)
            )
            method_front = method_data[method_mask].copy()
            global_hits = method_data[method_data["_is_global_pareto"]].copy()
            indicators = metrics.compute_indicators(
                method_front,
                global_front,
                minima,
                maxima,
            )

            run_results.append(
                {
                    "method": method,
                    "label": metrics.METHOD_LABELS.get(method, method),
                    "moo_run": int(run_id),
                    "method_pareto_points": int(method_mask.sum()),
                    "global_pareto_points": int(len(global_hits)),
                    "global_contribution": len(global_hits) / max(len(global_front), 1),
                    **indicators,
                }
            )

    return pd.DataFrame(run_results), methods


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.35, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(axis="both", which="major", width=0.45, length=2.4)


def draw_metric_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    methods: list[str],
    metric: str,
    show_method_labels: bool,
) -> None:
    values = [
        data.loc[data["method"] == method, metric].dropna().to_numpy(dtype=float)
        for method in methods
    ]
    positions = np.arange(1, len(methods) + 1)
    box = ax.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 0.85},
        whiskerprops={"color": "#4A4A4A", "linewidth": 0.55},
        capprops={"color": "#4A4A4A", "linewidth": 0.55},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#555555",
            "markersize": 2.4,
            "alpha": 0.68,
        },
    )

    for patch, method in zip(box["boxes"], methods):
        is_dadgp = method == "dadgp"
        patch.set_facecolor(METHOD_COLORS.get(method, "#999999"))
        patch.set_alpha(0.86 if is_dadgp else 0.52)
        patch.set_edgecolor(DADGP_EDGE_COLOR if is_dadgp else "#555555")
        patch.set_linewidth(1.05 if is_dadgp else 0.48)

    if show_method_labels:
        ax.set_yticks(positions)
        ax.set_yticklabels([metrics.METHOD_LABELS.get(method, method) for method in methods])
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels([])

    ax.invert_yaxis()
    ax.set_title(METRIC_LABELS.get(metric, metric))
    style_axes(ax)


def plot_boxplots(per_run: pd.DataFrame, methods: list[str]) -> None:
    metric_names = [source_col for source_col, _, _ in metrics.INDICATOR_SPECS]
    fig, axes = plt.subplots(3, 4, figsize=FIGURE_SIZE, constrained_layout=True)
    axes_flat = axes.ravel()

    for index, metric in enumerate(metric_names):
        draw_metric_boxplot(
            axes_flat[index],
            per_run,
            methods,
            metric,
            show_method_labels=index % 4 == 0,
        )

    for ax in axes_flat[len(metric_names) :]:
        ax.axis("off")

    fig.suptitle(
        "Per-run robust Pareto indicator distributions across 20 MOO runs",
        fontsize=8.2,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    per_run, methods = compute_per_run_indicators()
    plot_boxplots(per_run, methods)
    print(f"Saved boxplot PNG: {OUTPUT_PNG}")
    print(f"Saved boxplot PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
