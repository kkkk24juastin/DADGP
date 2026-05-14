# -*- coding: utf-8 -*-
"""Utilities for evaluating supplemental DA-DGP sensitivity experiments."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import font_manager

from config import (
    BASE_DIR,
    CN_FONTS,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK_X,
    FONT_SIZE_TICK_Y,
    N_RUNS,
    NUM_HIDDEN_DGP_DIMS,
    TARGET_VALUES,
)
from evaluate_models import (
    METRIC_TYPES,
    NUM_TASKS,
    build_metric_summary_tables,
    evaluate_local_predictions,
    load_data,
    load_weights_from_history,
    save_results,
    summarize_values,
)
from metrics import select_local_data
from supplemental_training_utils import (
    ConfigurableMultitaskDeepGP,
    SUPPLEMENTAL_DIR,
    default_device,
    source_run_dir,
)


ERROR_BAR_LABEL = {
    "std": "Mean ± SD",
    "ci95": "Mean ± 95% CI",
}

METRIC_YLABEL_MAP = {
    "rmse": "RMSE",
    "nlpd": "NLPD",
    "quality_loss": "QL",
}


def build_evaluation_arg_parser(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--end-run", type=int, default=N_RUNS)
    parser.add_argument(
        "--error-bar",
        choices=["std", "ci95"],
        default="ci95",
        help="绘图误差棒类型。",
    )
    parser.add_argument(
        "--use-existing-metrics",
        action="store_true",
        help="若 run 目录已有 local_metrics.xlsx，则优先复用而不重新预测。",
    )
    parser.add_argument("--no-plot", action="store_true", help="只保存表格，不绘图。")
    return parser


def configure_plot_style():
    available = [f.name for f in font_manager.fontManager.ttflist]
    for font_name in CN_FONTS:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font_scale=1.2)


def load_metadata(run_dir: Path):
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def resolve_train_data_path(run_dir: Path, run_id: int, metadata: dict):
    data_sources = metadata.get("data_sources", {})
    return first_existing_path([
        run_dir / "train_data.xlsx",
        metadata.get("saved_train_data"),
        data_sources.get("train_data"),
        source_run_dir(run_id) / "train_data.xlsx",
    ])


def resolve_test_data_path(run_id: int):
    return source_run_dir(run_id) / "test_data.xlsx"


def load_saved_weights(run_dir: Path):
    weights = load_weights_from_history(run_dir / "weight_history_da_dgp.xlsx")
    if weights:
        return weights

    weights_path = run_dir / "weights.xlsx"
    if not weights_path.exists():
        return {}

    df = pd.read_excel(weights_path)
    index_col = df.columns[0]
    weights = {}
    for task in range(1, NUM_TASKS + 1):
        row = df[df[index_col] == f"task{task}"]
        if not row.empty and "da_dgp" in row.columns:
            weights[f"weight_task{task}"] = row["da_dgp"].values[0]
    return weights


def load_saved_metrics(run_dir: Path):
    local_metrics_path = run_dir / "local_metrics.xlsx"
    if not local_metrics_path.exists():
        return None

    metrics = {}
    try:
        for metric in METRIC_TYPES:
            df = pd.read_excel(local_metrics_path, sheet_name=metric)
            index_col = df.columns[0]
            if "da_dgp" not in df.columns:
                return None
            for task in range(1, NUM_TASKS + 1):
                row = df[df[index_col] == f"task{task}"]
                if not row.empty:
                    metrics[f"local_{metric}_task{task}"] = row["da_dgp"].values[0]
    except Exception as exc:
        print(f"[skip] failed to load existing metrics: {local_metrics_path}: {exc}")
        return None

    metrics.update(load_saved_weights(run_dir))
    return metrics


def create_supplemental_model(train_data: dict, num_tasks: int, metadata: dict, device):
    num_hidden_dgp_dims = int(metadata.get("num_hidden_dgp_dims", NUM_HIDDEN_DGP_DIMS))
    num_dgp_layers = int(metadata.get("num_dgp_layers", 2))
    model = ConfigurableMultitaskDeepGP(
        train_data["x"].shape,
        num_hidden_dgp_dims=num_hidden_dgp_dims,
        num_tasks=num_tasks,
        num_dgp_layers=num_dgp_layers,
    )
    return model.to(device)


def evaluate_supplemental_run(
    run_dir: Path,
    run_id: int,
    device,
    use_existing_metrics: bool = False,
):
    if use_existing_metrics:
        saved_metrics = load_saved_metrics(run_dir)
        if saved_metrics:
            print(f"  复用已有指标: {run_dir}")
            return saved_metrics

    model_path = run_dir / "model_da_dgp.pt"
    if not model_path.exists():
        print(f"[skip] missing model: {model_path}")
        return None

    metadata = load_metadata(run_dir)
    train_data_path = resolve_train_data_path(run_dir, run_id, metadata)
    test_data_path = resolve_test_data_path(run_id)
    if train_data_path is None:
        print(f"[skip] run {run_id:02d}: missing train data for {run_dir}")
        return None
    if not test_data_path.exists():
        print(f"[skip] run {run_id:02d}: missing test data: {test_data_path}")
        return None

    train_data = load_data(train_data_path, device)
    test_data = load_data(test_data_path, device)
    num_tasks = test_data["y"].shape[-1]
    local_test_data, local_mask = select_local_data(test_data, TARGET_VALUES)
    local_count = int(local_mask.sum().item())
    total_count = int(test_data["y"].shape[0])
    print(f"  局部测试点: {local_count}/{total_count}")

    model = create_supplemental_model(train_data, num_tasks, metadata, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    metrics = evaluate_local_predictions(
        model, local_test_data, TARGET_VALUES, num_tasks
    )
    metrics.update(load_saved_weights(run_dir))

    save_results({"da_dgp": metrics}, run_dir)
    return metrics


def build_weight_summary_tables(condition_runs: dict, condition_order: list):
    wide_data = {}
    long_records = []
    task_index = [f"task{i}" for i in range(1, NUM_TASKS + 1)]

    for condition in condition_order:
        run_items = condition_runs.get(condition, [])
        means, stds, mean_pm_stds = [], [], []
        ci95s, ci95_lowers, ci95_uppers, mean_pm_ci95s, counts = [], [], [], [], []

        for task in range(1, NUM_TASKS + 1):
            key = f"weight_task{task}"
            values = [item["metrics"].get(key, float("nan")) for item in run_items]
            summary = summarize_values(values)
            means.append(summary["mean"])
            stds.append(summary["std"])
            mean_pm_stds.append(summary["mean±std"])
            ci95s.append(summary["ci95"])
            ci95_lowers.append(summary["ci95_lower"])
            ci95_uppers.append(summary["ci95_upper"])
            mean_pm_ci95s.append(summary["mean±ci95"])
            counts.append(summary["n"])
            long_records.append({
                "condition": condition,
                "task": f"task{task}",
                "mean": summary["mean"],
                "std": summary["std"],
                "mean±std": summary["mean±std"],
                "ci95": summary["ci95"],
                "ci95_lower": summary["ci95_lower"],
                "ci95_upper": summary["ci95_upper"],
                "mean±ci95": summary["mean±ci95"],
                "n": summary["n"],
            })

        wide_data[condition] = means
        wide_data[f"{condition}_std"] = stds
        wide_data[f"{condition}_mean±std"] = mean_pm_stds
        wide_data[f"{condition}_ci95"] = ci95s
        wide_data[f"{condition}_ci95_lower"] = ci95_lowers
        wide_data[f"{condition}_ci95_upper"] = ci95_uppers
        wide_data[f"{condition}_mean±ci95"] = mean_pm_ci95s
        wide_data[f"{condition}_n"] = counts

    return pd.DataFrame(wide_data, index=task_index), pd.DataFrame(long_records)


def save_sensitivity_tables(
    root_dir: Path,
    condition_runs: dict,
    condition_order: list,
    start_run: int,
    end_run: int,
):
    grouped_results = {
        condition: [item["metrics"] for item in condition_runs.get(condition, [])]
        for condition in condition_order
    }
    summary_wide_dfs, summary_long_dfs = build_metric_summary_tables(
        grouped_results,
        condition_order,
        group_column="condition",
    )

    all_path = root_dir / f"local_metrics_all_runs_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(all_path) as writer:
        for metric in METRIC_TYPES:
            records = []
            for condition in condition_order:
                for item in condition_runs.get(condition, []):
                    row = {"run": item["run"], "condition": condition}
                    for task in range(1, NUM_TASKS + 1):
                        key = f"local_{metric}_task{task}"
                        row[f"task{task}"] = item["metrics"].get(key, float("nan"))
                    records.append(row)
            pd.DataFrame(records).to_excel(writer, sheet_name=metric, index=False)
    print(f"保存敏感性逐run指标: '{all_path}'")

    mean_path = root_dir / f"local_metrics_mean_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(mean_path) as writer:
        for metric, df in summary_wide_dfs.items():
            df.to_excel(writer, sheet_name=metric)
    print(f"保存敏感性统计汇总: '{mean_path}'")

    summary_path = root_dir / f"local_metrics_summary_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        for metric, df in summary_long_dfs.items():
            df.to_excel(writer, sheet_name=metric, index=False)
    print(f"保存敏感性长表统计: '{summary_path}'")

    weights_wide_df, weights_long_df = build_weight_summary_tables(
        condition_runs,
        condition_order,
    )
    if not weights_long_df.empty and weights_long_df["n"].sum() > 0:
        weights_mean_path = root_dir / f"weights_mean_{start_run:02d}-{end_run:02d}.xlsx"
        weights_wide_df.to_excel(weights_mean_path)
        print(f"保存敏感性权重统计: '{weights_mean_path}'")

    return summary_long_dfs


def plot_sensitivity_errorbars(
    summary_long_dfs: dict,
    condition_specs: list,
    experiment_name: str,
    x_label: str,
    error_bar: str,
):
    configure_plot_style()
    out_dir = BASE_DIR / "fig" / "sensitivity" / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    condition_order = [spec["label"] for spec in condition_specs]
    display_labels = [spec.get("display", spec["label"]) for spec in condition_specs]
    x_positions = np.arange(len(condition_order))

    pdf_count = 0
    for task in range(1, NUM_TASKS + 1):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for metric_idx, metric in enumerate(METRIC_TYPES):
            ax = axes[metric_idx]
            df_metric = summary_long_dfs[metric]
            means, errors = [], []

            for condition in condition_order:
                row = df_metric[
                    (df_metric["condition"] == condition) &
                    (df_metric["task"] == f"task{task}")
                ]
                if row.empty:
                    means.append(np.nan)
                    errors.append(0.0)
                    continue

                mean_value = row.iloc[0]["mean"]
                error_value = row.iloc[0][error_bar]
                means.append(mean_value)
                errors.append(error_value if np.isfinite(error_value) else 0.0)

            ax.errorbar(
                x_positions,
                means,
                yerr=errors,
                fmt="-o",
                color="#1F77B4",
                ecolor="#1F77B4",
                elinewidth=2.0,
                capsize=4,
                capthick=1.6,
                linewidth=2.5,
                markersize=7,
            )

            ax.set_xlabel(f"{x_label} ({ERROR_BAR_LABEL[error_bar]})", fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel(METRIC_YLABEL_MAP[metric], fontsize=FONT_SIZE_LABEL)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                display_labels,
                rotation=30,
                ha="right",
                fontsize=FONT_SIZE_TICK_X,
            )
            ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_Y)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_linewidth(1.8)
            ax.spines["bottom"].set_linewidth(1.8)
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=1.2)

        fig.tight_layout()
        output_file = out_dir / f"task{task}_metrics_errorbars.pdf"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        pdf_count += 1
        print(f"[{pdf_count}/3] 图片已保存: {output_file}")


def run_sensitivity_evaluation(
    experiment_name: str,
    condition_specs: list,
    x_label: str,
    args,
):
    if args.start_run > args.end_run:
        raise ValueError("--start-run must be <= --end-run")

    device = default_device()
    root_dir = SUPPLEMENTAL_DIR / experiment_name
    condition_order = [spec["label"] for spec in condition_specs]
    condition_runs = {condition: [] for condition in condition_order}

    print(f"Using device: {device}")
    print(f"Input root: {root_dir}")
    print(f"Run range: {args.start_run} - {args.end_run}")
    if not root_dir.exists():
        print(f"[!] 敏感性结果目录不存在: {root_dir}")
        return

    for spec in condition_specs:
        condition = spec["label"]
        print(f"\n=== Evaluate condition: {condition} ===")
        for run_id in range(args.start_run, args.end_run + 1):
            run_dir = root_dir / condition / f"{run_id:02d}"
            if not run_dir.exists():
                print(f"[skip] missing run dir: {run_dir}")
                continue

            print(f"  run {run_id:02d}")
            metrics = evaluate_supplemental_run(
                run_dir,
                run_id,
                device,
                use_existing_metrics=args.use_existing_metrics,
            )
            if metrics:
                condition_runs[condition].append({
                    "run": run_id,
                    "metrics": metrics,
                })

    total_records = sum(len(items) for items in condition_runs.values())
    if total_records == 0:
        print("[!] 没有可汇总的敏感性评估结果。")
        return

    summary_long_dfs = save_sensitivity_tables(
        root_dir,
        condition_runs,
        condition_order,
        args.start_run,
        args.end_run,
    )

    if not args.no_plot:
        plot_sensitivity_errorbars(
            summary_long_dfs,
            condition_specs,
            experiment_name,
            x_label,
            args.error_bar,
        )
