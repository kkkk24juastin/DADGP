# -*- coding: utf-8 -*-
"""
基于当前真实仿真结果，对全部候选点一次性计算多种方法的四目标 Pareto 前沿并可视化。

四目标定义:
    1. 最大化 throughput_mbps
    2. 最小化 ber
    3. 最小化 papr_db
    4. 最大化 energy_efficiency

默认读取:
    result/<method>.xlsx 的 results sheet

默认输出:
    result/real_simulation_pareto_3obj_4obj_eval.xlsx
    fig/moo_real_simulation_pareto_fronts.png
    fig/moo_real_simulation_pareto_metrics.png

直接修改下方配置后运行:
    python analyze_real_simulation_pareto.py
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

_MPL_CONFIG_DIR = Path("/tmp") / "wireless-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from config import BASE_DIR, VALID_METHODS

RESULT_DIR = BASE_DIR / "result"
FIG_DIR = BASE_DIR / "fig"
DEFAULT_SUMMARY_FILE = RESULT_DIR / "real_simulation_pareto_3obj_4obj_eval.xlsx"
DEFAULT_FRONT_FIGURE = FIG_DIR / "moo_real_simulation_pareto_fronts.png"
DEFAULT_METRIC_FIGURE = FIG_DIR / "moo_real_simulation_pareto_metrics.png"

# ---------------------------------------------------------------------------
# 显式运行配置
# SELECTED_METHODS = None 表示按 VALID_METHODS 顺序扫描 result/<method>.xlsx
# 也可以写成 ["dadgp"] 或 ["dadgp", "baseline_dwa"]
# ---------------------------------------------------------------------------
SELECTED_METHODS: list[str] | None = None
INPUT_RESULT_DIR = RESULT_DIR
RESULT_SHEET_NAME = "results"
SUMMARY_FILE = DEFAULT_SUMMARY_FILE
FRONT_FIGURE = DEFAULT_FRONT_FIGURE
METRIC_FIGURE = DEFAULT_METRIC_FIGURE

POWER_MODEL_P_BB = 0.2
POWER_MODEL_P_RF = 0.8
POWER_MODEL_ETA_PA = 0.35
MIN_THROUGHPUT_MBPS = 1e-6
HV_REFERENCE_POINT_VALUE = 1.05

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
}

METHOD_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#A0CBE8",
]

METHOD_COLOR_MAP = {
    method_name: METHOD_COLORS[index]
    for index, method_name in enumerate(VALID_METHODS[: len(METHOD_COLORS)])
}

OBJECTIVES = [
    ("throughput_mbps", "Throughput (Mbps)", "max"),
    ("ber", "BER", "min"),
    ("papr_db", "PAPR (dB)", "min"),
    ("energy_efficiency", "Energy Efficiency (bit/J)", "max"),
]
THREE_OBJECTIVES = OBJECTIVES[:3]
FOUR_OBJECTIVES = OBJECTIVES.copy()
OBJECTIVE_COLUMNS = [column_name for column_name, _, _ in OBJECTIVES]
REQUIRED_COLUMNS = ["method", *OBJECTIVE_COLUMNS]
OPTIONAL_COLUMNS = [
    "moo_run",
    "energy_per_bit",
    "is_pareto",
    "real_quality_loss_task1",
    "real_quality_loss_task2",
    "real_quality_loss_task3",
    "real_total_quality_loss",
]
NUMERIC_COLUMNS = [
    "moo_run",
    "throughput_mbps",
    "ber",
    "papr_db",
    "energy_per_bit",
    "energy_efficiency",
    "real_quality_loss_task1",
    "real_quality_loss_task2",
    "real_quality_loss_task3",
    "real_total_quality_loss",
]
AUXILIARY_METRIC_COLUMNS = [
    "energy_per_bit",
    "real_quality_loss_task1",
    "real_quality_loss_task2",
    "real_quality_loss_task3",
    "real_total_quality_loss",
]

PROJECTION_SPECS = [
    ("ber", "throughput_mbps", "BER", "Throughput (Mbps)", "log"),
    ("papr_db", "throughput_mbps", "PAPR (dB)", "Throughput (Mbps)", "linear"),
    ("ber", "papr_db", "BER", "PAPR (dB)", "log"),
    (
        "energy_efficiency",
        "throughput_mbps",
        "Energy Efficiency (bit/J)",
        "Throughput (Mbps)",
        "linear",
    ),
    (
        "energy_efficiency",
        "ber",
        "Energy Efficiency (bit/J)",
        "BER",
        "linear",
    ),
    (
        "energy_efficiency",
        "papr_db",
        "Energy Efficiency (bit/J)",
        "PAPR (dB)",
        "linear",
    ),
]

GLOBAL_PARETO_COL = "is_global_pareto"
METHOD_PARETO_COL = "is_method_pareto"
RUN_GLOBAL_PARETO_COL = "is_run_global_pareto"
RUN_METHOD_PARETO_COL = "is_run_method_pareto"
LEGACY_GLOBAL_PARETO_COL = "is_global_pareto"
LEGACY_METHOD_PARETO_COL = "is_method_pareto"

PER_RUN_EXPORT_COLUMNS = [
    "composite_rank",
    "method",
    "label",
    "num_runs",
    "avg_global_pareto_hits_per_run",
    "global_pareto_hit_run_coverage",
    "global_front_contribution",
    "global_pareto_ratio",
    "avg_method_pareto_points_per_run",
    "method_pareto_ratio",
    "method_front_hv_norm",
    "hv_ratio_to_global_front",
    "igd_to_global_front",
    "gd_to_global_front",
    "closest_to_global_ideal",
    "pf_mean_throughput",
    "pf_mean_ber",
    "pf_mean_papr",
    "pf_mean_energy_efficiency",
    "composite_score",
]
OVERALL_EXPORT_COLUMNS = [
    "composite_rank",
    "method",
    "label",
    "total_points",
    "global_pareto_points",
    "global_front_contribution",
    "global_pareto_ratio",
    "method_pareto_points",
    "method_pareto_ratio",
    "all_methods_global_pareto_points",
    "method_front_hv_norm",
    "hv_ratio_to_global_front",
    "igd_to_global_front",
    "gd_to_global_front",
    "closest_to_global_ideal",
    "pf_mean_throughput",
    "pf_mean_ber",
    "pf_mean_papr",
    "pf_mean_energy_efficiency",
    "composite_score",
]
PER_RUN_COMPOSITE_METRICS = [
    ("global_pareto_hit_run_coverage", False),
    ("global_front_contribution", False),
    ("method_front_hv_norm", False),
    ("igd_to_global_front", True),
    ("closest_to_global_ideal", True),
]
OVERALL_COMPOSITE_METRICS = [
    ("global_front_contribution", False),
    ("global_pareto_ratio", False),
    ("method_front_hv_norm", False),
    ("igd_to_global_front", True),
    ("closest_to_global_ideal", True),
]


def set_active_objectives(objectives: list[tuple[str, str, str]]) -> None:
    global OBJECTIVES, OBJECTIVE_COLUMNS, REQUIRED_COLUMNS
    OBJECTIVES = list(objectives)
    OBJECTIVE_COLUMNS = [column_name for column_name, _, _ in OBJECTIVES]
    REQUIRED_COLUMNS = ["method", *OBJECTIVE_COLUMNS]


@contextmanager
def objective_scope(objectives: list[tuple[str, str, str]]):
    previous_objectives = OBJECTIVES.copy()
    previous_objective_columns = OBJECTIVE_COLUMNS.copy()
    previous_required_columns = REQUIRED_COLUMNS.copy()
    set_active_objectives(objectives)
    try:
        yield
    finally:
        globals()["OBJECTIVES"] = previous_objectives
        globals()["OBJECTIVE_COLUMNS"] = previous_objective_columns
        globals()["REQUIRED_COLUMNS"] = previous_required_columns


def resolve_selected_methods(selected_methods_config: list[str] | None) -> list[str]:
    if selected_methods_config is None:
        return VALID_METHODS.copy()

    if not isinstance(selected_methods_config, list):
        raise TypeError("SELECTED_METHODS 必须为 list[str] 或 None。")

    methods = [str(item).strip() for item in selected_methods_config if str(item).strip()]
    if not methods:
        raise ValueError("SELECTED_METHODS 不能为空列表。")

    invalid_methods = [method for method in methods if method not in VALID_METHODS]
    if invalid_methods:
        raise ValueError(
            f"存在无效方法: {', '.join(invalid_methods)}。"
            f"可选方法: {', '.join(VALID_METHODS)}"
        )

    return [method for method in VALID_METHODS if method in set(methods)]


def sort_by_method(frame: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    method_order = {method_name: index for index, method_name in enumerate(VALID_METHODS)}
    sort_columns = ["method"]
    if extra_columns:
        sort_columns.extend([column for column in extra_columns if column in frame.columns])
    return frame.sort_values(
        by=sort_columns,
        key=lambda column: column.map(method_order) if column.name == "method" else column,
        kind="stable",
    ).reset_index(drop=True)


def derive_energy_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if "energy_per_bit" in frame.columns and "energy_efficiency" in frame.columns:
        return frame

    if "x1" not in frame.columns or "throughput_mbps" not in frame.columns:
        missing_columns = ", ".join(
            column_name
            for column_name in ["x1", "throughput_mbps"]
            if column_name not in frame.columns
        )
        raise ValueError(f"结果文件缺少推导能耗指标所需列: {missing_columns}")

    ptx_dbm = pd.to_numeric(frame["x1"], errors="coerce").to_numpy(dtype=float)
    throughput_mbps = pd.to_numeric(frame["throughput_mbps"], errors="coerce").to_numpy(dtype=float)
    pout_w = 10.0 ** ((ptx_dbm - 30.0) / 10.0)
    total_power_w = POWER_MODEL_P_BB + POWER_MODEL_P_RF + pout_w / POWER_MODEL_ETA_PA
    bitrate_bps = np.maximum(throughput_mbps, MIN_THROUGHPUT_MBPS) * 1e6
    energy_per_bit = total_power_w / bitrate_bps

    frame["energy_per_bit"] = energy_per_bit
    frame["energy_efficiency"] = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)
    return frame


def normalize_columns(frame: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    canonical_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    lower_to_actual = {str(column).strip().lower(): column for column in frame.columns}
    rename_map = {}

    for canonical_name in canonical_columns:
        actual_name = lower_to_actual.get(canonical_name.lower())
        if actual_name is not None and actual_name != canonical_name:
            rename_map[actual_name] = canonical_name

    if rename_map:
        frame = frame.rename(columns=rename_map)

    if (
        "energy_per_bit" not in frame.columns
        and "energy_efficiency" not in frame.columns
        and {"x1", "throughput_mbps"}.issubset(frame.columns)
    ):
        frame = derive_energy_metrics(frame.copy())

    if "energy_per_bit" not in frame.columns and "energy_efficiency" in frame.columns:
        efficiency = pd.to_numeric(frame["energy_efficiency"], errors="coerce")
        frame["energy_per_bit"] = np.where(efficiency > 0, 1.0 / efficiency, np.nan)

    if "energy_efficiency" not in frame.columns and "energy_per_bit" in frame.columns:
        energy_per_bit = pd.to_numeric(frame["energy_per_bit"], errors="coerce")
        frame["energy_efficiency"] = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"{source_file} 缺少必要列: {missing_text}")

    return frame


def discover_result_files(input_result_dir: Path, selected_methods: list[str]) -> tuple[list[Path], list[str]]:
    if not input_result_dir.exists():
        raise FileNotFoundError(f"未找到结果目录: {input_result_dir}")

    existing_files: list[Path] = []
    missing_methods: list[str] = []
    for method_name in selected_methods:
        candidate = input_result_dir / f"{method_name}.xlsx"
        if candidate.exists():
            existing_files.append(candidate)
        else:
            missing_methods.append(method_name)
    return existing_files, missing_methods


def load_single_result_file(result_file: Path, sheet_name: str, method_name: str) -> pd.DataFrame:
    frame = pd.read_excel(result_file, sheet_name=sheet_name)
    if frame.empty:
        raise ValueError(f"{result_file} 的 {sheet_name} 工作表没有可用数据。")

    # 新文件格式是“一方法一文件”，以文件名作为方法权威来源。
    frame["method"] = method_name

    frame = normalize_columns(frame, result_file).copy()
    frame["method"] = frame["method"].astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "moo_run" not in frame.columns:
        print("警告: 输入结果缺少 moo_run 列，将所有记录视为同一次运行。")
        frame["moo_run"] = 1
    else:
        missing_run_mask = frame["moo_run"].isna()
        dropped_run_rows = int(missing_run_mask.sum())
        if dropped_run_rows:
            print(f"跳过 {dropped_run_rows} 行缺少 moo_run 的记录。")
            frame = frame.loc[~missing_run_mask].copy()
        frame["moo_run"] = frame["moo_run"].astype(int)

    frame = frame[frame["method"].isin(VALID_METHODS)].copy()
    if frame.empty:
        raise ValueError("结果文件中没有匹配到目标方法。")

    valid_rows = frame[REQUIRED_COLUMNS[1:]].notna().all(axis=1)
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows:
        print(f"跳过 {dropped_rows} 行缺少必要指标的记录。")
    frame = frame.loc[valid_rows].copy()

    frame["method"] = method_name

    if frame.empty:
        raise ValueError(f"{result_file} 过滤缺失值后没有可用数据。")

    return frame


def load_results(
    input_result_dir: Path,
    selected_methods: list[str],
    result_sheet_name: str,
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    result_files, missing_methods = discover_result_files(input_result_dir, selected_methods)
    if not result_files:
        raise FileNotFoundError(f"在 {input_result_dir} 下没有找到任何可用方法结果文件。")

    frames = []
    loaded_files = []
    for result_file in result_files:
        method_name = result_file.stem
        try:
            frame = load_single_result_file(result_file, result_sheet_name, method_name)
        except Exception as exc:
            print(f"警告: 跳过 {result_file.name}: {exc}")
            continue
        frames.append(frame)
        loaded_files.append(result_file)

    if not frames:
        raise ValueError("所有方法结果文件都无法读取为有效分析数据。")

    results = pd.concat(frames, ignore_index=True)
    return sort_by_method(results), loaded_files, missing_methods


def build_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    objective_columns = []
    for column_name, _, direction in OBJECTIVES:
        column_values = frame[column_name].to_numpy(dtype=float)
        objective_columns.append(-column_values if direction == "max" else column_values)
    return np.column_stack(objective_columns)


def compute_pareto_mask(objectives: np.ndarray) -> np.ndarray:
    total_points = objectives.shape[0]
    pareto_mask = np.ones(total_points, dtype=bool)
    for index in range(total_points):
        dominated_by_other = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated_by_other[index] = False
        if dominated_by_other.any():
            pareto_mask[index] = False
    return pareto_mask


def add_pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    results[GLOBAL_PARETO_COL] = compute_pareto_mask(build_objective_matrix(results))
    results[METHOD_PARETO_COL] = False

    for method_name in VALID_METHODS:
        method_rows = results[results["method"] == method_name]
        if method_rows.empty:
            continue
        method_mask = compute_pareto_mask(build_objective_matrix(method_rows))
        results.loc[method_rows.index, METHOD_PARETO_COL] = method_mask

    if RUN_GLOBAL_PARETO_COL != GLOBAL_PARETO_COL:
        results[RUN_GLOBAL_PARETO_COL] = results[GLOBAL_PARETO_COL]
    if RUN_METHOD_PARETO_COL != METHOD_PARETO_COL:
        results[RUN_METHOD_PARETO_COL] = results[METHOD_PARETO_COL]

    if "is_pareto" in results.columns:
        provided_mask = results["is_pareto"].astype(bool).to_numpy()
        computed_mask = results[GLOBAL_PARETO_COL].to_numpy()
        if not np.array_equal(provided_mask, computed_mask):
            mismatch_count = int(np.count_nonzero(provided_mask != computed_mask))
            print(
                f"警告: 输入文件中的 is_pareto 与全体点重算结果有 {mismatch_count} 处不一致，"
                "将以脚本重算结果为准。"
            )

    results[LEGACY_GLOBAL_PARETO_COL] = results[GLOBAL_PARETO_COL]
    results[LEGACY_METHOD_PARETO_COL] = results[METHOD_PARETO_COL]

    return sort_by_method(
        results,
        extra_columns=["moo_run", "solution_idx", METHOD_PARETO_COL, GLOBAL_PARETO_COL],
    )


def normalize_for_cost(frame: pd.DataFrame, minima: dict[str, float], maxima: dict[str, float]) -> np.ndarray:
    if frame.empty:
        return np.empty((0, len(OBJECTIVES)), dtype=float)

    normalized_columns = []
    for column_name, _, direction in OBJECTIVES:
        span = maxima[column_name] - minima[column_name]
        values = frame[column_name].to_numpy(dtype=float)
        if span <= 0:
            normalized = np.zeros_like(values, dtype=float)
        elif direction == "max":
            normalized = (maxima[column_name] - values) / span
        else:
            normalized = (values - minima[column_name]) / span
        normalized_columns.append(np.clip(normalized, 0.0, 1.0))
    return np.column_stack(normalized_columns)


def _metric_key(column_name: str) -> str:
    return {
        "throughput_mbps": "throughput",
        "ber": "ber",
        "papr_db": "papr",
        "energy_efficiency": "energy_efficiency",
    }.get(column_name, column_name)


def _best_metric_value(frame: pd.DataFrame, column_name: str, direction: str) -> float:
    if direction == "max":
        return float(frame[column_name].max())
    return float(frame[column_name].min())


def _worst_metric_value(frame: pd.DataFrame, column_name: str, direction: str) -> float:
    if direction == "max":
        return float(frame[column_name].min())
    return float(frame[column_name].max())


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _unique_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.reshape(0, values.shape[1] if values.ndim == 2 else len(OBJECTIVES))
    return np.unique(values, axis=0)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return distances[np.triu_indices(len(values), k=1)]


def _nearest_neighbor_distances(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _hypervolume_values(normalized_front: np.ndarray) -> tuple[float, float]:
    if len(normalized_front) == 0:
        return 0.0, 0.0

    ref_point = np.full(normalized_front.shape[1], HV_REFERENCE_POINT_VALUE, dtype=float)
    hypervolume = float(HV(ref_point=ref_point)(_unique_rows(normalized_front)))
    normalized_hypervolume = hypervolume / float(np.prod(ref_point))
    return hypervolume, normalized_hypervolume


def _distance_indicator_metrics(
    normalized_front: np.ndarray, normalized_reference_front: np.ndarray
) -> dict[str, float]:
    nan_value = float("nan")
    if len(normalized_front) == 0 or len(normalized_reference_front) == 0:
        return {
            "gd_to_global_front": nan_value,
            "gd_plus_to_global_front": nan_value,
            "igd_to_global_front": nan_value,
            "igd_plus_to_global_front": nan_value,
        }

    reference_front = _unique_rows(normalized_reference_front)
    front = _unique_rows(normalized_front)
    return {
        "gd_to_global_front": float(GD(reference_front)(front)),
        "gd_plus_to_global_front": float(GDPlus(reference_front)(front)),
        "igd_to_global_front": float(IGD(reference_front)(front)),
        "igd_plus_to_global_front": float(IGDPlus(reference_front)(front)),
    }


def _normalized_distribution_metrics(normalized_front: np.ndarray) -> dict[str, float]:
    nan_value = float("nan")
    if len(normalized_front) == 0:
        return {
            "mean_distance_to_global_ideal": nan_value,
            "median_distance_to_global_ideal": nan_value,
            "closest_to_global_ideal": nan_value,
            "worst_distance_to_global_ideal": nan_value,
            "mean_normalized_cost_sum": nan_value,
            "median_normalized_cost_sum": nan_value,
            "best_normalized_cost_sum": nan_value,
            "worst_normalized_cost_sum": nan_value,
            "mean_pairwise_distance": nan_value,
            "std_pairwise_distance": nan_value,
            "mean_nearest_neighbor_distance": nan_value,
            "min_nearest_neighbor_distance": nan_value,
            "max_nearest_neighbor_distance": nan_value,
            "spacing": nan_value,
        }

    distances_to_ideal = np.sqrt(np.sum(normalized_front**2, axis=1))
    normalized_cost_sums = normalized_front.sum(axis=1)
    pairwise = _pairwise_distances(normalized_front)
    nearest = _nearest_neighbor_distances(normalized_front)

    return {
        "mean_distance_to_global_ideal": float(distances_to_ideal.mean()),
        "median_distance_to_global_ideal": float(np.median(distances_to_ideal)),
        "closest_to_global_ideal": float(distances_to_ideal.min()),
        "worst_distance_to_global_ideal": float(distances_to_ideal.max()),
        "mean_normalized_cost_sum": float(normalized_cost_sums.mean()),
        "median_normalized_cost_sum": float(np.median(normalized_cost_sums)),
        "best_normalized_cost_sum": float(normalized_cost_sums.min()),
        "worst_normalized_cost_sum": float(normalized_cost_sums.max()),
        "mean_pairwise_distance": float(pairwise.mean()) if len(pairwise) else nan_value,
        "std_pairwise_distance": float(pairwise.std(ddof=0)) if len(pairwise) else nan_value,
        "mean_nearest_neighbor_distance": float(nearest.mean()) if len(nearest) else nan_value,
        "min_nearest_neighbor_distance": float(nearest.min()) if len(nearest) else nan_value,
        "max_nearest_neighbor_distance": float(nearest.max()) if len(nearest) else nan_value,
        "spacing": float(nearest.std(ddof=0)) if len(nearest) else nan_value,
    }


def summarize_indicator_metrics(
    front: pd.DataFrame,
    reference_front: pd.DataFrame,
    minima: dict[str, float],
    maxima: dict[str, float],
    global_hypervolume_norm: float,
) -> dict[str, float]:
    normalized_front = normalize_for_cost(front, minima, maxima)
    normalized_reference_front = normalize_for_cost(reference_front, minima, maxima)
    hypervolume, hypervolume_norm = _hypervolume_values(normalized_front)
    hypervolume_ratio = (
        float(hypervolume_norm / global_hypervolume_norm)
        if global_hypervolume_norm > 0
        else 0.0
    )

    metrics = {
        "hypervolume": hypervolume,
        "hypervolume_norm": hypervolume_norm,
        "hypervolume_ratio_to_global": hypervolume_ratio,
    }
    metrics.update(_distance_indicator_metrics(normalized_front, normalized_reference_front))
    return metrics


def _build_nan_front_metrics() -> dict[str, float]:
    nan_value = float("nan")
    metrics = {
        "closest_to_run_ideal": nan_value,
        "closest_to_global_ideal": nan_value,
        "mean_distance_to_global_ideal": nan_value,
        "median_distance_to_global_ideal": nan_value,
        "worst_distance_to_global_ideal": nan_value,
        "best_normalized_cost_sum": nan_value,
        "mean_normalized_cost_sum": nan_value,
        "median_normalized_cost_sum": nan_value,
        "worst_normalized_cost_sum": nan_value,
        "mean_normalized_front_span": nan_value,
        "normalized_front_volume": nan_value,
        "mean_pairwise_distance": nan_value,
        "std_pairwise_distance": nan_value,
        "mean_nearest_neighbor_distance": nan_value,
        "min_nearest_neighbor_distance": nan_value,
        "max_nearest_neighbor_distance": nan_value,
        "spacing": nan_value,
    }
    for column_name, _, _ in OBJECTIVES:
        metric_key = _metric_key(column_name)
        metrics[f"span_{metric_key}"] = nan_value
        metrics[f"best_{metric_key}"] = nan_value
        metrics[f"worst_{metric_key}"] = nan_value
        metrics[f"front_min_{metric_key}"] = nan_value
        metrics[f"front_max_{metric_key}"] = nan_value
        metrics[f"front_mean_{metric_key}"] = nan_value
        metrics[f"front_median_{metric_key}"] = nan_value
        metrics[f"front_std_{metric_key}"] = nan_value
    for column_name in AUXILIARY_METRIC_COLUMNS:
        metric_key = _metric_key(column_name)
        metrics[f"front_min_{metric_key}"] = nan_value
        metrics[f"front_max_{metric_key}"] = nan_value
        metrics[f"front_mean_{metric_key}"] = nan_value
        metrics[f"front_median_{metric_key}"] = nan_value
        metrics[f"front_std_{metric_key}"] = nan_value
    return metrics


def summarize_front_metrics(
    front: pd.DataFrame, minima: dict[str, float], maxima: dict[str, float]
) -> dict[str, float]:
    if front.empty:
        return _build_nan_front_metrics()

    normalized_front = normalize_for_cost(front, minima, maxima)
    span_values = normalized_front.max(axis=0) - normalized_front.min(axis=0)
    distribution_metrics = _normalized_distribution_metrics(normalized_front)

    metrics = {
        # 保留旧字段名，避免已有图表或表格引用失效。
        "closest_to_run_ideal": distribution_metrics["closest_to_global_ideal"],
        "mean_normalized_front_span": float(span_values.mean()),
        "normalized_front_volume": float(np.prod(span_values)),
    }
    metrics.update(distribution_metrics)
    for objective_index, (column_name, _, direction) in enumerate(OBJECTIVES):
        metric_key = _metric_key(column_name)
        column_values = pd.to_numeric(front[column_name], errors="coerce")
        metrics[f"span_{metric_key}"] = float(span_values[objective_index])
        metrics[f"best_{metric_key}"] = _best_metric_value(front, column_name, direction)
        metrics[f"worst_{metric_key}"] = _worst_metric_value(front, column_name, direction)
        metrics[f"front_min_{metric_key}"] = float(column_values.min())
        metrics[f"front_max_{metric_key}"] = float(column_values.max())
        metrics[f"front_mean_{metric_key}"] = float(column_values.mean())
        metrics[f"front_median_{metric_key}"] = float(column_values.median())
        metrics[f"front_std_{metric_key}"] = float(column_values.std(ddof=0))
    for column_name in AUXILIARY_METRIC_COLUMNS:
        if column_name not in front.columns:
            continue
        metric_key = _metric_key(column_name)
        column_values = pd.to_numeric(front[column_name], errors="coerce").dropna()
        if column_values.empty:
            continue
        metrics[f"front_min_{metric_key}"] = float(column_values.min())
        metrics[f"front_max_{metric_key}"] = float(column_values.max())
        metrics[f"front_mean_{metric_key}"] = float(column_values.mean())
        metrics[f"front_median_{metric_key}"] = float(column_values.median())
        metrics[f"front_std_{metric_key}"] = float(column_values.std(ddof=0))
    return metrics


def build_run_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for moo_run, run_rows in results.groupby("moo_run", sort=True):
        run_global_front = run_rows[run_rows[GLOBAL_PARETO_COL]].copy()
        summary_row = {
            "moo_run": int(moo_run),
            "total_points": int(len(run_rows)),
            "global_pareto_points_in_run": int(len(run_global_front)),
            "methods_present": int(run_rows["method"].nunique()),
            "methods_with_global_pareto": int(run_global_front["method"].nunique()),
        }
        for column_name, _, direction in OBJECTIVES:
            metric_key = _metric_key(column_name)
            summary_row[f"best_global_{metric_key}"] = _best_metric_value(
                run_global_front, column_name, direction
            )
            summary_row[f"mean_global_{metric_key}"] = float(
                run_global_front[column_name].mean()
            )
        summary_rows.append(summary_row)

    run_summary = pd.DataFrame(summary_rows)
    if run_summary.empty:
        return run_summary
    return run_summary.sort_values(by=["moo_run"], kind="stable").reset_index(drop=True)


def build_run_method_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    minima = {column: float(results[column].min()) for column, _, _ in OBJECTIVES}
    maxima = {column: float(results[column].max()) for column, _, _ in OBJECTIVES}
    global_front = results[results[GLOBAL_PARETO_COL]].copy()
    _, global_hypervolume_norm = _hypervolume_values(
        normalize_for_cost(global_front, minima, maxima)
    )
    total_global_pareto = int(results[GLOBAL_PARETO_COL].sum())
    methods_with_global_pareto = int(
        results.loc[results[GLOBAL_PARETO_COL], "method"].nunique()
    )

    for method_name in VALID_METHODS:
        method_rows = results[results["method"] == method_name].copy()
        if method_rows.empty:
            continue

        method_front = method_rows[method_rows[METHOD_PARETO_COL]].copy()
        global_front_hits = method_rows[method_rows[GLOBAL_PARETO_COL]].copy()
        global_hit_front_metrics = summarize_front_metrics(global_front_hits, minima, maxima)
        method_front_metrics = _prefix_metrics(
            summarize_front_metrics(method_front, minima, maxima),
            "method_front_",
        )
        global_hit_indicator_metrics = _prefix_metrics(
            summarize_indicator_metrics(
                global_front_hits,
                global_front,
                minima,
                maxima,
                global_hypervolume_norm,
            ),
            "global_hit_",
        )
        method_front_indicator_metrics = _prefix_metrics(
            summarize_indicator_metrics(
                method_front,
                global_front,
                minima,
                maxima,
                global_hypervolume_norm,
            ),
            "method_front_",
        )
        global_to_method_pareto_ratio = (
            float(len(global_front_hits) / len(method_front)) if len(method_front) else 0.0
        )

        summary_rows.append(
            {
                "method": method_name,
                "label": METHOD_LABELS.get(method_name, method_name),
                "num_runs": int(method_rows["moo_run"].nunique()),
                "total_points": int(len(method_rows)),
                "method_pareto_points": int(len(method_front)),
                "method_pareto_ratio": float(len(method_front) / len(method_rows)),
                "method_pareto_runs": int(method_front["moo_run"].nunique()),
                "method_pareto_run_coverage": float(
                    method_front["moo_run"].nunique() / method_rows["moo_run"].nunique()
                ),
                "global_pareto_points": int(len(global_front_hits)),
                "global_pareto_ratio": float(len(global_front_hits) / len(method_rows)),
                "global_pareto_runs": int(global_front_hits["moo_run"].nunique()),
                "global_pareto_run_coverage": float(
                    global_front_hits["moo_run"].nunique() / method_rows["moo_run"].nunique()
                ),
                "global_to_method_pareto_ratio": global_to_method_pareto_ratio,
                "total_global_pareto_points": total_global_pareto,
                "methods_with_global_pareto": methods_with_global_pareto,
                "global_front_hypervolume_norm": global_hypervolume_norm,
                "global_front_contribution": (
                    float(len(global_front_hits) / total_global_pareto)
                    if total_global_pareto
                    else 0.0
                ),
                **global_hit_front_metrics,
                **global_hit_indicator_metrics,
                **method_front_metrics,
                **method_front_indicator_metrics,
            }
        )

    run_method_summary = pd.DataFrame(summary_rows)
    if run_method_summary.empty:
        return run_method_summary
    return sort_by_method(run_method_summary)


def build_method_summary(run_method_summary: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method_name in VALID_METHODS:
        method_row = run_method_summary[run_method_summary["method"] == method_name].copy()
        if method_row.empty:
            continue
        source = method_row.iloc[0]
        summary_row = source.to_dict()
        for column_name, _, direction in OBJECTIVES:
            metric_key = _metric_key(column_name)
            summary_row[f"best_{metric_key}"] = source[f"best_{metric_key}"]
            summary_row[f"mean_front_{metric_key}"] = source[f"front_mean_{metric_key}"]
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    return sort_by_method(summary)


def infer_metric_direction(metric_name: str) -> str:
    lower_keywords = (
        "distance",
        "igd",
        "gd",
        "spacing",
        "ber",
        "papr",
        "energy_per_bit",
        "quality_loss",
        "cost_sum",
    )
    higher_keywords = (
        "hypervolume",
        "contribution",
        "ratio",
        "coverage",
        "points",
        "runs",
        "throughput",
        "energy_efficiency",
        "span",
        "volume",
    )
    if any(keyword in metric_name for keyword in lower_keywords):
        return "lower_is_better"
    if any(keyword in metric_name for keyword in higher_keywords):
        return "higher_is_better"
    return "reference"


def build_method_summary_by_metric(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return method_summary.copy()

    sorted_summary = sort_by_method(method_summary)
    metric_columns = [
        column
        for column in sorted_summary.columns
        if column not in {"method", "label"}
    ]

    rows = []
    for metric_name in metric_columns:
        row = {
            "metric": metric_name,
            "direction": infer_metric_direction(metric_name),
        }
        for _, method_row in sorted_summary.iterrows():
            row[str(method_row["label"])] = method_row[metric_name]
        rows.append(row)

    return pd.DataFrame(rows)


def add_composite_ranking(
    frame: pd.DataFrame,
    composite_metrics: list[tuple[str, bool]],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ranked = frame.copy()
    composite_score = pd.Series(0.0, index=ranked.index)
    for column_name, ascending in composite_metrics:
        composite_score = composite_score + ranked[column_name].rank(
            ascending=ascending,
            method="min",
            na_option="bottom",
        )
    ranked["composite_score"] = composite_score.astype(int)

    ranked = ranked.sort_values(
        by=[
            "composite_score",
            "method_front_hv_norm",
            "global_front_contribution",
            "igd_to_global_front",
            "closest_to_global_ideal",
        ],
        ascending=[True, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "composite_rank", np.arange(1, len(ranked) + 1))
    return ranked


def build_overall_eval_summary(method_summary: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method_name in VALID_METHODS:
        method_row = method_summary[method_summary["method"] == method_name]
        if method_row.empty:
            continue
        source = method_row.iloc[0]
        summary_rows.append(
            {
                "method": method_name,
                "label": source["label"],
                "total_points": int(source["total_points"]),
                "global_pareto_points": int(source["global_pareto_points"]),
                "global_front_contribution": float(source["global_front_contribution"]),
                "global_pareto_ratio": float(source["global_pareto_ratio"]),
                "method_pareto_points": int(source["method_pareto_points"]),
                "method_pareto_ratio": float(source["method_pareto_ratio"]),
                "all_methods_global_pareto_points": int(
                    source["total_global_pareto_points"]
                ),
                "method_front_hv_norm": float(
                    source["method_front_hypervolume_norm"]
                ),
                "hv_ratio_to_global_front": float(
                    source["method_front_hypervolume_ratio_to_global"]
                ),
                "igd_to_global_front": float(
                    source["method_front_igd_to_global_front"]
                ),
                "gd_to_global_front": float(
                    source["method_front_gd_to_global_front"]
                ),
                "closest_to_global_ideal": float(
                    source["method_front_closest_to_global_ideal"]
                ),
                **{
                    f"pf_mean_{_metric_key(col)}": float(source[f"front_mean_{_metric_key(col)}"])
                    for col, _, _ in OBJECTIVES
                    if f"front_mean_{_metric_key(col)}" in source.index
                },
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = add_composite_ranking(summary, OVERALL_COMPOSITE_METRICS)
    available_columns = [col for col in OVERALL_EXPORT_COLUMNS if col in summary.columns]
    return summary[available_columns] if not summary.empty else summary


def build_per_run_eval_summary(results: pd.DataFrame) -> pd.DataFrame:
    run_method_frames = []
    for moo_run, run_rows in results.groupby("moo_run", sort=True):
        tagged_run = add_pareto_flags(run_rows)
        run_method_summary = build_run_method_summary(tagged_run)
        if run_method_summary.empty:
            continue

        run_global_front_points = int(tagged_run[GLOBAL_PARETO_COL].sum())
        run_method_summary = run_method_summary.copy()
        run_method_summary["moo_run"] = int(moo_run)
        run_method_summary["run_global_front_contribution"] = (
            run_method_summary["global_pareto_points"] / run_global_front_points
            if run_global_front_points
            else 0.0
        )
        run_method_frames.append(run_method_summary)

    if not run_method_frames:
        return pd.DataFrame(columns=PER_RUN_EXPORT_COLUMNS)

    per_run = pd.concat(run_method_frames, ignore_index=True)
    summary_rows = []
    for method_name in VALID_METHODS:
        method_rows = per_run[per_run["method"] == method_name]
        if method_rows.empty:
            continue
        source = method_rows.iloc[0]
        summary_rows.append(
            {
                "method": method_name,
                "label": source["label"],
                "num_runs": int(method_rows["moo_run"].nunique()),
                "avg_global_pareto_hits_per_run": float(
                    method_rows["global_pareto_points"].mean()
                ),
                "global_pareto_hit_run_coverage": float(
                    (method_rows["global_pareto_points"] > 0).mean()
                ),
                "global_front_contribution": float(
                    method_rows["run_global_front_contribution"].mean()
                ),
                "global_pareto_ratio": float(
                    method_rows["global_pareto_ratio"].mean()
                ),
                "avg_method_pareto_points_per_run": float(
                    method_rows["method_pareto_points"].mean()
                ),
                "method_pareto_ratio": float(
                    method_rows["method_pareto_ratio"].mean()
                ),
                "method_front_hv_norm": float(
                    method_rows["method_front_hypervolume_norm"].mean()
                ),
                "hv_ratio_to_global_front": float(
                    method_rows["method_front_hypervolume_ratio_to_global"].mean()
                ),
                "igd_to_global_front": float(
                    method_rows["method_front_igd_to_global_front"].mean()
                ),
                "gd_to_global_front": float(
                    method_rows["method_front_gd_to_global_front"].mean()
                ),
                "closest_to_global_ideal": float(
                    method_rows["method_front_closest_to_global_ideal"].mean()
                ),
                **{
                    f"pf_mean_{_metric_key(col)}": float(method_rows[f"front_mean_{_metric_key(col)}"].mean())
                    for col, _, _ in OBJECTIVES
                    if f"front_mean_{_metric_key(col)}" in method_rows.columns
                },
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = add_composite_ranking(summary, PER_RUN_COMPOSITE_METRICS)
    available_columns = [col for col in PER_RUN_EXPORT_COLUMNS if col in summary.columns]
    return summary[available_columns] if not summary.empty else summary


def analyze_objective_set(
    results: pd.DataFrame,
    objectives: list[tuple[str, str, str]],
) -> dict[str, pd.DataFrame]:
    with objective_scope(objectives):
        tagged_results = add_pareto_flags(results)
        run_summary = build_run_summary(tagged_results)
        run_method_summary = build_run_method_summary(tagged_results)
        method_summary = build_method_summary(run_method_summary)
        return {
            "tagged_results": tagged_results,
            "run_summary": run_summary,
            "run_method_summary": run_method_summary,
            "method_summary": method_summary,
            "per_run_eval": build_per_run_eval_summary(results),
            "overall_eval": build_overall_eval_summary(method_summary),
        }


def build_eval_workbook_frames(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    four_objective_analysis = analyze_objective_set(results, FOUR_OBJECTIVES)
    return {
        "4obj_overall": four_objective_analysis["overall_eval"],
    }


def build_export_frames(
    results: pd.DataFrame,
    run_summary: pd.DataFrame,
    run_method_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    loaded_files: list[Path] | None = None,
    missing_methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    tagged_points = sort_by_method(
        results[
            [
                column
                for column in [
                    "method",
                    "moo_run",
                    "solution_idx",
                    "x1",
                    "x2",
                    "x3",
                    "x4",
                    "throughput_mbps",
                    "ber",
                    "papr_db",
                    "energy_per_bit",
                    "energy_efficiency",
                    "real_quality_loss_task1",
                    "real_quality_loss_task2",
                    "real_quality_loss_task3",
                    "real_total_quality_loss",
                    METHOD_PARETO_COL,
                    GLOBAL_PARETO_COL,
                    RUN_METHOD_PARETO_COL,
                    RUN_GLOBAL_PARETO_COL,
                ]
                if column in results.columns
            ]
        ],
        extra_columns=["moo_run", "solution_idx", METHOD_PARETO_COL, GLOBAL_PARETO_COL],
    )

    method_front_points = sort_by_method(
        tagged_points[tagged_points[METHOD_PARETO_COL]].copy(),
        extra_columns=["moo_run", "throughput_mbps"],
    )
    global_front_points = sort_by_method(
        tagged_points[tagged_points[GLOBAL_PARETO_COL]].copy(),
        extra_columns=["moo_run", "throughput_mbps"],
    )

    objective_config = pd.DataFrame(
        [
            {"objective": column_name, "label": label, "direction": direction}
            for column_name, label, direction in OBJECTIVES
        ]
    )

    export_frames = {
        "objective_config": objective_config,
        "method_summary_by_metric": build_method_summary_by_metric(method_summary),
        "run_summary": run_summary,
        "method_global_summary": run_method_summary,
        "method_summary": method_summary,
        "all_points_tagged": tagged_points,
        "method_pareto_points": method_front_points,
        "global_pareto_points": global_front_points,
    }
    if loaded_files is not None:
        export_frames["input_files"] = pd.DataFrame(
            {"input_file": [str(path) for path in loaded_files]}
        )
    if missing_methods:
        export_frames["missing_methods"] = pd.DataFrame({"method": missing_methods})
    return export_frames


def export_summary_workbook(frames: dict[str, pd.DataFrame], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def make_method_handles(methods: list[str]) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="#D3D3D3",
            markerfacecolor="#D3D3D3",
            markersize=7,
            alpha=0.8,
            label="All points",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            color="#111111",
            markersize=7,
            label="Global PF",
        ),
    ]
    for method_name in methods:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                color=METHOD_COLOR_MAP.get(method_name, "#999999"),
                markerfacecolor=METHOD_COLOR_MAP.get(method_name, "#999999"),
                markeredgecolor="white",
                markersize=6,
                label=METHOD_LABELS.get(method_name, method_name),
            )
        )
    return handles


def plot_single_projection(
    ax: plt.Axes,
    results: pd.DataFrame,
    methods: list[str],
    x_field: str,
    y_field: str,
    x_label: str,
    y_label: str,
    x_scale: str,
) -> None:
    background = results[[x_field, y_field]].dropna()
    if x_scale == "log":
        background = background[background[x_field] > 0]
    ax.scatter(
        background[x_field],
        background[y_field],
        s=12,
        c="#D3D3D3",
        alpha=0.35,
        edgecolors="none",
    )

    global_front = results[results[GLOBAL_PARETO_COL]].copy()
    if x_scale == "log":
        global_front = global_front[global_front[x_field] > 0]

    ax.scatter(
        global_front[x_field],
        global_front[y_field],
        s=70,
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.1,
        zorder=3,
    )

    for method_name in methods:
        method_front = results[
            (results["method"] == method_name) & results[METHOD_PARETO_COL]
        ].copy()
        if x_scale == "log":
            method_front = method_front[method_front[x_field] > 0]
        if method_front.empty:
            continue
        color = METHOD_COLOR_MAP.get(method_name, "#999999")
        for _, run_front in method_front.groupby("moo_run", sort=True):
            run_front = run_front.sort_values(by=x_field, kind="stable")
            if len(run_front) < 2:
                continue
            ax.plot(
                run_front[x_field],
                run_front[y_field],
                color=color,
                linewidth=1.4,
                alpha=0.55,
                zorder=4,
            )
        ax.scatter(
            method_front[x_field],
            method_front[y_field],
            s=34,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.3)
    if x_scale == "log":
        ax.set_xscale("log")


def prepare_three_objective_points(frame: pd.DataFrame) -> pd.DataFrame:
    points = frame[["ber", "papr_db", "throughput_mbps"]].dropna().copy()
    points = points[points["ber"] > 0]
    points["log10_ber"] = np.log10(points["ber"].to_numpy(dtype=float))
    return points


def plot_three_objective_front(ax: plt.Axes, results: pd.DataFrame, methods: list[str]) -> None:
    background = prepare_three_objective_points(results)
    ax.scatter(
        background["log10_ber"],
        background["papr_db"],
        background["throughput_mbps"],
        s=10,
        c="#D3D3D3",
        alpha=0.22,
        edgecolors="none",
        depthshade=False,
    )

    global_front = prepare_three_objective_points(results[results[GLOBAL_PARETO_COL]])
    ax.scatter(
        global_front["log10_ber"],
        global_front["papr_db"],
        global_front["throughput_mbps"],
        s=58,
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.0,
        depthshade=False,
    )

    for method_name in methods:
        method_front = prepare_three_objective_points(
            results[(results["method"] == method_name) & results[METHOD_PARETO_COL]]
        )
        if method_front.empty:
            continue
        color = METHOD_COLOR_MAP.get(method_name, "#999999")
        ax.scatter(
            method_front["log10_ber"],
            method_front["papr_db"],
            method_front["throughput_mbps"],
            s=30,
            color=color,
            edgecolors="white",
            linewidths=0.4,
            depthshade=False,
        )

    ax.set_xlabel("log10(BER)")
    ax.set_ylabel("PAPR (dB)")
    ax.set_zlabel("Throughput (Mbps)")
    ax.set_title("3D PF Projection: BER / PAPR / Throughput")
    ax.view_init(elev=24, azim=-135)


def plot_pareto_fronts(results: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    methods = [method_name for method_name in VALID_METHODS if method_name in set(results["method"])]

    fig = plt.figure(figsize=(20, 16), constrained_layout=True)
    ax_3d = fig.add_subplot(3, 3, 1, projection="3d")
    plot_three_objective_front(ax_3d, results, methods)

    axes = [fig.add_subplot(3, 3, index) for index in range(2, 8)]
    for ax, spec in zip(axes, PROJECTION_SPECS):
        plot_single_projection(ax, results, methods, *spec)

    total_points = int(len(results))
    total_global_pareto = int(results[GLOBAL_PARETO_COL].sum())
    fig.suptitle(
        f"All-Points Four-Objective Pareto Fronts ({total_points} points, PF points = {total_global_pareto})",
        fontsize=16,
    )
    fig.legend(
        handles=make_method_handles(methods),
        loc="lower center",
        ncol=min(4, len(methods) + 2),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, values: np.ndarray, formatter: str) -> None:
    finite_values = values[~np.isnan(values)]
    upper = float(finite_values.max()) if len(finite_values) else 0.0
    offset = upper * 0.03 if upper > 0 else 0.02
    for index, value in enumerate(values):
        if np.isnan(value):
            continue
        ax.text(index, value + offset, format(value, formatter), ha="center", va="bottom", fontsize=9)


def plot_metric_bars(ax: plt.Axes, summary: pd.DataFrame, field: str, title: str, formatter: str) -> None:
    methods = summary["method"].tolist()
    values = summary[field].to_numpy(dtype=float)
    labels = [METHOD_LABELS.get(method_name, method_name) for method_name in methods]
    colors = [METHOD_COLOR_MAP.get(method_name, "#999999") for method_name in methods]

    ax.bar(labels, values, color=colors, width=0.72)
    annotate_bars(ax, values, formatter)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_metric_summary(summary: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("global_pareto_points", "Global PF Points", ".0f"),
        ("global_front_contribution", "Share of All-Points Global PF", ".1%"),
        ("global_pareto_ratio", "Global PF Ratio Within Method", ".1%"),
        ("method_front_hypervolume_norm", "Method PF Hypervolume (higher is better)", ".3f"),
        ("method_front_igd_to_global_front", "Method PF IGD to Global PF (lower is better)", ".3f"),
        ("method_front_closest_to_global_ideal", "Method PF Distance to Ideal (lower is better)", ".3f"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for ax, (field, title, formatter) in zip(axes.ravel(), metric_specs):
        plot_metric_bars(ax, summary, field, title, formatter)

    fig.suptitle("All-Points Four-Objective Pareto Metrics by Method", fontsize=16)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    loaded_files: list[Path],
    missing_methods: list[str],
    results: pd.DataFrame,
    run_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
) -> None:
    display_columns = [
        "label",
        "num_runs",
        "total_points",
        "method_pareto_points",
        "global_pareto_points",
        "global_pareto_ratio",
        "global_front_contribution",
        "method_front_hypervolume_norm",
        "method_front_igd_to_global_front",
        "method_front_closest_to_global_ideal",
    ]
    print(f"Loaded {len(results)} rows from {len(loaded_files)} method files.")
    print(f"Result files: {', '.join(path.name for path in loaded_files)}")
    if missing_methods:
        print(f"Missing method files: {', '.join(missing_methods)}")
    print(f"Detected moo runs: {int(run_summary['moo_run'].nunique()) if not run_summary.empty else 0}")
    print(f"Matched methods: {', '.join(method_summary['label'].tolist())}")
    objective_text = ", ".join(
        f"{column_name} ({direction})" for column_name, _, direction in OBJECTIVES
    )
    print(f"Objectives: {objective_text}")
    print(f"All-points global Pareto points: {int(results[GLOBAL_PARETO_COL].sum())}")
    if not run_summary.empty:
        print(
            f"Average all-points global Pareto points per moo_run bucket: "
            f"{run_summary['global_pareto_points_in_run'].mean():.2f}"
        )
    print(method_summary[display_columns].to_string(index=False))
    compact_by_metric = build_method_summary_by_metric(method_summary)[
        [
            "metric",
            "direction",
            *method_summary["label"].astype(str).tolist(),
        ]
    ]
    compact_metrics = [
        "global_pareto_points",
        "global_front_contribution",
        "global_pareto_ratio",
        "method_front_hypervolume_norm",
        "method_front_igd_to_global_front",
        "method_front_gd_to_global_front",
        "method_front_closest_to_global_ideal",
        "global_to_method_pareto_ratio",
    ]
    compact_by_metric = compact_by_metric[
        compact_by_metric["metric"].isin(compact_metrics)
    ]
    print("\nTransposed core metrics (methods as columns):")
    print(compact_by_metric.to_string(index=False))


def main() -> None:
    selected_methods = resolve_selected_methods(SELECTED_METHODS)
    input_result_dir = Path(INPUT_RESULT_DIR)
    summary_file = Path(SUMMARY_FILE)
    front_figure = Path(FRONT_FIGURE)
    metric_figure = Path(METRIC_FIGURE)

    results, loaded_files, missing_methods = load_results(
        input_result_dir=input_result_dir,
        selected_methods=selected_methods,
        result_sheet_name=RESULT_SHEET_NAME,
    )
    four_objective_analysis = analyze_objective_set(results, FOUR_OBJECTIVES)
    tagged_results = four_objective_analysis["tagged_results"]
    run_summary = four_objective_analysis["run_summary"]
    method_summary = four_objective_analysis["method_summary"]
    export_frames = {
        "4obj_overall": four_objective_analysis["overall_eval"],
    }

    export_summary_workbook(export_frames, summary_file)
    plot_pareto_fronts(tagged_results, front_figure)
    plot_metric_summary(method_summary, metric_figure)
    print_summary(loaded_files, missing_methods, tagged_results, run_summary, method_summary)
    print(f"Saved summary workbook to: {summary_file}")
    print(f"Saved Pareto front figure to: {front_figure}")
    print(f"Saved metric figure to: {metric_figure}")


if __name__ == "__main__":
    main()
