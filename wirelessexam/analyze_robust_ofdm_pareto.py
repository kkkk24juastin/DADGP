# -*- coding: utf-8 -*-
"""
对稳健 OFDM 仿真工作簿进行后处理。

流程边界:
1. robust_select_ofdm_existing_noise.m 只负责重复运行 OFDM 仿真，并导出
   经验均值/方差列。
2. 本脚本负责计算稳健质量损失、稳健 Pareto 标记、稳健最优行和排序。

默认输入:
    result/robust_simulation_<method>.xlsx, sheet all_candidates

默认输出:
    result/robust_selection_<method>.xlsx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from config import BASE_DIR, MOO_TARGET_VALUES, VALID_METHODS


RESULT_DIR = BASE_DIR / "result"

# SELECTED_METHODS = None 表示扫描全部 VALID_METHODS。
# 示例: SELECTED_METHODS = ["dadgp"]
SELECTED_METHODS: list[str] | None = None
INPUT_RESULT_DIR = RESULT_DIR
OUTPUT_RESULT_DIR = RESULT_DIR
SUMMARY_OUTPUT_FILE = RESULT_DIR / "robust_pareto_4obj_per_run_eval.xlsx"

INPUT_PREFIX = "robust_simulation_"
OUTPUT_PREFIX = "robust_selection_"
INPUT_SHEET_NAME = "all_candidates"
INPUT_METADATA_SHEET_NAME = "metadata"
ALL_CANDIDATES_SHEET_NAME = "all_candidates"
ROBUST_PARETO_SHEET_NAME = "robust_pareto"
ROBUST_BEST_SHEET_NAME = "robust_best"
METADATA_SHEET_NAME = "metadata"

X_COLUMNS = ["x1", "x2", "x3", "x4"]
IDENTITY_COLUMNS = ["method", "moo_run", "solution_idx"]
ROBUST_MEAN_COLUMNS = [
    "robust_mean_throughput_mbps",
    "robust_mean_ber",
    "robust_mean_papr_db",
]
ROBUST_VAR_COLUMNS = [
    "robust_var_throughput_mbps",
    "robust_var_ber",
    "robust_var_papr_db",
]
ROBUST_LOSS_COLUMNS = [
    "robust_loss_task1",
    "robust_loss_task2",
    "robust_loss_task3",
]
ROBUST_TOTAL_LOSS_COLUMN = "robust_total_loss"
ROBUST_PARETO_COLUMN = "is_robust_pareto"
ROBUST_BEST_COLUMN = "is_robust_best"
ROBUST_RANK_COLUMN = "robust_rank"

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

FOUR_OBJECTIVES = [
    ("throughput_mbps", "Throughput (Mbps)", "max"),
    ("ber", "BER", "min"),
    ("papr_db", "PAPR (dB)", "min"),
    ("energy_efficiency", "Energy Efficiency (bit/J)", "max"),
]
FOUR_OBJECTIVE_COLUMNS = [column_name for column_name, _, _ in FOUR_OBJECTIVES]
FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN = "is_4obj_run_global_pareto"
FOUR_OBJECTIVE_METHOD_PARETO_COLUMN = "is_4obj_run_method_pareto"

PER_RUN_COMPOSITE_METRICS = [
    ("global_pareto_hit_run_coverage", False),
    ("global_front_contribution", False),
    ("method_front_hv_norm", False),
    ("igd_to_global_front", True),
    ("closest_to_global_ideal", True),
]

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
    "spacing",
    "pf_mean_throughput",
    "pf_mean_ber",
    "pf_mean_papr",
    "pf_mean_energy_efficiency",
    "global_pf_mean_throughput",
    "global_pf_mean_ber",
    "global_pf_mean_papr",
    "global_pf_mean_energy_efficiency",
    "composite_score",
]

REQUIRED_INPUT_COLUMNS = X_COLUMNS + ROBUST_MEAN_COLUMNS + ROBUST_VAR_COLUMNS
NUMERIC_COLUMNS = (
    X_COLUMNS
    + ["moo_run", "solution_idx"]
    + ROBUST_MEAN_COLUMNS
    + ROBUST_VAR_COLUMNS
    + ROBUST_LOSS_COLUMNS
    + [ROBUST_TOTAL_LOSS_COLUMN, ROBUST_RANK_COLUMN]
)


def resolve_selected_methods(selected_methods_config: list[str] | None) -> list[str]:
    if selected_methods_config is None:
        return VALID_METHODS.copy()

    if not isinstance(selected_methods_config, list):
        raise TypeError("SELECTED_METHODS must be list[str] or None.")

    methods = [str(item).strip() for item in selected_methods_config if str(item).strip()]
    if not methods:
        raise ValueError("SELECTED_METHODS must not be an empty list.")

    invalid_methods = [method for method in methods if method not in VALID_METHODS]
    if invalid_methods:
        raise ValueError(
            f"Invalid methods: {', '.join(invalid_methods)}. "
            f"Available methods: {', '.join(VALID_METHODS)}"
        )

    return [method for method in VALID_METHODS if method in set(methods)]


def method_order_map() -> dict[str, int]:
    return {method: index for index, method in enumerate(VALID_METHODS)}


def sort_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "method",
        ROBUST_RANK_COLUMN,
        ROBUST_TOTAL_LOSS_COLUMN,
        *ROBUST_LOSS_COLUMNS,
        "moo_run",
        "solution_idx",
    ]
    sort_columns = [column for column in sort_columns if column in frame.columns]
    order = method_order_map()
    return frame.sort_values(
        by=sort_columns,
        key=lambda column: column.map(order) if column.name == "method" else column,
        kind="stable",
    ).reset_index(drop=True)


def compute_pareto_mask(objectives: np.ndarray) -> np.ndarray:
    if objectives.ndim != 2:
        raise ValueError("objectives must be a 2D array.")
    if objectives.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    pareto_mask = np.ones(objectives.shape[0], dtype=bool)
    for index in range(objectives.shape[0]):
        dominated_by_other = np.all(objectives <= objectives[index], axis=1) & np.any(
            objectives < objectives[index], axis=1
        )
        dominated_by_other[index] = False
        if dominated_by_other.any():
            pareto_mask[index] = False
    return pareto_mask


def ensure_identity_columns(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    results = frame.copy()
    results["method"] = method
    if "moo_run" not in results.columns:
        results["moo_run"] = 1
    if "solution_idx" not in results.columns:
        results["solution_idx"] = np.arange(1, len(results) + 1, dtype=np.int64)
    return results


def normalize_columns(frame: pd.DataFrame, method: str, source_file: Path) -> pd.DataFrame:
    lower_to_actual = {str(column).strip().lower(): column for column in frame.columns}
    rename_map = {}
    canonical_columns = set(IDENTITY_COLUMNS + REQUIRED_INPUT_COLUMNS)
    for canonical_name in canonical_columns:
        actual_name = lower_to_actual.get(canonical_name.lower())
        if actual_name is not None and actual_name != canonical_name:
            rename_map[actual_name] = canonical_name
    if rename_map:
        frame = frame.rename(columns=rename_map)

    missing_columns = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{source_file} is missing required columns: {', '.join(missing_columns)}"
        )

    frame = ensure_identity_columns(frame, method)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_numeric = [
        column
        for column in REQUIRED_INPUT_COLUMNS
        if frame[column].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(
            f"{source_file} contains invalid numeric values in: "
            f"{', '.join(invalid_numeric)}"
        )

    frame["moo_run"] = frame["moo_run"].fillna(1).astype(np.int64)
    default_solution_idx = pd.Series(
        np.arange(1, len(frame) + 1, dtype=np.int64),
        index=frame.index,
    )
    frame["solution_idx"] = frame["solution_idx"].where(
        frame["solution_idx"].notna(),
        default_solution_idx,
    ).astype(np.int64)
    return frame


def add_robust_losses(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    targets = np.asarray(MOO_TARGET_VALUES, dtype=np.float64).reshape(1, -1)
    if targets.shape[1] != len(ROBUST_MEAN_COLUMNS):
        raise ValueError(
            "MOO_TARGET_VALUES must contain exactly three target values."
        )

    means = results[ROBUST_MEAN_COLUMNS].to_numpy(dtype=np.float64)
    variances = results[ROBUST_VAR_COLUMNS].to_numpy(dtype=np.float64)
    variances = np.clip(variances, a_min=0.0, a_max=None)
    losses = (means - targets) ** 2 + variances

    for index, column in enumerate(ROBUST_LOSS_COLUMNS):
        results[column] = losses[:, index]
    results[ROBUST_TOTAL_LOSS_COLUMN] = losses.sum(axis=1)
    return results


def add_robust_selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    objectives = results[ROBUST_LOSS_COLUMNS].to_numpy(dtype=np.float64)
    results[ROBUST_PARETO_COLUMN] = compute_pareto_mask(objectives)

    sort_order = results.sort_values(
        by=[
            ROBUST_TOTAL_LOSS_COLUMN,
            *ROBUST_LOSS_COLUMNS,
            "moo_run",
            "solution_idx",
        ],
        kind="stable",
    ).index
    rank = pd.Series(np.arange(1, len(results) + 1, dtype=np.int64), index=sort_order)
    results[ROBUST_RANK_COLUMN] = rank.sort_index().to_numpy()

    results[ROBUST_BEST_COLUMN] = False
    for (_, moo_run), run_rows in results.groupby(["method", "moo_run"], sort=True):
        best_index = run_rows.sort_values(
            by=[
                ROBUST_TOTAL_LOSS_COLUMN,
                *ROBUST_LOSS_COLUMNS,
                "solution_idx",
            ],
            kind="stable",
        ).index[0]
        results.loc[best_index, ROBUST_BEST_COLUMN] = True

    return sort_candidates(results)


def read_metadata(source_file: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(source_file, sheet_name=INPUT_METADATA_SHEET_NAME)
    except ValueError:
        return pd.DataFrame()


def build_metadata(
    method: str,
    source_file: Path,
    output_file: Path,
    all_candidates: pd.DataFrame,
    robust_pareto: pd.DataFrame,
    robust_best: pd.DataFrame,
    input_metadata: pd.DataFrame,
) -> pd.DataFrame:
    source_metadata = {}
    if not input_metadata.empty:
        first_row = input_metadata.iloc[0].to_dict()
        for column in [
            "simulation_seeds",
            "num_repetitions",
            "modulation_order",
            "noise_source",
            "fixed_noise_figure_db",
        ]:
            if column in first_row:
                source_metadata[column] = first_row[column]

    metadata = {
        "method": method,
        "input_simulation_file": str(source_file),
        "input_sheet": INPUT_SHEET_NAME,
        "output_selection_file": str(output_file),
        "target_values": tuple(float(value) for value in MOO_TARGET_VALUES),
        "robust_objective_definition": (
            "robust_loss_i = (mean_seed(y_i) - target_i)^2 + var_seed(y_i)"
        ),
        "pareto_objectives": ", ".join(ROBUST_LOSS_COLUMNS),
        "pareto_scope": "all candidates in the method-level robust simulation workbook",
        "robust_best_scope": "one minimum robust_total_loss row per method and moo_run",
        "rank_definition": "1-based order by robust_total_loss within the method workbook",
        "num_candidates": int(len(all_candidates)),
        "num_robust_pareto": int(len(robust_pareto)),
        "num_robust_best": int(len(robust_best)),
        **source_metadata,
    }
    return pd.DataFrame([metadata])


def load_method_simulation(method: str) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    source_file = INPUT_RESULT_DIR / f"{INPUT_PREFIX}{method}.xlsx"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing robust simulation file: {source_file}")

    frame = pd.read_excel(source_file, sheet_name=INPUT_SHEET_NAME)
    if frame.empty:
        raise ValueError(f"{source_file} sheet {INPUT_SHEET_NAME} is empty.")

    metadata = read_metadata(source_file)
    return normalize_columns(frame, method, source_file), metadata, source_file


def write_method_workbook(
    output_file: Path,
    all_candidates: pd.DataFrame,
    robust_pareto: pd.DataFrame,
    robust_best: pd.DataFrame,
    metadata: pd.DataFrame,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        all_candidates.to_excel(writer, sheet_name=ALL_CANDIDATES_SHEET_NAME, index=False)
        robust_pareto.to_excel(writer, sheet_name=ROBUST_PARETO_SHEET_NAME, index=False)
        robust_best.to_excel(writer, sheet_name=ROBUST_BEST_SHEET_NAME, index=False)
        metadata.to_excel(writer, sheet_name=METADATA_SHEET_NAME, index=False)


def process_method(method: str) -> tuple[Path, pd.DataFrame]:
    candidates, input_metadata, source_file = load_method_simulation(method)
    selected = add_robust_selection_columns(add_robust_losses(candidates))
    robust_pareto = selected[selected[ROBUST_PARETO_COLUMN]].copy()
    robust_pareto = sort_candidates(robust_pareto)
    robust_best = selected[selected[ROBUST_BEST_COLUMN]].copy()
    robust_best = sort_candidates(robust_best)

    output_file = OUTPUT_RESULT_DIR / f"{OUTPUT_PREFIX}{method}.xlsx"
    metadata = build_metadata(
        method=method,
        source_file=source_file,
        output_file=output_file,
        all_candidates=selected,
        robust_pareto=robust_pareto,
        robust_best=robust_best,
        input_metadata=input_metadata,
    )
    write_method_workbook(output_file, selected, robust_pareto, robust_best, metadata)

    print(
        f"{method}: candidates={len(selected)}, "
        f"robust_pareto={len(robust_pareto)}, "
        f"robust_best={len(robust_best)} -> {output_file}"
    )
    return output_file, selected


def build_four_objective_frame(selected_candidates: pd.DataFrame) -> pd.DataFrame:
    frame = selected_candidates.copy()
    frame["throughput_mbps"] = frame["robust_mean_throughput_mbps"]
    frame["ber"] = frame["robust_mean_ber"]
    frame["papr_db"] = frame["robust_mean_papr_db"]

    ptx_dbm = pd.to_numeric(frame["x1"], errors="coerce").to_numpy(dtype=float)
    throughput_mbps = pd.to_numeric(
        frame["throughput_mbps"], errors="coerce"
    ).to_numpy(dtype=float)
    pout_w = 10.0 ** ((ptx_dbm - 30.0) / 10.0)
    total_power_w = POWER_MODEL_P_BB + POWER_MODEL_P_RF + pout_w / POWER_MODEL_ETA_PA
    bitrate_bps = np.maximum(throughput_mbps, MIN_THROUGHPUT_MBPS) * 1e6
    energy_per_bit = total_power_w / bitrate_bps
    frame["energy_per_bit"] = energy_per_bit
    frame["energy_efficiency"] = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)

    required_columns = ["method", "moo_run", "solution_idx", *FOUR_OBJECTIVE_COLUMNS]
    invalid_rows = frame[required_columns].isna().any(axis=1)
    if invalid_rows.any():
        frame = frame.loc[~invalid_rows].copy()
    return sort_candidates(frame)


def build_four_objective_matrix(frame: pd.DataFrame) -> np.ndarray:
    objective_columns = []
    for column_name, _, direction in FOUR_OBJECTIVES:
        values = frame[column_name].to_numpy(dtype=float)
        objective_columns.append(-values if direction == "max" else values)
    return np.column_stack(objective_columns)


def add_four_objective_pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    results = frame.copy()
    results[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN] = compute_pareto_mask(
        build_four_objective_matrix(results)
    )
    results[FOUR_OBJECTIVE_METHOD_PARETO_COLUMN] = False

    for method_name in VALID_METHODS:
        method_rows = results[results["method"] == method_name]
        if method_rows.empty:
            continue
        method_mask = compute_pareto_mask(build_four_objective_matrix(method_rows))
        results.loc[method_rows.index, FOUR_OBJECTIVE_METHOD_PARETO_COLUMN] = method_mask

    return sort_four_objective_points(
        results,
        extra_columns=[
            "moo_run",
            "solution_idx",
            FOUR_OBJECTIVE_METHOD_PARETO_COLUMN,
            FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN,
        ],
    )


def sort_four_objective_points(
    frame: pd.DataFrame, extra_columns: list[str] | None = None
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    sort_columns = ["method"]
    if extra_columns:
        sort_columns.extend([column for column in extra_columns if column in frame.columns])
    order = method_order_map()
    return frame.sort_values(
        by=sort_columns,
        key=lambda column: column.map(order) if column.name == "method" else column,
        kind="stable",
    ).reset_index(drop=True)


def normalize_four_objective_for_cost(
    frame: pd.DataFrame,
    minima: dict[str, float],
    maxima: dict[str, float],
) -> np.ndarray:
    if frame.empty:
        return np.empty((0, len(FOUR_OBJECTIVES)), dtype=float)

    normalized_columns = []
    for column_name, _, direction in FOUR_OBJECTIVES:
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


def _unique_rows(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        column_count = values.shape[1] if values.ndim == 2 else len(FOUR_OBJECTIVES)
        return values.reshape(0, column_count)
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


def _metric_key(column_name: str) -> str:
    return {
        "throughput_mbps": "throughput",
        "ber": "ber",
        "papr_db": "papr",
        "energy_efficiency": "energy_efficiency",
    }.get(column_name, column_name)


def _best_metric_value(frame: pd.DataFrame, column_name: str, direction: str) -> float:
    if frame.empty:
        return float("nan")
    if direction == "max":
        return float(frame[column_name].max())
    return float(frame[column_name].min())


def summarize_four_objective_front(
    front: pd.DataFrame,
    minima: dict[str, float],
    maxima: dict[str, float],
) -> dict[str, float]:
    nan_value = float("nan")
    if front.empty:
        metrics = {
            "closest_to_global_ideal": nan_value,
            "mean_distance_to_global_ideal": nan_value,
            "mean_normalized_cost_sum": nan_value,
            "mean_pairwise_distance": nan_value,
            "mean_nearest_neighbor_distance": nan_value,
            "spacing": nan_value,
            "mean_normalized_front_span": nan_value,
        }
        for column_name, _, _ in FOUR_OBJECTIVES:
            metric_key = _metric_key(column_name)
            metrics[f"best_{metric_key}"] = nan_value
            metrics[f"front_mean_{metric_key}"] = nan_value
        return metrics

    normalized_front = normalize_four_objective_for_cost(front, minima, maxima)
    distances_to_ideal = np.sqrt(np.sum(normalized_front**2, axis=1))
    normalized_cost_sums = normalized_front.sum(axis=1)
    pairwise = _pairwise_distances(normalized_front)
    nearest = _nearest_neighbor_distances(normalized_front)
    span_values = normalized_front.max(axis=0) - normalized_front.min(axis=0)

    metrics = {
        "closest_to_global_ideal": float(distances_to_ideal.min()),
        "mean_distance_to_global_ideal": float(distances_to_ideal.mean()),
        "mean_normalized_cost_sum": float(normalized_cost_sums.mean()),
        "mean_pairwise_distance": float(pairwise.mean()) if len(pairwise) else nan_value,
        "mean_nearest_neighbor_distance": float(nearest.mean()) if len(nearest) else nan_value,
        "spacing": float(nearest.std(ddof=0)) if len(nearest) else nan_value,
        "mean_normalized_front_span": float(span_values.mean()),
    }
    for column_name, _, direction in FOUR_OBJECTIVES:
        metric_key = _metric_key(column_name)
        metrics[f"best_{metric_key}"] = _best_metric_value(front, column_name, direction)
        metrics[f"front_mean_{metric_key}"] = float(front[column_name].mean())
    return metrics


def summarize_four_objective_indicators(
    front: pd.DataFrame,
    reference_front: pd.DataFrame,
    minima: dict[str, float],
    maxima: dict[str, float],
    global_hypervolume_norm: float,
) -> dict[str, float]:
    normalized_front = normalize_four_objective_for_cost(front, minima, maxima)
    normalized_reference_front = normalize_four_objective_for_cost(
        reference_front, minima, maxima
    )
    hypervolume, hypervolume_norm = _hypervolume_values(normalized_front)
    metrics = {
        "hypervolume": hypervolume,
        "hypervolume_norm": hypervolume_norm,
        "hypervolume_ratio_to_global": (
            float(hypervolume_norm / global_hypervolume_norm)
            if global_hypervolume_norm > 0
            else 0.0
        ),
    }

    if len(normalized_front) == 0 or len(normalized_reference_front) == 0:
        metrics.update(
            {
                "gd_to_global_front": float("nan"),
                "gd_plus_to_global_front": float("nan"),
                "igd_to_global_front": float("nan"),
                "igd_plus_to_global_front": float("nan"),
            }
        )
        return metrics

    reference = _unique_rows(normalized_reference_front)
    front_points = _unique_rows(normalized_front)
    metrics.update(
        {
            "gd_to_global_front": float(GD(reference)(front_points)),
            "gd_plus_to_global_front": float(GDPlus(reference)(front_points)),
            "igd_to_global_front": float(IGD(reference)(front_points)),
            "igd_plus_to_global_front": float(IGDPlus(reference)(front_points)),
        }
    )
    return metrics


def build_four_objective_run_summary(tagged_run_frames: list[pd.DataFrame]) -> pd.DataFrame:
    summary_rows = []
    for tagged_run in tagged_run_frames:
        if tagged_run.empty:
            continue
        moo_run = int(tagged_run["moo_run"].iloc[0])
        global_front = tagged_run[tagged_run[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN]].copy()
        row = {
            "moo_run": moo_run,
            "total_points": int(len(tagged_run)),
            "global_pareto_points_in_run": int(len(global_front)),
            "methods_present": int(tagged_run["method"].nunique()),
            "methods_with_global_pareto": int(global_front["method"].nunique()),
        }
        for column_name, _, direction in FOUR_OBJECTIVES:
            metric_key = _metric_key(column_name)
            row[f"best_global_{metric_key}"] = _best_metric_value(
                global_front, column_name, direction
            )
            row[f"mean_global_{metric_key}"] = (
                float(global_front[column_name].mean()) if len(global_front) else float("nan")
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary
    return summary.sort_values(by=["moo_run"], kind="stable").reset_index(drop=True)


def build_four_objective_run_method_summary(tagged_run: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    minima = {column: float(tagged_run[column].min()) for column, _, _ in FOUR_OBJECTIVES}
    maxima = {column: float(tagged_run[column].max()) for column, _, _ in FOUR_OBJECTIVES}
    global_front = tagged_run[tagged_run[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN]].copy()
    _, global_hypervolume_norm = _hypervolume_values(
        normalize_four_objective_for_cost(global_front, minima, maxima)
    )
    total_global_pareto = int(tagged_run[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN].sum())
    run_global_front_contribution_denominator = max(total_global_pareto, 1)

    for method_name in VALID_METHODS:
        method_rows = tagged_run[tagged_run["method"] == method_name].copy()
        if method_rows.empty:
            continue

        method_front = method_rows[
            method_rows[FOUR_OBJECTIVE_METHOD_PARETO_COLUMN]
        ].copy()
        global_front_hits = method_rows[
            method_rows[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN]
        ].copy()
        method_front_metrics = summarize_four_objective_front(
            method_front, minima, maxima
        )
        method_front_indicators = summarize_four_objective_indicators(
            method_front,
            global_front,
            minima,
            maxima,
            global_hypervolume_norm,
        )
        global_hit_metrics = summarize_four_objective_front(
            global_front_hits, minima, maxima
        )

        row = {
            "method": method_name,
            "label": METHOD_LABELS.get(method_name, method_name),
            "moo_run": int(method_rows["moo_run"].iloc[0]),
            "total_points": int(len(method_rows)),
            "method_pareto_points": int(len(method_front)),
            "method_pareto_ratio": float(len(method_front) / len(method_rows)),
            "global_pareto_points": int(len(global_front_hits)),
            "global_pareto_ratio": float(len(global_front_hits) / len(method_rows)),
            "run_global_front_contribution": float(
                len(global_front_hits) / run_global_front_contribution_denominator
            ),
            "run_global_pareto_points": total_global_pareto,
            "global_front_hypervolume_norm": global_hypervolume_norm,
            "method_front_hv_norm": method_front_indicators["hypervolume_norm"],
            "hv_ratio_to_global_front": method_front_indicators[
                "hypervolume_ratio_to_global"
            ],
            "igd_to_global_front": method_front_indicators["igd_to_global_front"],
            "igd_plus_to_global_front": method_front_indicators[
                "igd_plus_to_global_front"
            ],
            "gd_to_global_front": method_front_indicators["gd_to_global_front"],
            "gd_plus_to_global_front": method_front_indicators[
                "gd_plus_to_global_front"
            ],
            "closest_to_global_ideal": method_front_metrics[
                "closest_to_global_ideal"
            ],
            "spacing": method_front_metrics["spacing"],
            "mean_normalized_front_span": method_front_metrics[
                "mean_normalized_front_span"
            ],
        }
        for column_name, _, _ in FOUR_OBJECTIVES:
            metric_key = _metric_key(column_name)
            row[f"pf_mean_{metric_key}"] = method_front_metrics[
                f"front_mean_{metric_key}"
            ]
            row[f"global_pf_mean_{metric_key}"] = global_hit_metrics[
                f"front_mean_{metric_key}"
            ]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary
    return sort_four_objective_points(summary, extra_columns=["moo_run"])


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


def build_four_objective_per_run_eval(
    run_method_summary: pd.DataFrame,
) -> pd.DataFrame:
    if run_method_summary.empty:
        return pd.DataFrame(columns=PER_RUN_EXPORT_COLUMNS)

    summary_rows = []
    for method_name in VALID_METHODS:
        method_rows = run_method_summary[
            run_method_summary["method"] == method_name
        ].copy()
        if method_rows.empty:
            continue

        source = method_rows.iloc[0]
        row = {
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
            "global_pareto_ratio": float(method_rows["global_pareto_ratio"].mean()),
            "avg_method_pareto_points_per_run": float(
                method_rows["method_pareto_points"].mean()
            ),
            "method_pareto_ratio": float(method_rows["method_pareto_ratio"].mean()),
            "method_front_hv_norm": float(method_rows["method_front_hv_norm"].mean()),
            "hv_ratio_to_global_front": float(
                method_rows["hv_ratio_to_global_front"].mean()
            ),
            "igd_to_global_front": float(method_rows["igd_to_global_front"].mean()),
            "igd_plus_to_global_front": float(
                method_rows["igd_plus_to_global_front"].mean()
            ),
            "gd_to_global_front": float(method_rows["gd_to_global_front"].mean()),
            "gd_plus_to_global_front": float(
                method_rows["gd_plus_to_global_front"].mean()
            ),
            "closest_to_global_ideal": float(
                method_rows["closest_to_global_ideal"].mean()
            ),
            "spacing": float(method_rows["spacing"].mean()),
        }
        for column_name, _, _ in FOUR_OBJECTIVES:
            metric_key = _metric_key(column_name)
            row[f"pf_mean_{metric_key}"] = float(
                method_rows[f"pf_mean_{metric_key}"].mean()
            )
            row[f"global_pf_mean_{metric_key}"] = float(
                method_rows[f"global_pf_mean_{metric_key}"].mean()
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    ranked = add_composite_ranking(summary, PER_RUN_COMPOSITE_METRICS)
    export_columns = [column for column in PER_RUN_EXPORT_COLUMNS if column in ranked.columns]
    return ranked[export_columns]


def build_four_objective_tagged_points(results: pd.DataFrame) -> pd.DataFrame:
    tagged_frames = []
    for _, run_rows in results.groupby("moo_run", sort=True):
        tagged_frames.append(add_four_objective_pareto_flags(run_rows))
    if not tagged_frames:
        return pd.DataFrame()
    return sort_four_objective_points(
        pd.concat(tagged_frames, ignore_index=True),
        extra_columns=[
            "moo_run",
            "solution_idx",
            FOUR_OBJECTIVE_METHOD_PARETO_COLUMN,
            FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN,
        ],
    )


def build_best_point_summary(
    results: pd.DataFrame,
    selection_name: str,
    selected_rows: list[pd.Series],
) -> pd.DataFrame:
    if not selected_rows:
        return pd.DataFrame()

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    summary = (
        selected.groupby("method", as_index=False)
        .agg(
            selected_points=("moo_run", "count"),
            mean_robust_total_loss=(ROBUST_TOTAL_LOSS_COLUMN, "mean"),
            mean_throughput_mbps=("throughput_mbps", "mean"),
            mean_ber=("ber", "mean"),
            mean_papr_db=("papr_db", "mean"),
            mean_energy_efficiency=("energy_efficiency", "mean"),
        )
    )
    summary["selection"] = selection_name
    summary["label"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    return sort_four_objective_points(summary)


def build_closest_ideal_best_points(results: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for _, run_rows in results.groupby("moo_run", sort=True):
        minima = {column: float(run_rows[column].min()) for column, _, _ in FOUR_OBJECTIVES}
        maxima = {column: float(run_rows[column].max()) for column, _, _ in FOUR_OBJECTIVES}
        for method_name in VALID_METHODS:
            method_rows = run_rows[run_rows["method"] == method_name].copy()
            if method_rows.empty:
                continue
            normalized = normalize_four_objective_for_cost(method_rows, minima, maxima)
            distances = np.sqrt(np.sum(normalized**2, axis=1))
            method_rows["ideal_distance"] = distances
            selected_rows.append(
                method_rows.sort_values(
                    by=[
                        "ideal_distance",
                        "throughput_mbps",
                        "ber",
                        "papr_db",
                        "energy_efficiency",
                        "solution_idx",
                    ],
                    ascending=[True, False, True, True, False, True],
                    kind="stable",
                ).iloc[0]
            )

    summary = build_best_point_summary(
        results,
        "one closest four-objective ideal point per method and moo_run",
        selected_rows,
    )
    if summary.empty:
        return summary
    selected = pd.DataFrame(selected_rows)
    ideal_distance = (
        selected.groupby("method", as_index=False)
        .agg(mean_ideal_distance=("ideal_distance", "mean"))
    )
    return summary.merge(ideal_distance, on="method", how="left")


def build_min_total_loss_best_points(results: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for (_, _), run_rows in results.groupby(["method", "moo_run"], sort=True):
        selected_rows.append(
            run_rows.sort_values(
                by=[ROBUST_TOTAL_LOSS_COLUMN, *ROBUST_LOSS_COLUMNS, "solution_idx"],
                kind="stable",
            ).iloc[0]
        )

    return build_best_point_summary(
        results,
        "one minimum robust_total_loss point per method and moo_run",
        selected_rows,
    )


def build_four_objective_summary_frames(
    selected_candidates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    four_objective_results = build_four_objective_frame(selected_candidates)
    tagged_points = build_four_objective_tagged_points(four_objective_results)

    run_method_frames = []
    tagged_run_frames = []
    for _, run_rows in four_objective_results.groupby("moo_run", sort=True):
        tagged_run = add_four_objective_pareto_flags(run_rows)
        tagged_run_frames.append(tagged_run)
        run_method_frames.append(build_four_objective_run_method_summary(tagged_run))

    run_method_summary = (
        pd.concat(run_method_frames, ignore_index=True)
        if run_method_frames
        else pd.DataFrame()
    )
    run_summary = build_four_objective_run_summary(tagged_run_frames)
    per_run_eval = build_four_objective_per_run_eval(run_method_summary)
    method_front_points = tagged_points[
        tagged_points[FOUR_OBJECTIVE_METHOD_PARETO_COLUMN]
    ].copy()
    global_front_points = tagged_points[
        tagged_points[FOUR_OBJECTIVE_GLOBAL_PARETO_COLUMN]
    ].copy()
    objective_config = pd.DataFrame(
        [
            {"objective": column_name, "label": label, "direction": direction}
            for column_name, label, direction in FOUR_OBJECTIVES
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "summary_scope": "four-objective per-run robust evaluation",
                "input_workbooks": f"{INPUT_RESULT_DIR}/{INPUT_PREFIX}<method>.xlsx",
                "objectives": ", ".join(
                    f"{name} ({direction})" for name, _, direction in FOUR_OBJECTIVES
                ),
                "energy_efficiency_definition": (
                    "1 / energy_per_bit; energy_per_bit = "
                    "(0.2 + 0.8 + 10^((x1 - 30)/10) / 0.35) / "
                    "max(robust_mean_throughput_mbps, 1e-6) / 1e6"
                ),
                "per_run_scope": (
                    "Pareto fronts and indicators are recomputed inside each moo_run "
                    "bucket across all selected methods, then averaged by method."
                ),
                "composite_rank_definition": (
                    "rank sum of global_pareto_hit_run_coverage, "
                    "global_front_contribution, method_front_hv_norm, "
                    "igd_to_global_front, and closest_to_global_ideal"
                ),
                "hv_reference_point": HV_REFERENCE_POINT_VALUE,
            }
        ]
    )

    return {
        "metadata": metadata,
        "objective_config": objective_config,
        "4obj_per_run_eval": per_run_eval,
        "4obj_run_summary": run_summary,
        "4obj_run_method_summary": sort_four_objective_points(
            run_method_summary, extra_columns=["moo_run"]
        ),
        "4obj_all_points_tagged": tagged_points,
        "4obj_method_front_points": sort_four_objective_points(
            method_front_points,
            extra_columns=["moo_run", "solution_idx"],
        ),
        "4obj_global_front_points": sort_four_objective_points(
            global_front_points,
            extra_columns=["moo_run", "solution_idx"],
        ),
        "4obj_closest_ideal_best": build_closest_ideal_best_points(
            four_objective_results
        ),
        "robust_total_loss_best": build_min_total_loss_best_points(
            four_objective_results
        ),
    }


def write_summary_workbook(frames: dict[str, pd.DataFrame], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    methods = resolve_selected_methods(SELECTED_METHODS)
    output_paths: list[Path] = []
    selected_candidate_frames: list[pd.DataFrame] = []
    missing_methods: list[str] = []

    for method in methods:
        try:
            output_path, selected = process_method(method)
            output_paths.append(output_path)
            selected_candidate_frames.append(selected)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            missing_methods.append(method)

    if not output_paths:
        raise RuntimeError(
            "No robust selection workbooks were generated. "
            "Run robust_select_ofdm_existing_noise.m first."
        )

    all_selected_candidates = pd.concat(selected_candidate_frames, ignore_index=True)
    summary_frames = build_four_objective_summary_frames(all_selected_candidates)
    write_summary_workbook(summary_frames, SUMMARY_OUTPUT_FILE)

    print("\nFinished robust OFDM Pareto analysis.")
    print(f"Generated workbooks: {len(output_paths)}")
    print(f"Generated four-objective per-run summary: {SUMMARY_OUTPUT_FILE}")
    if missing_methods:
        print(f"Missing simulation files: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()
