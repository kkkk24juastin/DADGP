# -*- coding: utf-8 -*-
"""
Compute true OFDM EVM for every candidate point of every method.

Inputs:
    result/robust_simulation_<method>.xlsx, sheet all_candidates

Outputs:
    result/ofdm_evm_all_points.xlsx

The Python script owns file IO and orchestration. MATLAB Engine is used only to
reuse the current OFDM simulator (`ofdm_link_quality_ex.m`) through the
batch helper `evaluate_ofdm_evm_batch.m`.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

try:
    import matlab
    import matlab.engine
except ImportError as exc:  # pragma: no cover - only hit without MATLAB Engine.
    matlab = None
    MATLAB_ENGINE_IMPORT_ERROR = exc
else:
    MATLAB_ENGINE_IMPORT_ERROR = None

from config import BASE_DIR


# ---------------------------------------------------------------------------
# Explicit run configuration. Edit these values directly.
# ---------------------------------------------------------------------------
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

INPUT_RESULT_DIR = BASE_DIR / "result"
INPUT_PREFIX = "robust_simulation_"
INPUT_SHEET_NAME = "all_candidates"
OUTPUT_FILE = INPUT_RESULT_DIR / "ofdm_evm_all_points.xlsx"

MODULATION_ORDER = 16
SIMULATION_SEED = 42

X_COLUMNS = ["x1", "x2", "x3", "x4"]
EVM_COLUMNS = [
    "real_evm_rms",
    "real_evm_db",
    "evm_eval_ber",
    "evm_eval_papr_db",
    "evm_eval_throughput_mbps",
    "evm_eval_snr_db",
    "evm_modulation_order",
    "evm_simulation_seed",
]


def ensure_matlab_engine_available() -> None:
    if MATLAB_ENGINE_IMPORT_ERROR is not None:
        raise ImportError(
            "matlab.engine is not importable in this Python environment. "
            "Install/configure MATLAB Engine for Python before running this script."
        ) from MATLAB_ENGINE_IMPORT_ERROR


def method_label(method: str) -> str:
    labels = {
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
    return labels.get(method, method)


def load_method_candidates(method: str) -> tuple[pd.DataFrame, Path]:
    source_file = INPUT_RESULT_DIR / f"{INPUT_PREFIX}{method}.xlsx"
    if not source_file.exists():
        raise FileNotFoundError(f"Missing input workbook: {source_file}")

    frame = pd.read_excel(source_file, sheet_name=INPUT_SHEET_NAME)
    if frame.empty:
        raise ValueError(f"{source_file} sheet {INPUT_SHEET_NAME} is empty.")

    missing_columns = [column for column in X_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"{source_file} is missing required columns: {', '.join(missing_columns)}"
        )

    frame = frame.copy()
    frame["method"] = method
    frame["label"] = method_label(method)
    if "moo_run" not in frame.columns:
        frame["moo_run"] = 1
    if "solution_idx" not in frame.columns:
        frame["solution_idx"] = np.arange(1, len(frame) + 1)

    for column in X_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=X_COLUMNS).reset_index(drop=True)
    return frame, source_file


def as_matlab_column(values: pd.Series):
    array = values.astype(float).to_numpy().reshape(-1, 1)
    return matlab.double(array.tolist())


def compute_method_evm(engine, frame: pd.DataFrame) -> pd.DataFrame:
    x1 = as_matlab_column(frame["x1"])
    x2 = as_matlab_column(frame["x2"])
    x3 = as_matlab_column(frame["x3"])
    x4 = as_matlab_column(frame["x4"])

    start = perf_counter()
    raw_metrics = engine.evaluate_ofdm_evm_batch(
        x1,
        x2,
        x3,
        x4,
        float(MODULATION_ORDER),
        float(SIMULATION_SEED),
        nargout=1,
    )
    elapsed_sec = perf_counter() - start

    metrics = np.asarray(raw_metrics, dtype=float)
    if metrics.ndim != 2 or metrics.shape[1] != 5:
        raise ValueError(f"Unexpected EVM metric matrix shape: {metrics.shape}")

    result = frame.copy()
    result["real_evm_rms"] = metrics[:, 0]
    result["real_evm_db"] = 20.0 * np.log10(np.maximum(metrics[:, 0], 1e-300))
    result["evm_eval_ber"] = metrics[:, 1]
    result["evm_eval_papr_db"] = metrics[:, 2]
    result["evm_eval_throughput_mbps"] = metrics[:, 3]
    result["evm_eval_snr_db"] = metrics[:, 4]
    result["evm_modulation_order"] = MODULATION_ORDER
    result["evm_simulation_seed"] = SIMULATION_SEED
    result["evm_eval_elapsed_sec_total"] = elapsed_sec
    result["evm_eval_elapsed_sec_per_point"] = elapsed_sec / max(len(result), 1)
    return result


def build_method_summary(all_points: pd.DataFrame) -> pd.DataFrame:
    grouped = all_points.groupby(["method", "label"], sort=False)
    summary = grouped.agg(
        num_points=("real_evm_rms", "size"),
        min_evm_rms=("real_evm_rms", "min"),
        mean_evm_rms=("real_evm_rms", "mean"),
        median_evm_rms=("real_evm_rms", "median"),
        std_evm_rms=("real_evm_rms", "std"),
        max_evm_rms=("real_evm_rms", "max"),
        min_evm_db=("real_evm_db", "min"),
        mean_evm_db=("real_evm_db", "mean"),
        best_evm_moo_run=("moo_run", lambda values: np.nan),
        best_evm_solution_idx=("solution_idx", lambda values: np.nan),
    ).reset_index()

    best_rows = (
        all_points.sort_values(["method", "real_evm_rms"], kind="stable")
        .groupby("method", sort=False)
        .first()
        .reset_index()
    )
    best_lookup = best_rows.set_index("method")
    summary["best_evm_moo_run"] = summary["method"].map(best_lookup["moo_run"])
    summary["best_evm_solution_idx"] = summary["method"].map(
        best_lookup["solution_idx"]
    )
    summary["evm_rank_by_min"] = summary["min_evm_rms"].rank(
        method="min", ascending=True
    )
    summary["evm_rank_by_mean"] = summary["mean_evm_rms"].rank(
        method="min", ascending=True
    )
    return summary.sort_values("evm_rank_by_min", kind="stable").reset_index(drop=True)


def build_metadata(loaded_files: list[Path], elapsed_sec: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "all method candidate EVM evaluation",
                "input_pattern": str(INPUT_RESULT_DIR / f"{INPUT_PREFIX}<method>.xlsx"),
                "input_sheet": INPUT_SHEET_NAME,
                "output_file": str(OUTPUT_FILE),
                "modulation_order": MODULATION_ORDER,
                "simulation_seed": SIMULATION_SEED,
                "num_methods_requested": len(METHOD_ORDER),
                "num_methods_loaded": len(loaded_files),
                "elapsed_sec": elapsed_sec,
                "matlab_helper": "evaluate_ofdm_evm_batch.m",
                "evm_definition": (
                    "metrics.evm_rms from ofdm_link_quality_ex; RMS distance between "
                    "equalized received data symbols and transmitted data symbols, "
                    "normalized by transmitted symbol power."
                ),
            }
        ]
    )


def write_output(all_points: pd.DataFrame, loaded_files: list[Path], elapsed_sec: float) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    method_summary = build_method_summary(all_points)
    metadata = build_metadata(loaded_files, elapsed_sec)

    preferred_columns = [
        "method",
        "label",
        "moo_run",
        "solution_idx",
        *X_COLUMNS,
        *EVM_COLUMNS,
    ]
    ordered_columns = [
        column for column in preferred_columns if column in all_points.columns
    ] + [
        column
        for column in all_points.columns
        if column not in preferred_columns
    ]

    with pd.ExcelWriter(OUTPUT_FILE) as writer:
        all_points.reindex(columns=ordered_columns).to_excel(
            writer, sheet_name="all_points", index=False
        )
        method_summary.to_excel(writer, sheet_name="method_summary", index=False)
        metadata.to_excel(writer, sheet_name="metadata", index=False)


def main() -> None:
    ensure_matlab_engine_available()
    loaded_files: list[Path] = []
    evaluated_frames: list[pd.DataFrame] = []
    total_start = perf_counter()

    print("Starting MATLAB Engine...")
    engine = matlab.engine.start_matlab()
    try:
        engine.addpath(str(BASE_DIR), nargout=0)

        for method in METHOD_ORDER:
            try:
                frame, source_file = load_method_candidates(method)
            except FileNotFoundError as exc:
                print(f"Warning: {exc}")
                continue

            print(f"\n=== Computing EVM | method={method} | points={len(frame)} ===")
            evaluated = compute_method_evm(engine, frame)
            loaded_files.append(source_file)
            evaluated_frames.append(evaluated)
            print(
                f"  min EVM={evaluated['real_evm_rms'].min():.6g}, "
                f"mean EVM={evaluated['real_evm_rms'].mean():.6g}"
            )
    finally:
        engine.quit()

    if not evaluated_frames:
        raise RuntimeError("No method workbooks were evaluated.")

    all_points = pd.concat(evaluated_frames, ignore_index=True)
    elapsed_sec = perf_counter() - total_start
    write_output(all_points, loaded_files, elapsed_sec)

    summary = build_method_summary(all_points)
    print("\nFinished all-point EVM evaluation.")
    print(f"Output workbook: {OUTPUT_FILE}")
    print("\nTop methods by minimum EVM:")
    print(
        summary[
            ["method", "num_points", "min_evm_rms", "mean_evm_rms", "evm_rank_by_min"]
        ]
        .head(13)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
