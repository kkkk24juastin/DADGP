# -*- coding: utf-8 -*-
"""
面向 OFDM 案例的三质量损失 Bayesian optimization 候选点生成脚本。

默认流程:
    python bo_compare.py

在 moo/ 下为每种 BO 方法输出一个 MOO 风格工作簿。
本脚本有意采用“直接修改下方常量”的配置方式，以保持与仓库中其他实验脚本一致。
"""

from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from common import set_seed
from config import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_MOO_POP_SIZE,
    DEFAULT_MOO_RUNS,
    MOO_DIR,
    MOO_LOWER_BOUND,
    MOO_TARGET_VALUES,
    MOO_UPPER_BOUND,
)
from gpytorch.mlls import SumMarginalLogLikelihood

try:
    import matlab.engine
except ImportError as exc:  # pragma: no cover - exercised only without engine.
    matlab = None
    MATLAB_ENGINE_IMPORT_ERROR = exc
else:
    MATLAB_ENGINE_IMPORT_ERROR = None

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 显式运行配置。直接修改这些值。
# ---------------------------------------------------------------------------
SELECTED_METHODS: list[str] | None = None
BO_METHODS = ["bo_qparego", "bo_qehvi", "bo_qnehvi"]
ALL_METHODS = BO_METHODS.copy()
ACQUISITION_OBJECTIVE_ID = "three_quality_loss_v1"

N_RUNS = DEFAULT_MOO_RUNS
BO_BUDGET = DEFAULT_MOO_POP_SIZE
BO_BASE_SEED = 42
MATLAB_SIMULATION_SEED = 42
MODULATION_ORDER = 16

OUTPUT_DIR = MOO_DIR
TRAIN_FILE = DATA_DIR / "train.xlsx"

RESUME_EXISTING = True
OVERWRITE_EXISTING = False

DTYPE = torch.double
MC_SAMPLES = 64
NUM_RESTARTS = 8
RAW_SAMPLES = 128
ACQ_MAXITER = 80
FIT_MAXITER = 75

POWER_MODEL_P_BB = 0.2
POWER_MODEL_P_RF = 0.8
POWER_MODEL_ETA_PA = 0.35
MIN_THROUGHPUT_MBPS = 1e-6

X_COLUMNS = ["x1", "x2", "x3", "x4"]
ENGINEERING_COLUMNS = [
    "throughput_mbps",
    "ber",
    "papr_db",
    "energy_per_bit",
    "energy_efficiency",
]
PRED_COLUMNS = ["pred_y_task1", "pred_y_task2", "pred_y_task3"]
VAR_COLUMNS = ["pred_var_task1", "pred_var_task2", "pred_var_task3"]
STD_COLUMNS = ["pred_std_task1", "pred_std_task2", "pred_std_task3"]
QUALITY_LOSS_COLUMNS = [
    "quality_loss_task1",
    "quality_loss_task2",
    "quality_loss_task3",
]
OBJECTIVE_COLUMNS = [
    "moo_objective_task1",
    "moo_objective_task2",
    "moo_objective_task3",
]
ACQUISITION_SCORE_COLUMNS = [
    "neg_quality_loss_task1",
    "neg_quality_loss_task2",
    "neg_quality_loss_task3",
]
BO_TRACE_COLUMNS = [
    "bo_eval_idx",
    "observed_y_task1",
    "observed_y_task2",
    "observed_y_task3",
    "observed_total_quality_loss",
    "simulation_seed",
    "modulation_order",
    "elapsed_sec",
    "acquisition_name",
    "acquisition_value",
    "ref_point",
    "candidate_source",
]
RESULT_COLUMNS = [
    "method",
    "moo_run",
    "solution_idx",
    *X_COLUMNS,
    *PRED_COLUMNS,
    *VAR_COLUMNS,
    *STD_COLUMNS,
    *QUALITY_LOSS_COLUMNS,
    *OBJECTIVE_COLUMNS,
    *BO_TRACE_COLUMNS,
]
RESULT_SHEET_NAME = "results"
METADATA_SHEET_NAME = "metadata"


@dataclass(frozen=True)
class ObjectiveScaler:
    offset: np.ndarray
    scale: np.ndarray

    def transform_np(self, values: np.ndarray) -> np.ndarray:
        return (values - self.offset.reshape(1, -1)) / self.scale.reshape(1, -1)

    def transform_torch(self, values: torch.Tensor) -> torch.Tensor:
        offset = torch.as_tensor(self.offset, dtype=values.dtype, device=values.device)
        scale = torch.as_tensor(self.scale, dtype=values.dtype, device=values.device)
        return (values - offset) / scale


def resolve_selected_methods() -> list[str]:
    if SELECTED_METHODS is None:
        return ALL_METHODS.copy()
    methods = [str(method).strip() for method in SELECTED_METHODS if str(method).strip()]
    invalid = [method for method in methods if method not in ALL_METHODS]
    if invalid:
        raise ValueError(
            f"Invalid methods: {', '.join(invalid)}. "
            f"Valid methods: {', '.join(ALL_METHODS)}"
        )
    return [method for method in ALL_METHODS if method in set(methods)]


def resolve_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_run_seed(bo_run: int) -> int:
    return int(BO_BASE_SEED) + int(bo_run) - 1


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def derive_energy_metrics(ptx_dbm: np.ndarray, throughput_mbps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pout_w = 10.0 ** ((ptx_dbm - 30.0) / 10.0)
    total_power_w = POWER_MODEL_P_BB + POWER_MODEL_P_RF + pout_w / POWER_MODEL_ETA_PA
    bitrate_bps = np.maximum(throughput_mbps, MIN_THROUGHPUT_MBPS) * 1e6
    energy_per_bit = total_power_w / bitrate_bps
    energy_efficiency = np.where(energy_per_bit > 0, 1.0 / energy_per_bit, np.nan)
    return energy_per_bit, energy_efficiency


def build_engineering_matrix(frame: pd.DataFrame) -> np.ndarray:
    required = [*X_COLUMNS, "y1", "y2", "y3"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{TRAIN_FILE} is missing columns: {', '.join(missing)}")

    throughput = pd.to_numeric(frame["y1"], errors="coerce").to_numpy(dtype=float)
    ber = pd.to_numeric(frame["y2"], errors="coerce").to_numpy(dtype=float)
    papr = pd.to_numeric(frame["y3"], errors="coerce").to_numpy(dtype=float)
    x1 = pd.to_numeric(frame["x1"], errors="coerce").to_numpy(dtype=float)
    energy_per_bit, energy_efficiency = derive_energy_metrics(x1, throughput)
    return np.column_stack([throughput, ber, papr, energy_per_bit, energy_efficiency])


def objective_scores_from_engineering(engineering: np.ndarray) -> np.ndarray:
    """Return BoTorch maximization scores for the three quality-loss objectives."""
    losses, _ = compute_true_quality_losses(engineering)
    return -losses


def compute_true_quality_losses(engineering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute true three-task quality losses from simulator outputs."""
    targets = np.asarray(MOO_TARGET_VALUES, dtype=float).reshape(1, -1)
    y_values = engineering[:, :3]
    losses = (y_values - targets) ** 2
    return losses, losses.sum(axis=1)


def load_initial_observations() -> tuple[np.ndarray, np.ndarray, ObjectiveScaler]:
    frame = pd.read_excel(TRAIN_FILE)
    x_values = frame[X_COLUMNS].to_numpy(dtype=float)
    engineering = build_engineering_matrix(frame)
    scores = objective_scores_from_engineering(engineering)

    score_min = np.nanmin(scores, axis=0)
    score_max = np.nanmax(scores, axis=0)
    score_range = np.where(score_max - score_min < 1e-12, 1.0, score_max - score_min)
    scaler = ObjectiveScaler(offset=score_min, scale=score_range)
    return x_values, scores, scaler


def normalize_x_np(x_values: np.ndarray) -> np.ndarray:
    lower = np.asarray(MOO_LOWER_BOUND, dtype=float).reshape(1, -1)
    upper = np.asarray(MOO_UPPER_BOUND, dtype=float).reshape(1, -1)
    return (x_values - lower) / (upper - lower)


def unnormalize_x_np(x_norm: np.ndarray) -> np.ndarray:
    lower = np.asarray(MOO_LOWER_BOUND, dtype=float).reshape(1, -1)
    upper = np.asarray(MOO_UPPER_BOUND, dtype=float).reshape(1, -1)
    return lower + x_norm * (upper - lower)


def validate_x_bounds(x_values: np.ndarray) -> np.ndarray:
    lower = np.asarray(MOO_LOWER_BOUND, dtype=float).reshape(1, -1)
    upper = np.asarray(MOO_UPPER_BOUND, dtype=float).reshape(1, -1)
    return np.clip(x_values, lower, upper)


class MatlabOfdmEvaluator:
    def __init__(self, project_dir: Path):
        if MATLAB_ENGINE_IMPORT_ERROR is not None:
            raise ImportError(
                "matlab.engine is not importable in this Python environment."
            ) from MATLAB_ENGINE_IMPORT_ERROR

        print("Starting MATLAB Engine...")
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(str(project_dir), nargout=0)

    def evaluate(self, x_values: np.ndarray) -> tuple[np.ndarray, float]:
        x = np.asarray(x_values, dtype=float).reshape(-1)
        if x.shape[0] != 4:
            raise ValueError(f"Expected 4 design variables, got shape {x.shape}.")

        start_time = time.perf_counter()
        values = self.engine.evaluate_ofdm_point(
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(MODULATION_ORDER),
            float(MATLAB_SIMULATION_SEED),
            nargout=5,
        )
        elapsed_sec = time.perf_counter() - start_time
        engineering = np.asarray(values, dtype=float).reshape(1, -1)
        if engineering.shape[1] != len(ENGINEERING_COLUMNS):
            raise RuntimeError("evaluate_ofdm_point returned an unexpected number of outputs.")
        if not np.isfinite(engineering).all():
            raise RuntimeError(f"Non-finite simulator output for x={x.tolist()}: {engineering}")
        return engineering, elapsed_sec

    def close(self) -> None:
        if getattr(self, "engine", None) is not None:
            self.engine.quit()
            self.engine = None


def output_path_for_method(method: str) -> Path:
    return Path(OUTPUT_DIR) / f"{method}.xlsx"


def load_existing_completed_rows(method: str) -> pd.DataFrame:
    path = output_path_for_method(method)
    if OVERWRITE_EXISTING or not RESUME_EXISTING or not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    try:
        metadata = pd.read_excel(path, sheet_name=METADATA_SHEET_NAME)
    except Exception:
        metadata = pd.DataFrame()
    objective_ids = set()
    if "acquisition_objective_id" in metadata.columns:
        objective_ids = {
            str(value).strip()
            for value in metadata["acquisition_objective_id"].dropna().tolist()
        }
    if objective_ids != {ACQUISITION_OBJECTIVE_ID}:
        print(
            f"  {method}: existing workbook uses a different acquisition objective; "
            "starting fresh."
        )
        return pd.DataFrame(columns=RESULT_COLUMNS)

    existing = pd.read_excel(path, sheet_name=RESULT_SHEET_NAME)
    if existing.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    existing["moo_run"] = pd.to_numeric(existing["moo_run"], errors="coerce").astype("Int64")
    run_counts = existing.groupby("moo_run").size()
    complete_runs = set(run_counts[run_counts >= BO_BUDGET].index.astype(int))
    completed = existing[existing["moo_run"].astype(int).isin(complete_runs)].copy()
    return completed.reindex(columns=RESULT_COLUMNS)


def completed_runs(existing: pd.DataFrame) -> set[int]:
    if existing.empty:
        return set()
    run_counts = existing.groupby("moo_run").size()
    return set(run_counts[run_counts >= BO_BUDGET].index.astype(int))


def build_result_row(
    method: str,
    moo_run: int,
    solution_idx: int,
    x_values: np.ndarray,
    engineering: np.ndarray,
    elapsed_sec: float,
    acquisition_name: str,
    acquisition_value: float,
    ref_point: np.ndarray,
    candidate_source: str,
) -> dict:
    engineering = np.asarray(engineering, dtype=float).reshape(1, -1)
    losses, total_loss = compute_true_quality_losses(engineering)
    row = {
        "method": method,
        "moo_run": int(moo_run),
        "solution_idx": int(solution_idx),
        "bo_eval_idx": int(solution_idx),
        "simulation_seed": int(MATLAB_SIMULATION_SEED),
        "modulation_order": int(MODULATION_ORDER),
        "elapsed_sec": float(elapsed_sec),
        "acquisition_name": acquisition_name,
        "acquisition_value": float(acquisition_value) if np.isfinite(acquisition_value) else math.nan,
        "ref_point": str(tuple(float(value) for value in ref_point.reshape(-1))),
        "candidate_source": candidate_source,
    }
    for column, value in zip(X_COLUMNS, np.asarray(x_values, dtype=float).reshape(-1)):
        row[column] = float(value)
    for column, value in zip(PRED_COLUMNS, engineering[:, :3].reshape(-1)):
        row[column] = float(value)
    for column in VAR_COLUMNS:
        row[column] = 0.0
    for column in STD_COLUMNS:
        row[column] = 0.0
    for column, value in zip(QUALITY_LOSS_COLUMNS, losses.reshape(-1)):
        row[column] = float(value)
    for column, value in zip(OBJECTIVE_COLUMNS, losses.reshape(-1)):
        row[column] = float(value)
    for column, value in zip(
        ["observed_y_task1", "observed_y_task2", "observed_y_task3"],
        engineering[:, :3].reshape(-1),
    ):
        row[column] = float(value)
    row["observed_total_quality_loss"] = float(total_loss[0])
    return row


def build_metadata_frame(method: str, num_rows: int, completed_run_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method,
                "acquisition_objective_id": ACQUISITION_OBJECTIVE_ID,
                "output_file": str(output_path_for_method(method)),
                "output_schema": "moo_candidate_workbook",
                "n_runs": int(N_RUNS),
                "bo_budget": int(BO_BUDGET),
                "num_rows": int(num_rows),
                "completed_runs": int(completed_run_count),
                "initial_observation_file": str(TRAIN_FILE),
                "initial_observations": int(pd.read_excel(TRAIN_FILE).shape[0]),
                "moo_target_values": str(tuple(float(value) for value in MOO_TARGET_VALUES)),
                "candidate_selection_objectives": "minimize quality_loss_task1, quality_loss_task2, quality_loss_task3",
                "objective_scores": ", ".join(ACQUISITION_SCORE_COLUMNS),
                "objective_score_definition": "score_i = -((y_i - MOO_TARGET_VALUES[i])^2)",
                "pred_column_note": "BO workbooks store simulator observations in pred_y_task* for MOO-style candidate compatibility.",
                "variance_column_note": "pred_var_task* and pred_std_task* are zero for observed BO feedback points.",
                "simulation_seed": int(MATLAB_SIMULATION_SEED),
                "modulation_order": int(MODULATION_ORDER),
                "mc_samples": int(MC_SAMPLES),
                "num_restarts": int(NUM_RESTARTS),
                "raw_samples": int(RAW_SAMPLES),
                "fit_maxiter": int(FIT_MAXITER),
                "acq_maxiter": int(ACQ_MAXITER),
            }
        ]
    )


def write_method_workbook(method: str, frame: pd.DataFrame) -> None:
    path = output_path_for_method(method)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.reindex(columns=RESULT_COLUMNS)
    frame = frame.sort_values(["moo_run", "solution_idx"], kind="stable").reset_index(drop=True)
    metadata = build_metadata_frame(method, len(frame), len(completed_runs(frame)))
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=RESULT_SHEET_NAME, index=False)
        metadata.to_excel(writer, sheet_name=METADATA_SHEET_NAME, index=False)


def initialize_botorch_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    models = [
        SingleTaskGP(train_x, train_y[:, objective_idx : objective_idx + 1])
        for objective_idx in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models).to(train_x)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": int(FIT_MAXITER)}},
    )
    model.eval()
    return model


def sobol_fallback_candidate(seed: int, device: torch.device) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=int(seed))
    return engine.draw(1).to(device=device, dtype=DTYPE)


def optimize_bo_candidate(
    method: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    ref_point: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, str]:
    model = initialize_botorch_model(train_x, train_y)
    sampler = SobolQMCNormalSampler(torch.Size([MC_SAMPLES]), seed=int(seed))
    bounds = torch.stack(
        [
            torch.zeros(4, device=device, dtype=DTYPE),
            torch.ones(4, device=device, dtype=DTYPE),
        ]
    )

    if method == "bo_qparego":
        weights = sample_simplex(
            train_y.shape[-1],
            n=1,
            qmc=True,
            seed=int(seed),
            device=device,
            dtype=DTYPE,
        ).squeeze(0)
        scalarization = get_chebyshev_scalarization(weights=weights, Y=train_y)
        scalarized_objective = GenericMCObjective(
            lambda samples, X=None: scalarization(samples)
        )
        best_f = scalarization(train_y).max()
        acq_func = qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=scalarized_objective,
        )
    elif method == "bo_qehvi":
        partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
    elif method == "bo_qnehvi":
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
    else:
        raise ValueError(f"Unsupported BO method: {method}")

    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=int(NUM_RESTARTS),
        raw_samples=int(RAW_SAMPLES),
        options={"batch_limit": 5, "maxiter": int(ACQ_MAXITER)},
    )
    candidates = candidates.detach().clamp(0.0, 1.0)
    return candidates, float(acq_value.detach().cpu().reshape(-1)[0]), "acquisition"


def select_candidate(
    method: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    ref_point: torch.Tensor,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, str]:
    try:
        return optimize_bo_candidate(method, train_x, train_y, ref_point, seed, device)
    except Exception as exc:
        print(f"  Warning: {method} acquisition failed ({exc}); using Sobol fallback.")
        candidate = sobol_fallback_candidate(seed + 99173, device)
        return candidate, math.nan, "sobol_fallback"


def run_bo_method(
    method: str,
    evaluator: MatlabOfdmEvaluator,
    initial_x: np.ndarray,
    initial_scores: np.ndarray,
    scaler: ObjectiveScaler,
    device: torch.device,
) -> None:
    existing = load_existing_completed_rows(method)
    done_runs = completed_runs(existing)
    rows = existing.to_dict("records")
    ref_point_np = scaler.transform_np(initial_scores).min(axis=0) - 0.05
    ref_point = torch.as_tensor(ref_point_np, dtype=DTYPE, device=device)

    for moo_run in range(1, N_RUNS + 1):
        if moo_run in done_runs:
            print(f"  {method} run {moo_run}/{N_RUNS}: already complete, skipping.")
            continue

        run_seed = resolve_run_seed(moo_run)
        seed_everything(run_seed)
        train_x = torch.as_tensor(normalize_x_np(initial_x), dtype=DTYPE, device=device)
        train_y = torch.as_tensor(
            scaler.transform_np(initial_scores),
            dtype=DTYPE,
            device=device,
        )

        print(f"\n=== {method} | run {moo_run}/{N_RUNS} | seed={run_seed} ===")
        for solution_idx in range(1, BO_BUDGET + 1):
            candidate_seed = run_seed * 1000 + solution_idx
            candidate_norm, acq_value, candidate_source = select_candidate(
                method=method,
                train_x=train_x,
                train_y=train_y,
                ref_point=ref_point,
                seed=candidate_seed,
                device=device,
            )
            x_values = validate_x_bounds(
                unnormalize_x_np(candidate_norm.detach().cpu().numpy())
            ).reshape(-1)
            engineering, elapsed_sec = evaluator.evaluate(x_values)
            score = objective_scores_from_engineering(engineering)
            score_scaled = scaler.transform_np(score)

            rows.append(
                build_result_row(
                    method=method,
                    moo_run=moo_run,
                    solution_idx=solution_idx,
                    x_values=x_values,
                    engineering=engineering,
                    elapsed_sec=elapsed_sec,
                    acquisition_name=method,
                    acquisition_value=acq_value,
                    ref_point=ref_point_np,
                    candidate_source=candidate_source,
                )
            )

            train_x = torch.cat(
                [
                    train_x,
                    torch.as_tensor(normalize_x_np(x_values.reshape(1, -1)), dtype=DTYPE, device=device),
                ],
                dim=0,
            )
            train_y = torch.cat(
                [
                    train_y,
                    torch.as_tensor(score_scaled, dtype=DTYPE, device=device),
                ],
                dim=0,
            )
            print(
                f"  eval {solution_idx:02d}/{BO_BUDGET}: "
                f"thr={engineering[0, 0]:.4f}, ber={engineering[0, 1]:.6f}, "
                f"papr={engineering[0, 2]:.4f}, ee={engineering[0, 4]:.4e}"
            )

        write_method_workbook(method, pd.DataFrame(rows))

    write_method_workbook(method, pd.DataFrame(rows))
    print(f"{method}: saved {len(rows)} rows to {output_path_for_method(method)}")


def main() -> None:
    selected_methods = resolve_selected_methods()
    device = resolve_device()
    print(f"Selected methods: {', '.join(selected_methods)}")
    print(f"Device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    initial_x, initial_scores, scaler = load_initial_observations()
    evaluator = MatlabOfdmEvaluator(BASE_DIR)
    try:
        for method in selected_methods:
            if method in BO_METHODS:
                run_bo_method(
                    method=method,
                    evaluator=evaluator,
                    initial_x=initial_x,
                    initial_scores=initial_scores,
                    scaler=scaler,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
