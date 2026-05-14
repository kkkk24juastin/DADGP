# -*- coding: utf-8 -*-
"""
基于 pymoo NSGA-II 的多目标优化。

当前版本采用未归一化质量损失目标：
1. 对选定 method 的代理模型分别执行 NSGA-II；
2. 目标定义为：最小化 (预测均值 - 目标值)^2 + 预测方差；
3. 每次优化运行保存整组 Pareto 解集；
4. 每个 method 单独导出一个 workbook，便于只更新单个模型。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from common import set_seed
from config import (
    MODEL_DIR,
    MOO_DIR,
    MOO_LOWER_BOUND,
    MOO_TARGET_VALUES,
    MOO_UPPER_BOUND,
    VALID_METHODS,
    DEFAULT_MOO_RUNS,
    DEFAULT_MOO_POP_SIZE,
    DEFAULT_MOO_N_GEN,
    MOO_EVAL_BATCH_SIZE,
    MOO_CPU_EVAL_BATCH_SIZE,
    MOO_LIKELIHOOD_SAMPLES,
)
from experiment_utils import load_model_checkpoint

# ---------------------------------------------------------------------------
# 显式运行配置
# 直接修改这里即可。
# SELECTED_METHODS = None 表示按 VALID_METHODS 顺序尝试全部方法。
# 也可以写成 ["dadgp"] 或 ["dadgp", "baseline_dwa"]。
# ---------------------------------------------------------------------------
SELECTED_METHODS: list[str] | None = ["dadgp"]
N_RUNS = DEFAULT_MOO_RUNS
POP_SIZE = DEFAULT_MOO_POP_SIZE
N_GEN = DEFAULT_MOO_N_GEN
MOO_BASE_SEED: int | None = 42
OUTPUT_DIR = MOO_DIR

X_COLUMNS = ["x1", "x2", "x3", "x4"]
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
RESULT_SHEET_NAME = "results"
METADATA_SHEET_NAME = "metadata"


def _resolve_selected_methods(selected_methods_config: list[str] | None) -> list[str]:
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

    # 保持 VALID_METHODS 中的既有顺序，避免输出顺序随输入抖动。
    return [method for method in VALID_METHODS if method in set(methods)]


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数，当前收到: {value}")


def _resolve_run_seed(moo_base_seed: int | None, moo_run: int) -> int | None:
    if moo_base_seed is None:
        return None
    return int(moo_base_seed) + moo_run - 1


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """兼容单点解场景，将结果统一为二维数组。"""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _variance_to_std(variances: np.ndarray) -> np.ndarray:
    """将方差稳定地转换为标准差。"""
    return np.sqrt(np.clip(variances, a_min=0.0, a_max=None))


def _resolve_device() -> torch.device:
    """优先使用 GPU，不可用时回退到 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_eval_batch_size(device: torch.device) -> tuple[int, str]:
    """根据设备选择预测批大小，并记录来源。"""
    if device.type == "cpu":
        return MOO_CPU_EVAL_BATCH_SIZE, "config_cpu"
    return MOO_EVAL_BATCH_SIZE, "config_default"


def _load_method_contexts(
    device: torch.device, selected_methods: list[str]
) -> tuple[dict[str, dict], list[str]]:
    """加载选定 method 的模型、规范化参数与元数据。"""
    contexts = {}
    missing_methods = []

    for method in selected_methods:
        ckpt_path = MODEL_DIR / f"{method}.pt"
        if not ckpt_path.exists():
            missing_methods.append(method)
            print(f"  缺失模型: {ckpt_path.name}")
            continue

        checkpoint, model, norm_params = load_model_checkpoint(ckpt_path, device)
        contexts[method] = {
            "checkpoint": checkpoint,
            "model": model,
            "norm_params": norm_params,
        }
        print(f"  已加载: {ckpt_path.name}")

    if not contexts:
        raise FileNotFoundError(f"在 {MODEL_DIR} 下找不到任何可用模型文件。")

    return contexts, missing_methods


def _normalize_x(
    x_np: np.ndarray, norm_params: dict, device: torch.device
) -> torch.Tensor:
    """将原始输入归一化到 [0, 1]（与训练时一致）。"""
    x = torch.from_numpy(x_np).float().to(device)
    x_min = norm_params["x_min"].to(device=x.device, dtype=x.dtype)
    x_range = norm_params["x_range"].to(device=x.device, dtype=x.dtype)
    return (x - x_min) / x_range


def _denormalize_mean(mean_norm: torch.Tensor, norm_params: dict) -> torch.Tensor:
    """将标准化预测均值还原到原始尺度。"""
    y_mean = norm_params["y_mean"].to(device=mean_norm.device, dtype=mean_norm.dtype)
    y_std = norm_params["y_std"].to(device=mean_norm.device, dtype=mean_norm.dtype)
    return mean_norm * y_std + y_mean


def _denormalize_var(var_norm: torch.Tensor, norm_params: dict) -> torch.Tensor:
    """将标准化预测方差还原到原始尺度（方差按 std^2 缩放）。"""
    y_std = norm_params["y_std"].to(device=var_norm.device, dtype=var_norm.dtype)
    return var_norm * (y_std**2)


def _predict_with_model(
    x_np: np.ndarray,
    model,
    norm_params: dict,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单个 method 的代理模型执行推理。

    返回：
        mean_orig  shape (N, 3)
        var_orig   shape (N, 3)
    """
    if x_np.size == 0:
        empty = np.empty((0, len(PRED_COLUMNS)), dtype=np.float64)
        return empty, empty

    x_norm = _normalize_x(x_np, norm_params, device)

    means_batches = []
    vars_batches = []
    for start in range(0, x_norm.shape[0], eval_batch_size):
        batch = x_norm[start : start + eval_batch_size]
        with torch.no_grad():
            mean_batch, var_batch = model.predict(batch)
        means_batches.append(mean_batch.cpu())
        vars_batches.append(var_batch.cpu())

    mean_norm = torch.cat(means_batches, dim=0)
    var_norm = torch.cat(vars_batches, dim=0)

    mean_orig = _denormalize_mean(mean_norm, norm_params)
    var_orig = _denormalize_var(var_norm, norm_params)
    return mean_orig.numpy(), var_orig.numpy()


def _build_quality_loss_objectives(
    means: np.ndarray, variances: np.ndarray, target_values: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """构建未归一化质量损失与 pymoo 最小化目标。"""
    if means.shape[1] != 3 or variances.shape[1] != 3:
        raise ValueError("当前质量损失目标仅支持 3 个任务输出。")

    target_array = np.asarray(target_values, dtype=np.float64).reshape(1, -1)
    if target_array.shape[1] != means.shape[1]:
        raise ValueError(
            "MOO_TARGET_VALUES 的长度必须与任务输出维度一致，"
            f"当前长度: {target_array.shape[1]}, 输出维度: {means.shape[1]}"
        )

    safe_variances = np.clip(variances, a_min=0.0, a_max=None)
    quality_losses = (means - target_array) ** 2 + safe_variances
    return quality_losses, quality_losses.copy()


class QualityLossProblem(Problem):
    """
    三目标质量损失优化问题。

    决策变量：4 维，对应无线网络参数。
    目标函数：
        f_i = (mean_i - target_i)^2 + variance_i
    """

    def __init__(self, model, norm_params, device, eval_batch_size, target_values):
        super().__init__(
            n_var=4,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array(MOO_LOWER_BOUND, dtype=np.float64),
            xu=np.array(MOO_UPPER_BOUND, dtype=np.float64),
        )
        self.model = model
        self.norm_params = norm_params
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.target_values = tuple(float(value) for value in target_values)

    def _evaluate(self, x, out, *args, **kwargs):
        x_np = x.astype(np.float32)
        means, variances = _predict_with_model(
            x_np,
            self.model,
            self.norm_params,
            self.device,
            self.eval_batch_size,
        )
        _, objectives = _build_quality_loss_objectives(
            means,
            variances,
            self.target_values,
        )
        out["F"] = objectives


def _build_method_result_frame(
    method: str,
    moo_run: int,
    x_values: np.ndarray,
    pred_means: np.ndarray,
    pred_vars: np.ndarray,
    pred_stds: np.ndarray,
    quality_losses: np.ndarray,
    objective_values: np.ndarray,
) -> pd.DataFrame:
    """整理单个 method 单次运行返回的 Pareto 解集。"""
    frame = pd.DataFrame(
        np.hstack(
            [
                x_values,
                pred_means,
                pred_vars,
                pred_stds,
                quality_losses,
                objective_values,
            ]
        ),
        columns=X_COLUMNS
        + PRED_COLUMNS
        + VAR_COLUMNS
        + STD_COLUMNS
        + QUALITY_LOSS_COLUMNS
        + OBJECTIVE_COLUMNS,
    )

    frame.insert(0, "method", method)

    frame = frame.sort_values(
        by=OBJECTIVE_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)
    frame.insert(1, "moo_run", moo_run)
    frame.insert(2, "solution_idx", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame["moo_run"] = frame["moo_run"].astype(np.int64)
    return frame


def run_single_method_moo(
    method: str,
    method_context: dict,
    moo_run: int,
    total_runs: int,
    device: torch.device,
    eval_batch_size: int,
    pop_size: int,
    n_gen: int,
    run_seed: int | None,
) -> pd.DataFrame:
    """执行单个 method 的一次 NSGA-II，并返回该次运行的 Pareto 解集。"""
    seed_text = "None" if run_seed is None else str(run_seed)
    print(
        f"\n=== Method={method} | run={moo_run}/{total_runs} | "
        f"seed={seed_text} | device={device} ==="
    )
    if run_seed is not None:
        set_seed(run_seed)

    problem = QualityLossProblem(
        model=method_context["model"],
        norm_params=method_context["norm_params"],
        device=device,
        eval_batch_size=eval_batch_size,
        target_values=MOO_TARGET_VALUES,
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    termination = get_termination("n_gen", n_gen)

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=run_seed,
        verbose=True,
    )

    if result.X is None or result.F is None:
        raise RuntimeError(f"{method} 在第 {moo_run} 次运行中没有返回有效解。")

    x_values = _ensure_2d(result.X)
    pred_means, pred_vars = _predict_with_model(
        x_values.astype(np.float32),
        method_context["model"],
        method_context["norm_params"],
        device,
        eval_batch_size,
    )

    if x_values.shape[0] != pred_means.shape[0]:
        raise RuntimeError(f"{method} 的决策变量数量与预测结果数量不一致。")

    pred_stds = _variance_to_std(pred_vars)
    quality_losses, objective_values = _build_quality_loss_objectives(
        pred_means,
        pred_vars,
        MOO_TARGET_VALUES,
    )
    frame = _build_method_result_frame(
        method=method,
        moo_run=moo_run,
        x_values=x_values,
        pred_means=pred_means,
        pred_vars=pred_vars,
        pred_stds=pred_stds,
        quality_losses=quality_losses,
        objective_values=objective_values,
    )
    print(f"  Pareto 解数量: {len(frame)} | 已保存该运行的完整 Pareto 解集")
    return frame


def run_moo(
    selected_methods: list[str],
    n_runs: int,
    pop_size: int,
    n_gen: int,
    moo_base_seed: int | None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict], list[str], dict]:
    """执行选定 method 的 NSGA-II，返回按 method 聚合后的结果。"""
    device = _resolve_device()
    eval_batch_size, eval_batch_size_source = _resolve_eval_batch_size(device)

    print("=== 加载代理模型 ===")
    method_contexts, missing_methods = _load_method_contexts(device, selected_methods)

    results_by_method = {}
    for method in selected_methods:
        if method not in method_contexts:
            continue

        method_runs = []
        for moo_run in range(1, n_runs + 1):
            run_seed = _resolve_run_seed(moo_base_seed, moo_run)
            run_frame = run_single_method_moo(
                method=method,
                method_context=method_contexts[method],
                moo_run=moo_run,
                total_runs=n_runs,
                device=device,
                eval_batch_size=eval_batch_size,
                pop_size=pop_size,
                n_gen=n_gen,
                run_seed=run_seed,
            )
            method_runs.append(run_frame)

        results_by_method[method] = pd.concat(method_runs, ignore_index=True)

    summary_meta = {
        "selected_methods": selected_methods,
        "completed_methods": len(results_by_method),
        "missing_methods": len(missing_methods),
        "device": str(device),
        "eval_batch_size": int(eval_batch_size),
        "eval_batch_size_source": eval_batch_size_source,
        "pop_size": int(pop_size),
        "n_gen": int(n_gen),
        "n_runs": int(n_runs),
        "objective_type": "unnormalized_quality_loss",
        "moo_target_values": tuple(float(value) for value in MOO_TARGET_VALUES),
        "likelihood_samples": int(MOO_LIKELIHOOD_SAMPLES),
    }
    return results_by_method, method_contexts, missing_methods, summary_meta


def _build_metadata_frame(
    method: str,
    method_frame: pd.DataFrame,
    method_context: dict | None,
    summary_meta: dict,
    output_file: Path,
) -> pd.DataFrame:
    checkpoint = method_context.get("checkpoint") if method_context is not None else {}
    return pd.DataFrame(
        [
            {
                "method": method,
                "output_file": str(output_file),
                "model_dataset_id": checkpoint.get("dataset_id", ""),
                "objective_type": summary_meta["objective_type"],
                "moo_target_values": str(summary_meta["moo_target_values"]),
                "likelihood_samples": summary_meta["likelihood_samples"],
                "device": summary_meta["device"],
                "eval_batch_size": summary_meta["eval_batch_size"],
                "eval_batch_size_source": summary_meta["eval_batch_size_source"],
                "pop_size": summary_meta["pop_size"],
                "n_gen": summary_meta["n_gen"],
                "n_runs": summary_meta["n_runs"],
                "num_rows": int(len(method_frame)),
            }
        ]
    )


def _write_method_workbook(
    method: str,
    method_frame: pd.DataFrame,
    method_context: dict | None,
    summary_meta: dict,
    output_file: Path,
) -> None:
    """将单个 method 的结果保存为独立 workbook。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_frame = _build_metadata_frame(
        method=method,
        method_frame=method_frame,
        method_context=method_context,
        summary_meta=summary_meta,
        output_file=output_file,
    )
    with pd.ExcelWriter(output_file) as writer:
        method_frame.to_excel(writer, sheet_name=RESULT_SHEET_NAME, index=False)
        metadata_frame.to_excel(writer, sheet_name=METADATA_SHEET_NAME, index=False)


def save_moo_results(
    results_by_method: dict[str, pd.DataFrame],
    method_contexts: dict[str, dict],
    summary_meta: dict,
    output_dir: Path,
) -> dict[str, Path]:
    """将每个 method 的 Pareto 解集分别保存为独立 workbook。"""
    output_paths = {}
    for method, method_frame in results_by_method.items():
        output_file = output_dir / f"{method}.xlsx"
        _write_method_workbook(
            method=method,
            method_frame=method_frame,
            method_context=method_contexts.get(method),
            summary_meta=summary_meta,
            output_file=output_file,
        )
        output_paths[method] = output_file
    return output_paths


def main() -> None:
    selected_methods = _resolve_selected_methods(SELECTED_METHODS)
    _validate_positive_int("N_RUNS", N_RUNS)
    _validate_positive_int("POP_SIZE", POP_SIZE)
    _validate_positive_int("N_GEN", N_GEN)
    output_dir = Path(OUTPUT_DIR)

    results_by_method, method_contexts, missing_methods, summary_meta = run_moo(
        selected_methods=selected_methods,
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        n_gen=N_GEN,
        moo_base_seed=MOO_BASE_SEED,
    )

    output_paths = save_moo_results(
        results_by_method=results_by_method,
        method_contexts=method_contexts,
        summary_meta=summary_meta,
        output_dir=output_dir,
    )

    print("\n=== 优化完成 ===")
    for method in selected_methods:
        method_df = results_by_method.get(method)
        if method_df is None:
            continue
        print(f"{method}: {len(method_df)} 条 Pareto 解已保存到 {output_paths[method]}")

    if missing_methods:
        print(f"缺失模型: {', '.join(missing_methods)}")


if __name__ == "__main__":
    main()
