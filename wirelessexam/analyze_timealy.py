# -*- coding: utf-8 -*-
"""
时间复杂度与训练耗时分析脚本。

默认对 config.py 中的全部 VALID_METHODS 按 NUM_EPOCHS 完整训练计时，
并将分析结果输出到 timealy/ 目录。
"""

import argparse
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from algo_ablation_no_sample_attn import run_dadgp_no_sample_weights
from algo_dadgp import DADGP, run_training_loop
from algo_dwa import run_dwa_baseline_training_loop
from algo_equal import run_baseline_training_loop
from algo_mgda import run_mgda_training_loop
from algo_supplemental_baselines import (
    run_independent_dgp_training_loop,
    run_independent_hetgp_training_loop,
)
from algo_uw import run_uncertainty_weighting_training_loop
from common import create_loss_function, set_seed
from config import (
    BASE_DIR,
    BATCH_SIZE,
    INPUT_DIMENSIONS,
    LMC_NUM_LATENTS,
    NUM_EPOCHS,
    NUM_HIDDEN_DGP_DIMS,
    NUM_INDUCING_POINTS,
    NUM_TASKS,
    PRI_TASKS,
    SAMPLE_ATTENTION_METHODS,
    SIGMA_VALUES,
    TARGET_VALUES,
    TASK_TO_INDEX,
    TRAIN_TASKS,
    VALID_METHODS,
)
from data_loading import setup_experiment_data, setup_sample_weights
from experiment_utils import build_model
from models_dgp import IndependentDeepGP, IndependentHeteroscedasticGP, LMCDeepGP


TIMEALY_DIR = BASE_DIR / "timealy"


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析各训练方法的时间复杂度与实测训练耗时。"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=VALID_METHODS,
        default=None,
        help="指定要分析的方法；默认使用 config.VALID_METHODS 的全部方法。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"训练 epoch 数；默认使用 config.NUM_EPOCHS={NUM_EPOCHS}。",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="运行设备；默认 auto，CUDA 可用时使用 cuda。",
    )
    parser.add_argument("--seed", type=int, default=None, help="可选随机种子。")
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="关闭数据归一化，与 main_experiment.py 保持一致。",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="为每个方法额外导出 torch.profiler operator 汇总表。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TIMEALY_DIR,
        help="输出目录；默认 timealy/。",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 --device cuda，但当前环境 torch.cuda.is_available() 为 False。")
    return torch.device(device_arg)


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine() or "cpu"


def build_sample_attention_config(methods, train_data, val_data, normalization_params):
    sample_attention_methods = set(SAMPLE_ATTENTION_METHODS)
    if not any(method in sample_attention_methods for method in methods):
        print("当前分析方法不使用按任务高斯样本注意力。")
        return None

    sample_weights = setup_sample_weights(
        train_data,
        val_data,
        TARGET_VALUES,
        SIGMA_VALUES,
        normalization_params,
    )
    enabled_methods = [method for method in methods if method in sample_attention_methods]
    print("按任务高斯样本注意力将用于: " + ", ".join(enabled_methods))
    return sample_weights


def resolve_method_sample_weights(method_name, sample_weights):
    if method_name in set(SAMPLE_ATTENTION_METHODS):
        return sample_weights
    return None


def build_method_runner(method_name, train_data, datasets, split_sizes, device, sample_weights):
    num_tasks = train_data["y"].size(-1)
    task_ids = list(TRAIN_TASKS.keys())
    method_sample_weights = resolve_method_sample_weights(method_name, sample_weights)

    if method_name == "dadgp":
        model = build_model(train_data, num_tasks, device)
        dadgp = DADGP(
            model,
            device,
            TRAIN_TASKS,
            PRI_TASKS,
            sample_weights=method_sample_weights,
        )
        dadgp.loss_fn = create_loss_function(
            sample_weights=method_sample_weights,
            split_sizes=split_sizes,
            task_to_index=TASK_TO_INDEX,
        )
        return model, lambda epochs: run_training_loop(model, dadgp, datasets, epochs)

    if method_name == "ablation_no_sample_attn":
        model = build_model(train_data, num_tasks, device)
        dadgp = DADGP(
            model,
            device,
            TRAIN_TASKS,
            PRI_TASKS,
            sample_weights=None,
        )
        dadgp.loss_fn = create_loss_function(
            sample_weights=None,
            split_sizes=split_sizes,
            task_to_index=TASK_TO_INDEX,
        )
        return model, lambda epochs: run_dadgp_no_sample_weights(
            model,
            dadgp,
            datasets,
            epochs,
        )

    if method_name == "baseline_equal":
        model = build_model(train_data, num_tasks, device)
        return model, lambda epochs: run_baseline_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            method_sample_weights,
        )

    if method_name == "baseline_pure_dgp":
        model = build_model(train_data, num_tasks, device)
        return model, lambda epochs: run_baseline_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            None,
        )

    if method_name == "baseline_dwa":
        model = build_model(train_data, num_tasks, device)
        return model, lambda epochs: run_dwa_baseline_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            method_sample_weights,
        )

    if method_name == "baseline_uw":
        model = build_model(train_data, num_tasks, device)
        return model, lambda epochs: run_uncertainty_weighting_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            method_sample_weights,
        )

    if method_name == "baseline_mgda":
        model = build_model(train_data, num_tasks, device)
        return model, lambda epochs: run_mgda_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            method_sample_weights,
        )

    if method_name == "baseline_indep_dgp":
        model = IndependentDeepGP(
            train_data["x"].shape,
            num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
            num_tasks=num_tasks,
        ).to(device)
        return model, lambda epochs: run_independent_dgp_training_loop(
            model,
            datasets,
            epochs,
        )

    if method_name == "baseline_indep_hetgp":
        model = IndependentHeteroscedasticGP(train_data["x"], train_data["y"]).to(device)
        return model, lambda epochs: run_independent_hetgp_training_loop(
            model,
            datasets,
            epochs,
        )

    if method_name == "baseline_lmc_dgp":
        model = LMCDeepGP(
            train_data["x"].shape,
            num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
            num_tasks=num_tasks,
        ).to(device)
        return model, lambda epochs: run_baseline_training_loop(
            model,
            datasets,
            epochs,
            task_ids,
            None,
        )

    raise ValueError(f"不支持的方法: {method_name}")


def profile_sort_key(device):
    if device.type == "cuda":
        return "self_cuda_time_total"
    return "self_cpu_time_total"


def run_with_optional_profiler(method_name, runner, epochs, device, output_dir, enable_profile):
    if not enable_profile:
        return runner(epochs)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities) as prof:
        history = runner(epochs)
        synchronize_if_needed(device)

    table = prof.key_averages().table(
        sort_by=profile_sort_key(device),
        row_limit=-1,
    )
    (output_dir / f"profiler_{method_name}.txt").write_text(table, encoding="utf-8")
    return history


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def final_loss_from_history(history):
    losses = history.get("losses", []) if history else []
    if not losses:
        return None
    return float(losses[-1])


def run_timed_method(
    method_name,
    datasets,
    split_sizes,
    sample_weights,
    epochs,
    device,
    seed,
    output_dir,
    enable_profile,
):
    if seed is not None:
        set_seed(seed)
        print(f"Reset seed to {seed} for {method_name}")

    train_data = datasets["train"]
    synchronize_if_needed(device)
    start_perf = time.perf_counter()
    start_time = datetime.now()
    status = "ok"
    error = ""
    history = None
    total_params = 0
    trainable_params = 0

    try:
        model, runner = build_method_runner(
            method_name,
            train_data,
            datasets,
            split_sizes,
            device,
            sample_weights,
        )
        total_params, trainable_params = count_parameters(model)
        history = run_with_optional_profiler(
            method_name,
            runner,
            epochs,
            device,
            output_dir,
            enable_profile,
        )
        synchronize_if_needed(device)
    except Exception as exc:
        synchronize_if_needed(device)
        status = "failed"
        error = repr(exc)
        print(f"{method_name} 运行失败: {error}")

    end_time = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf
    num_train = int(train_data["y"].size(0))
    batches_per_epoch = math.ceil(num_train / BATCH_SIZE) if num_train else 0
    total_batches = int(batches_per_epoch * epochs)

    return {
        "method": method_name,
        "status": status,
        "error": error,
        "device": str(device),
        "device_name": device_name(device),
        "epochs": int(epochs),
        "train_samples": num_train,
        "val_samples": int(datasets["val"]["y"].size(0)),
        "batch_size": int(BATCH_SIZE),
        "batches_per_epoch": int(batches_per_epoch),
        "total_batches": total_batches,
        "elapsed_seconds": float(elapsed_seconds),
        "seconds_per_epoch": float(elapsed_seconds / epochs) if epochs > 0 else None,
        "seconds_per_batch": (
            float(elapsed_seconds / total_batches) if total_batches > 0 else None
        ),
        "final_loss": final_loss_from_history(history),
        "history_epochs": len(history.get("losses", [])) if history else 0,
        "num_parameters": int(total_params),
        "num_trainable_parameters": int(trainable_params),
        "started_at": start_time.isoformat(timespec="seconds"),
        "ended_at": end_time.isoformat(timespec="seconds"),
    }


def dgp_complexity_formula(prefix_multiplier="1"):
    return (
        f"O({prefix_multiplier} * E * ceil(N/B) * "
        "(H + T) * (B*M^2 + M^3))"
    )


def build_complexity_rows(methods, train_samples, val_samples, epochs):
    rows = []
    base = {
        "input_dimensions": int(INPUT_DIMENSIONS),
        "num_tasks": int(NUM_TASKS),
        "num_hidden_dgp_dims": int(NUM_HIDDEN_DGP_DIMS),
        "num_inducing_points": int(NUM_INDUCING_POINTS),
        "batch_size": int(BATCH_SIZE),
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "epochs": int(epochs),
        "batches_per_epoch": int(math.ceil(train_samples / BATCH_SIZE))
        if train_samples
        else 0,
        "lmc_num_latents": int(LMC_NUM_LATENTS),
    }

    descriptions = {
        "dadgp": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("~4"),
            "extra_cost_note": (
                "每 batch 包含常规模型更新、虚拟 Adam step、验证集 unrolled backward、"
                "两次 finite-difference Hessian-vector 近似，实际常数显著高于普通 DGP。"
            ),
        },
        "ablation_no_sample_attn": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("~4"),
            "extra_cost_note": (
                "与 DADGP 类似保留 meta/unrolled/Hessian 近似，但不计算样本注意力权重。"
            ),
        },
        "baseline_equal": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("1"),
            "extra_cost_note": "固定等权任务损失；每 batch 一次 forward/backward。",
        },
        "baseline_pure_dgp": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("1"),
            "extra_cost_note": "不使用样本注意力；每 batch 一次 forward/backward。",
        },
        "baseline_dwa": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("1"),
            "extra_cost_note": "DWA 权重更新为轻量张量运算，主成本仍为 DGP 训练。",
        },
        "baseline_uw": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("1"),
            "extra_cost_note": "额外学习 log_vars，参数量和耗时增量很小。",
        },
        "baseline_mgda": {
            "model_family": "shared two-layer multitask variational DGP",
            "training_cost": dgp_complexity_formula("T"),
            "extra_cost_note": (
                "每 batch 对各任务分别求梯度并解小规模 Frank-Wolfe 子问题，"
                "梯度计算常数约随任务数 T 增长。"
            ),
        },
        "baseline_indep_dgp": {
            "model_family": "T independent two-layer single-task variational DGPs",
            "training_cost": dgp_complexity_formula("T"),
            "extra_cost_note": "每个任务独立 DGP，不共享参数；主成本随任务数近似线性增长。",
        },
        "baseline_indep_hetgp": {
            "model_family": "T independent exact heteroscedastic GPs",
            "training_cost": "O(E * T * N^3)",
            "extra_cost_note": (
                "Exact GP 每 epoch 使用全训练集；初始化/预测噪声估计还包含 kNN/cdist 的 O(N^2 * D) 成本。"
            ),
        },
        "baseline_lmc_dgp": {
            "model_family": "shared DGP hidden layer with LMC variational output",
            "training_cost": (
                "O(E * ceil(N/B) * (H*(B*M^2 + M^3) + L*(B*M^2 + M^3)))"
            ),
            "extra_cost_note": "输出层 latent 数 L=LMC_NUM_LATENTS，通常与任务数同阶。",
        },
    }

    for method in methods:
        info = descriptions[method]
        rows.append(
            {
                "method": method,
                **base,
                "model_family": info["model_family"],
                "training_time_complexity": info["training_cost"],
                "symbol_legend": (
                    "E=epochs, N=train_samples, B=batch_size, M=num_inducing_points, "
                    "D=input_dimensions, H=num_hidden_dgp_dims, T=num_tasks, L=lmc_num_latents"
                ),
                "extra_cost_note": info["extra_cost_note"],
            }
        )
    return rows


def write_summary(
    output_path,
    methods,
    timing_rows,
    start_time,
    end_time,
    device,
    normalize,
    seed,
    profile_enabled,
):
    def fmt_seconds(value):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.4f}s"

    ok_rows = [row for row in timing_rows if row["status"] == "ok"]
    failed_rows = [row for row in timing_rows if row["status"] != "ok"]
    sorted_rows = sorted(ok_rows, key=lambda row: row["elapsed_seconds"], reverse=True)

    lines = [
        "时间复杂度与训练耗时分析摘要",
        "=" * 40,
        f"开始时间: {start_time.isoformat(timespec='seconds')}",
        f"结束时间: {end_time.isoformat(timespec='seconds')}",
        f"总耗时秒: {(end_time - start_time).total_seconds():.4f}",
        f"方法: {', '.join(methods)}",
        f"设备: {device} ({device_name(device)})",
        f"PyTorch: {torch.__version__}",
        f"Python: {platform.python_version()}",
        f"系统: {platform.platform()}",
        f"归一化: {normalize}",
        f"随机种子: {seed}",
        f"Profiler: {profile_enabled}",
        "",
        "配置",
        "-" * 40,
        f"NUM_EPOCHS: {timing_rows[0]['epochs'] if timing_rows else NUM_EPOCHS}",
        f"BATCH_SIZE: {BATCH_SIZE}",
        f"INPUT_DIMENSIONS: {INPUT_DIMENSIONS}",
        f"NUM_TASKS: {NUM_TASKS}",
        f"NUM_HIDDEN_DGP_DIMS: {NUM_HIDDEN_DGP_DIMS}",
        f"NUM_INDUCING_POINTS: {NUM_INDUCING_POINTS}",
        f"LMC_NUM_LATENTS: {LMC_NUM_LATENTS}",
        "",
        "耗时排序（慢到快）",
        "-" * 40,
    ]
    if sorted_rows:
        for row in sorted_rows:
            lines.append(
                f"{row['method']}: {fmt_seconds(row['elapsed_seconds'])} "
                f"({fmt_seconds(row['seconds_per_epoch'])}/epoch, "
                f"{fmt_seconds(row['seconds_per_batch'])}/batch)"
            )
    else:
        lines.append("没有成功完成的方法。")

    if sorted_rows:
        fastest = sorted_rows[-1]
        slowest = sorted_rows[0]
        lines.extend(
            [
                "",
                "最快/最慢",
                "-" * 40,
                f"最快: {fastest['method']} ({fastest['elapsed_seconds']:.4f}s)",
                f"最慢: {slowest['method']} ({slowest['elapsed_seconds']:.4f}s)",
            ]
        )

    if failed_rows:
        lines.extend(["", "失败方法", "-" * 40])
        for row in failed_rows:
            lines.append(f"{row['method']}: {row['error']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_methods(methods):
    invalid = [method for method in methods if method not in VALID_METHODS]
    if invalid:
        raise ValueError("无效方法: " + ", ".join(invalid))


def main():
    args = parse_args()
    methods = args.methods or list(VALID_METHODS)
    validate_methods(methods)
    if args.epochs < 0:
        raise ValueError("--epochs 必须大于等于 0。")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    normalize = not args.disable_normalize
    analysis_start = datetime.now()

    if args.seed is not None:
        set_seed(args.seed)

    print(f"使用设备: {device} ({device_name(device)})")
    print(f"输出目录: {output_dir}")
    print(f"分析方法: {', '.join(methods)}")
    print(f"训练 epoch: {args.epochs}")

    datasets, normalization_params = setup_experiment_data(device, normalize=normalize)
    train_data, val_data = datasets["train"], datasets["val"]
    split_sizes = {
        "train": train_data["y"].size(0),
        "val": val_data["y"].size(0),
    }
    sample_weights = build_sample_attention_config(
        methods,
        train_data,
        val_data,
        normalization_params,
    )

    complexity_rows = build_complexity_rows(
        methods,
        train_samples=int(train_data["y"].size(0)),
        val_samples=int(val_data["y"].size(0)),
        epochs=args.epochs,
    )
    pd.DataFrame(complexity_rows).to_csv(
        output_dir / "time_complexity.csv",
        index=False,
    )

    timing_rows = []
    for method_name in methods:
        print(f"\n=== Time Analysis: {method_name} ===")
        row = run_timed_method(
            method_name,
            datasets,
            split_sizes,
            sample_weights,
            args.epochs,
            device,
            args.seed,
            output_dir,
            args.profile,
        )
        timing_rows.append(row)
        print(
            f"{method_name} 状态: {row['status']}, "
            f"耗时: {row['elapsed_seconds']:.4f}s"
        )

    pd.DataFrame(timing_rows).to_csv(output_dir / "training_time.csv", index=False)

    analysis_end = datetime.now()
    write_summary(
        output_dir / "summary.txt",
        methods,
        timing_rows,
        analysis_start,
        analysis_end,
        device,
        normalize,
        args.seed,
        args.profile,
    )

    print(f"\n已保存时间复杂度分析: {output_dir / 'time_complexity.csv'}")
    print(f"已保存训练耗时分析: {output_dir / 'training_time.csv'}")
    print(f"已保存摘要: {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
