# -*- coding: utf-8 -*-
"""
wireless 主实验入口。

参考上一级 3tasks 的结构，将原先的大型单文件重构为：
- config/common/models/metrics/data/algo/moo 等公共模块
- 当前文件统一负责常规模型与消融方法的训练编排、模型保存和训练损失曲线绘图
- 多目标优化由 moo_optimization.py 手动执行，不在训练流程中自动触发

同时保留原脚本常用的对外符号，避免已有调用方式直接失效。
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
_MPL_CONFIG_DIR = Path("/tmp") / "wireless-matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
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
    MODEL_DIR,
    NUM_EPOCHS,
    NUM_HIDDEN_DGP_DIMS,
    PRI_TASKS,
    SAMPLE_ATTENTION_METHODS,
    SIGMA_VALUES,
    TASK_TO_INDEX,
    TARGET_VALUES,
    TRAIN_TASKS,
    VALID_METHODS,
)
from data_loading import (
    setup_experiment_data,
    setup_sample_weights,
)
from experiment_utils import build_model, save_model_checkpoint
from models_dgp import IndependentDeepGP, IndependentHeteroscedasticGP, LMCDeepGP

SINGLE_RUN_LABEL = "single_run"


def run_single_experiment(
    device,
    num_epochs,
    methods=None,
    normalize=True,
    seed=None,
):
    """在单次实验数据上训练并保存多个模型。"""
    if seed is not None:
        set_seed(seed)

    if methods is None:
        methods = VALID_METHODS.copy()

    sample_attention_methods = set(SAMPLE_ATTENTION_METHODS)

    datasets, normalization_params = setup_experiment_data(
        device,
        normalize=normalize,
    )
    train_data, val_data = datasets["train"], datasets["val"]
    num_tasks = train_data["y"].size(-1)
    split_sizes = {
        "train": train_data["y"].size(0),
        "val": val_data["y"].size(0),
    }

    sample_weights = None
    if any(method in sample_attention_methods for method in methods):
        sample_weights = setup_sample_weights(
            train_data,
            val_data,
            TARGET_VALUES,
            SIGMA_VALUES,
            normalization_params,
        )
        print(
            "按任务高斯样本注意力将用于: "
            + ", ".join(method for method in methods if method in sample_attention_methods)
        )
    else:
        print("当前轮次没有方法使用按任务高斯样本注意力。")

    results_by_method = {}
    histories_by_method = {}
    model_output_dir = MODEL_DIR
    model_output_dir.mkdir(parents=True, exist_ok=True)

    def reset_method_seed(method_name):
        """确保同一轮实验内的每个方法都从相同随机起点开始。"""
        if seed is not None:
            set_seed(seed)
            print(f"Reset seed to {seed} for {method_name}")

    def resolve_method_sample_weights(method_name):
        if method_name in sample_attention_methods:
            return sample_weights
        return None

    def build_sample_attention_config(method_name):
        return {
            "enabled": method_name in sample_attention_methods,
            "methods": list(SAMPLE_ATTENTION_METHODS),
            "mode": "per_task_gaussian",
            "apply_to_val": True,
            "target_values": tuple(TARGET_VALUES),
            "sigma_values": tuple(SIGMA_VALUES),
        }

    def finalize_method(method_name, model, extra_metrics=None, history=None):
        model_path = model_output_dir / f"{method_name}.pt"
        save_model_checkpoint(
            model,
            model_path,
            train_data,
            normalization_params,
            method_name,
            SINGLE_RUN_LABEL,
            sample_attention_config=build_sample_attention_config(method_name),
        )

        method_record = {
            "model_path": str(model_path.relative_to(MODEL_DIR.parent)),
        }
        if extra_metrics is not None:
            method_record.update(extra_metrics)
        results_by_method[method_name] = method_record

        if history is not None and history.get("losses"):
            histories_by_method[method_name] = history

    if "dadgp" in methods:
        print("\n=== Running DADGP ===")
        reset_method_seed("dadgp")
        model = build_model(train_data, num_tasks, device)
        dadgp = DADGP(
            model,
            device,
            TRAIN_TASKS,
            PRI_TASKS,
            sample_weights=resolve_method_sample_weights("dadgp"),
        )
        dadgp.loss_fn = create_loss_function(
            sample_weights=resolve_method_sample_weights("dadgp"),
            split_sizes=split_sizes,
            task_to_index=TASK_TO_INDEX,
        )
        history = run_training_loop(model, dadgp, datasets, num_epochs)

        final_weights = dadgp.get_normalized_weights().detach().cpu().numpy()
        finalize_method(
            "dadgp",
            model,
            extra_metrics={
                f"weight_{task_id}": final_weights[idx]
                for idx, task_id in enumerate(TRAIN_TASKS)
            },
            history=history,
        )

    if "ablation_no_sample_attn" in methods:
        print("\n=== Running Ablation: -Sample Attn ===")
        reset_method_seed("ablation_no_sample_attn")
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
        history = run_dadgp_no_sample_weights(model, dadgp, datasets, num_epochs)

        final_weights = dadgp.get_normalized_weights().detach().cpu().numpy()
        finalize_method(
            "ablation_no_sample_attn",
            model,
            extra_metrics={
                f"weight_{task_id}": final_weights[idx]
                for idx, task_id in enumerate(TRAIN_TASKS)
            },
            history=history,
        )

    if "baseline_equal" in methods:
        print("\n=== Running Baseline Equal ===")
        reset_method_seed("baseline_equal")
        model = build_model(train_data, num_tasks, device)
        history = run_baseline_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            resolve_method_sample_weights("baseline_equal"),
        )
        finalize_method("baseline_equal", model, history=history)

    if "baseline_pure_dgp" in methods:
        print("\n=== Running Baseline Pure DGP ===")
        reset_method_seed("baseline_pure_dgp")
        model = build_model(train_data, num_tasks, device)
        history = run_baseline_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            None,
        )
        finalize_method("baseline_pure_dgp", model, history=history)

    if "baseline_dwa" in methods:
        print("\n=== Running Baseline DWA ===")
        reset_method_seed("baseline_dwa")
        model = build_model(train_data, num_tasks, device)
        history = run_dwa_baseline_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            resolve_method_sample_weights("baseline_dwa"),
        )
        finalize_method("baseline_dwa", model, history=history)

    if "baseline_uw" in methods:
        print("\n=== Running Baseline Uncertainty Weighting ===")
        reset_method_seed("baseline_uw")
        model = build_model(train_data, num_tasks, device)
        history = run_uncertainty_weighting_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            resolve_method_sample_weights("baseline_uw"),
        )
        finalize_method("baseline_uw", model, history=history)

    if "baseline_mgda" in methods:
        print("\n=== Running Baseline MGDA ===")
        reset_method_seed("baseline_mgda")
        model = build_model(train_data, num_tasks, device)
        history = run_mgda_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            resolve_method_sample_weights("baseline_mgda"),
        )
        finalize_method("baseline_mgda", model, history=history)

    if "baseline_indep_dgp" in methods:
        print("\n=== Running Baseline Indep-DGP ===")
        reset_method_seed("baseline_indep_dgp")
        model = IndependentDeepGP(
            train_data["x"].shape,
            num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
            num_tasks=num_tasks,
        ).to(device)
        history = run_independent_dgp_training_loop(model, datasets, num_epochs)
        finalize_method("baseline_indep_dgp", model, history=history)

    if "baseline_indep_hetgp" in methods:
        print("\n=== Running Baseline Indep-HetGP ===")
        reset_method_seed("baseline_indep_hetgp")
        model = IndependentHeteroscedasticGP(train_data["x"], train_data["y"]).to(device)
        history = run_independent_hetgp_training_loop(model, datasets, num_epochs)
        finalize_method("baseline_indep_hetgp", model, history=history)

    if "baseline_lmc_dgp" in methods:
        print("\n=== Running Baseline LMC-DGP ===")
        reset_method_seed("baseline_lmc_dgp")
        model = LMCDeepGP(
            train_data["x"].shape,
            num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS,
            num_tasks=num_tasks,
        ).to(device)
        history = run_baseline_training_loop(
            model,
            datasets,
            num_epochs,
            list(TRAIN_TASKS.keys()),
            None,
        )
        finalize_method("baseline_lmc_dgp", model, history=history)

    return results_by_method, histories_by_method


def plot_results(history, output_path):
    """绘制损失曲线。"""
    plt.figure(figsize=(8, 6))
    plt.plot(history["losses"], label="Total Weighted Loss")
    plt.title("Total Weighted Loss Trajectory")
    plt.xlabel("Epoch")
    plt.ylabel("Average Total Weighted Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def export_training_trace(history, method_name, step_output_path, epoch_output_path):
    """导出任务权重与 meta 梯度轨迹，便于复现实验分析。"""
    task_ids = list(TRAIN_TASKS.keys())
    weights = history.get("weights", [])
    if weights:
        step_rows = []
        step_meta_grad = history.get("step_meta_grad", [])
        step_val_meta_loss = history.get("step_val_meta_loss", [])
        step_val_task_losses = history.get("step_val_task_losses", [])
        for step_idx, step_weights in enumerate(weights):
            row = {"method": method_name, "step": step_idx + 1}
            for task_idx, task_id in enumerate(task_ids):
                if task_idx < len(step_weights):
                    row[f"weight_{task_id}"] = float(step_weights[task_idx])

            if step_idx < len(step_meta_grad) and step_meta_grad[step_idx] is not None:
                for task_idx, task_id in enumerate(task_ids):
                    if task_idx < len(step_meta_grad[step_idx]):
                        row[f"meta_grad_{task_id}"] = float(
                            step_meta_grad[step_idx][task_idx]
                        )
            if step_idx < len(step_val_meta_loss):
                row["val_meta_loss"] = step_val_meta_loss[step_idx]
            if (
                step_idx < len(step_val_task_losses)
                and step_val_task_losses[step_idx] is not None
            ):
                for task_idx, task_id in enumerate(task_ids):
                    if task_idx < len(step_val_task_losses[step_idx]):
                        row[f"val_loss_{task_id}"] = float(
                            step_val_task_losses[step_idx][task_idx]
                        )
            step_rows.append(row)
        pd.DataFrame(step_rows).to_csv(step_output_path, index=False)

    epoch_rows = []
    for epoch_idx, loss_value in enumerate(history.get("losses", [])):
        row = {
            "method": method_name,
            "epoch": epoch_idx + 1,
            "loss": float(loss_value),
        }
        val_meta_loss = history.get("val_meta_loss", [])
        if epoch_idx < len(val_meta_loss):
            row["val_meta_loss"] = val_meta_loss[epoch_idx]

        val_task_losses = history.get("val_task_losses", [])
        if epoch_idx < len(val_task_losses) and val_task_losses[epoch_idx] is not None:
            for task_idx, task_id in enumerate(task_ids):
                if task_idx < len(val_task_losses[epoch_idx]):
                    row[f"val_loss_{task_id}"] = float(
                        val_task_losses[epoch_idx][task_idx]
                    )

        meta_grad = history.get("meta_grad", [])
        if epoch_idx < len(meta_grad) and meta_grad[epoch_idx] is not None:
            for task_idx, task_id in enumerate(task_ids):
                if task_idx < len(meta_grad[epoch_idx]):
                    row[f"meta_grad_{task_id}"] = float(meta_grad[epoch_idx][task_idx])
        epoch_rows.append(row)

    if epoch_rows:
        pd.DataFrame(epoch_rows).to_csv(epoch_output_path, index=False)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="wireless DADGP 单次多方法实验")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="要运行的方法列表，逗号分隔；默认运行全部方法",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="基础随机种子（默认 42）",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="禁用最小最大归一化",
    )
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.methods is None:
        methods_to_run = VALID_METHODS.copy()
    else:
        methods_to_run = [
            item.strip() for item in args.methods.split(",") if item.strip()
        ]
        invalid_methods = [
            method for method in methods_to_run if method not in VALID_METHODS
        ]
        if invalid_methods:
            print(f"警告: 以下方法无效，将被忽略: {', '.join(invalid_methods)}")
        methods_to_run = [
            method for method in methods_to_run if method in VALID_METHODS
        ]

    if not methods_to_run:
        print("没有有效的方法可运行。")
        return

    methods_str = ", ".join(methods_to_run)
    print(f"Running single experiment with methods ({methods_str})")
    if args.base_seed is not None:
        print(f"Using seed: {args.base_seed}")

    results, histories_by_method = run_single_experiment(
        device,
        NUM_EPOCHS,
        methods=methods_to_run,
        normalize=not args.disable_normalize,
        seed=args.base_seed,
    )

    for method_name, history in histories_by_method.items():
        plot_results(
            history,
            MODEL_DIR / f"{method_name}_training.png",
        )
        export_training_trace(
            history,
            method_name,
            MODEL_DIR / f"{method_name}_training_trace.csv",
            MODEL_DIR / f"{method_name}_epoch_trace.csv",
        )

    print("\n--- Saved Models ---")
    for method, metrics in results.items():
        print(f"{method} -> {metrics['model_path']}")

    print(f"\nTotal saved models: {len(results)}")
    print(f"Total training plots: {len(histories_by_method)}")


if __name__ == "__main__":
    main()
