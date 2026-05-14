# -*- coding: utf-8 -*-
"""
主实验入口：运行实验并保存数据和模型

本文件是运行所有实验的入口点，包含：
- set_seed：设置随机种子以确保可复现性
- run_single_experiment：运行单个完整实验（所有方法），只保存数据和模型
- main：主函数，运行多次实验

实验脚本只负责：
1. 生成并保存训练、验证、测试数据集
2. 训练模型并保存模型文件
3. 保存权重历史（DA-DGP方法）

命令行参数：
- --start-run: 起始运行编号（默认1）
- --end-run: 结束运行编号（默认20）
- --methods: 要运行的方法列表（逗号分隔，默认全部）
- --overwrite: 覆盖已存在的结果（默认询问）
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
import sys

from config import (
    NUM_EPOCHS, N_RUNS, NUM_HIDDEN_DGP_DIMS,
    TRAIN_TASKS, PRI_TASKS, TARGET_VALUES, SIGMA_VALUES,
    EXPERIMENT_SAMPLES, BASE_DIR, WEIGHT_INIT,
)
from models_dgp import (
    IndependentDeepGP,
    IndependentHeteroscedasticGP,
    LMCDeepGP,
    MultitaskDeepGP,
)
from data_generation import setup_experiment_data, setup_sample_weights
from common import create_loss_function
from algo_da_dgp import DADGP, run_training_loop
from algo_equal import run_baseline_training_loop
from algo_dwa import run_dwa_baseline_training_loop
from algo_uw import run_uncertainty_weighting_training_loop
from algo_mgda import run_mgda_training_loop
from algo_ablation_no_sample_attn import run_ablation_no_sample_attn
from algo_supplemental_baselines import (
    run_independent_dgp_training_loop,
    run_independent_hetgp_training_loop,
)


# ==========================================================================================
# 随机种子设置（确保可复现性）
# ==========================================================================================

def set_seed(seed: int):
    """设置全局随机种子以确保实验可复现。

    Args:
        seed: 随机种子值
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================================================================
# 模型保存与加载
# ==========================================================================================

def save_model(model, save_path: Path):
    """保存模型状态字典。"""
    torch.save(model.state_dict(), save_path)


def save_experiment_data(datasets: dict, save_dir: Path):
    """保存实验数据到指定目录（xlsx格式）。"""
    for split in ["train", "val", "test"]:
        data = datasets[split]
        x_np = data["x"].cpu().numpy()
        y_np = data["y"].cpu().numpy()

        n_samples = x_np.shape[0]
        n_input_dims = x_np.shape[1]
        n_tasks = y_np.shape[1]

        columns = [f"x{i+1}" for i in range(n_input_dims)] + [f"y{i+1}" for i in range(n_tasks)]
        data_array = np.hstack([x_np, y_np])
        df = pd.DataFrame(data_array, columns=columns)
        df.to_excel(save_dir / f"{split}_data.xlsx", index=False)


# ==========================================================================================
# 检查已有结果
# ==========================================================================================

# 方法对应的模型文件名
METHOD_MODEL_FILES = {
    "da_dgp": "model_da_dgp.pt",
    "baseline_equal": "model_baseline_equal.pt",
    "baseline_pure_dgp": "model_baseline_pure_dgp.pt",
    "baseline_dwa": "model_baseline_dwa.pt",
    "baseline_uw": "model_baseline_uw.pt",
    "baseline_mgda": "model_baseline_mgda.pt",
    "baseline_indep_dgp": "model_baseline_indep_dgp.pt",
    "baseline_indep_hetgp": "model_baseline_indep_hetgp.pt",
    "baseline_lmc_dgp": "model_baseline_lmc_dgp.pt",
    "ablation_no_sample_attn": "model_ablation_no_sample_attn.pt",
}

VALID_METHODS = [
    "da_dgp", "baseline_equal", "baseline_pure_dgp", "baseline_dwa",
    "baseline_uw", "baseline_mgda", "baseline_indep_dgp",
    "baseline_indep_hetgp", "baseline_lmc_dgp", "ablation_no_sample_attn"
]


def check_existing_results(run_dir: Path, methods: list) -> dict:
    """检查已存在的实验数据和模型。

    Args:
        run_dir: 实验运行目录
        methods: 要运行的方法列表

    Returns:
        dict: 包含已有数据和方法的字典
    """
    existing = {
        "data_exists": False,
        "existing_methods": [],
        "missing_methods": [],
    }

    if not run_dir.exists():
        return existing

    # 检查数据文件
    data_files = ["train_data.xlsx", "val_data.xlsx", "test_data.xlsx"]
    existing["data_exists"] = all((run_dir / f).exists() for f in data_files)

    # 检查模型文件
    for method in methods:
        model_file = METHOD_MODEL_FILES.get(method)
        if model_file and (run_dir / model_file).exists():
            existing["existing_methods"].append(method)

    existing["missing_methods"] = [m for m in methods if m not in existing["existing_methods"]]

    return existing


def prompt_overwrite(existing: dict, run_dir: Path) -> tuple:
    """询问用户是否覆盖已有数据和模型。

    Returns:
        (overwrite_data, overwrite_models): 是否覆盖数据和模型
    """
    print(f"\n检查目录: {run_dir}")
    print("-" * 40)

    if existing["data_exists"]:
        print("[!] 已存在数据文件: train_data.xlsx, val_data.xlsx, test_data.xlsx")

    if existing["existing_methods"]:
        print(f"[!] 已存在模型: {', '.join(existing['existing_methods'])}")

    if existing["missing_methods"]:
        print(f"[+] 需要运行的方法: {', '.join(existing['missing_methods'])}")

    if not existing["data_exists"] and not existing["existing_methods"]:
        print("[+] 目录为空，将运行全部实验")
        return (True, True)

    print("-" * 40)

    # 询问覆盖数据
    if existing["data_exists"]:
        response = input("是否覆盖已有数据文件? (y/n/a=全部跳过): ").strip().lower()
        if response == "a":
            return (False, False)
        overwrite_data = response == "y"
    else:
        overwrite_data = True

    # 询问覆盖模型
    if existing["existing_methods"]:
        response = input("是否覆盖已有模型? (y/n): ").strip().lower()
        overwrite_models = response == "y"
        if not overwrite_models:
            to_run = existing["missing_methods"]
            if to_run:
                print(f"将只运行: {', '.join(to_run)}")
            else:
                print("所有方法已完成")
    else:
        overwrite_models = True

    return (overwrite_data, overwrite_models)


# ==========================================================================================
# 单次实验运行
# ==========================================================================================

def run_single_experiment(device, num_epochs, methods=None, save_dir=None, seed=None,
                          overwrite_data=True, overwrite_models=True):
    """运行一次完整的实验，只保存数据和模型。

    Args:
        device: 计算设备
        num_epochs: 训练轮数
        methods: 要运行的方法列表
        save_dir: 保存目录
        seed: 随机种子
        overwrite_data: 是否覆盖已有数据
        overwrite_models: 是否覆盖已有模型

    Returns:
        无返回值
    """
    if seed is not None:
        set_seed(seed)

    if methods is None:
        methods = VALID_METHODS.copy()

    # 检查已有结果，确定要运行的方法
    if save_dir is not None and save_dir.exists():
        existing = check_existing_results(save_dir, methods)

        if not overwrite_models and existing["existing_methods"]:
            methods_to_run = existing["missing_methods"]
            if not methods_to_run:
                print(f"\n[+] 所有方法已完成，跳过运行")
                return
            methods = methods_to_run
            print(f"[+] 将运行: {', '.join(methods_to_run)}")
        elif overwrite_models:
            methods_to_run = methods
        else:
            methods_to_run = methods
    else:
        methods_to_run = methods

    # 设置数据
    datasets = setup_experiment_data(device, samples=EXPERIMENT_SAMPLES, seed=seed)
    train_data, val_data, test_data = (
        datasets["train"],
        datasets["val"],
        datasets["test"],
    )

    # 保存数据
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        if overwrite_data or not (save_dir / "train_data.xlsx").exists():
            save_experiment_data(datasets, save_dir)

    num_tasks = train_data["y"].size(-1)
    sample_weights = setup_sample_weights(train_data, val_data, TARGET_VALUES, SIGMA_VALUES)

    def reset_method_seed(method_name):
        """确保同一次实验内的每个方法都从相同随机起点开始。"""
        if seed is not None:
            set_seed(seed)
            print(f"[seed] reset to {seed} for {method_name}")

    # ==================== DA-DGP ====================
    if "da_dgp" in methods_to_run:
        print("\n=== Running DA-DGP ===")
        reset_method_seed("da_dgp")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        dadgp = DADGP(
            model, device, TRAIN_TASKS, PRI_TASKS,
            weight_init=WEIGHT_INIT, sample_weights=sample_weights
        )
        dadgp.loss_fn = create_loss_function(
            sample_weights=sample_weights,
            split_sizes={
                "train": train_data["y"].size(0),
                "val": val_data["y"].size(0),
            },
        )
        history = run_training_loop(model, dadgp, datasets, num_epochs)

        if save_dir is not None:
            save_model(model, save_dir / "model_da_dgp.pt")
            # 保存权重历史
            if history is not None:
                weight_records = []
                weights_array = np.array(history["weights"])
                n_steps = weights_array.shape[0]
                losses_per_epoch = len(history["losses"])
                steps_per_epoch = n_steps // losses_per_epoch if losses_per_epoch > 0 else n_steps

                for step_idx in range(n_steps):
                    epoch = step_idx // steps_per_epoch + 1 if steps_per_epoch > 0 else 1
                    batch_in_epoch = step_idx % steps_per_epoch + 1 if steps_per_epoch > 0 else step_idx + 1
                    record = {"step": step_idx + 1, "epoch": epoch, "batch": batch_in_epoch}
                    for task_idx, task_id in enumerate(TRAIN_TASKS):
                        record[f"weight_{task_id}"] = weights_array[step_idx, task_idx]
                    weight_records.append(record)

                for epoch_idx, loss in enumerate(history["losses"]):
                    if epoch_idx < len(weight_records):
                        weight_records[epoch_idx * steps_per_epoch]["epoch_loss"] = loss

                weight_df = pd.DataFrame(weight_records)
                weight_df.to_excel(save_dir / "weight_history_da_dgp.xlsx", index=False)

    # ==================== 等权重基线 ====================
    if "baseline_equal" in methods_to_run:
        print("\n=== Running Baseline Equal ===")
        reset_method_seed("baseline_equal")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_baseline_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), sample_weights)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_equal.pt")

    # ==================== Pure DGP基线 ====================
    if "baseline_pure_dgp" in methods_to_run:
        print("\n=== Running Baseline Pure DGP ===")
        reset_method_seed("baseline_pure_dgp")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_baseline_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), None)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_pure_dgp.pt")

    # ==================== DWA基线 ====================
    if "baseline_dwa" in methods_to_run:
        print("\n=== Running Baseline DWA ===")
        reset_method_seed("baseline_dwa")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_dwa_baseline_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), sample_weights)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_dwa.pt")

    # ==================== UW基线 ====================
    if "baseline_uw" in methods_to_run:
        print("\n=== Running Baseline Uncertainty Weighting ===")
        reset_method_seed("baseline_uw")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_uncertainty_weighting_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), sample_weights)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_uw.pt")

    # ==================== MGDA基线 ====================
    if "baseline_mgda" in methods_to_run:
        print("\n=== Running Baseline MGDA ===")
        reset_method_seed("baseline_mgda")
        model = MultitaskDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_mgda_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), sample_weights)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_mgda.pt")

    # ==================== Indep-DGP基线 ====================
    if "baseline_indep_dgp" in methods_to_run:
        print("\n=== Running Baseline Indep-DGP ===")
        reset_method_seed("baseline_indep_dgp")
        model = IndependentDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_independent_dgp_training_loop(model, datasets, num_epochs)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_indep_dgp.pt")

    # ==================== Indep-HetGP基线 ====================
    if "baseline_indep_hetgp" in methods_to_run:
        print("\n=== Running Baseline Indep-HetGP ===")
        reset_method_seed("baseline_indep_hetgp")
        model = IndependentHeteroscedasticGP(train_data["x"], train_data["y"]).to(device)
        run_independent_hetgp_training_loop(model, datasets, num_epochs)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_indep_hetgp.pt")

    # ==================== LMC-DGP基线 ====================
    if "baseline_lmc_dgp" in methods_to_run:
        print("\n=== Running Baseline LMC-DGP ===")
        reset_method_seed("baseline_lmc_dgp")
        model = LMCDeepGP(
            train_data["x"].shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        ).to(device)
        run_baseline_training_loop(model, datasets, num_epochs, list(TRAIN_TASKS.keys()), None)

        if save_dir is not None:
            save_model(model, save_dir / "model_baseline_lmc_dgp.pt")

    # ==================== 消融实验：无样本加权 ====================
    if "ablation_no_sample_attn" in methods_to_run:
        print("\n=== Running Ablation: No Sample Attention ===")
        reset_method_seed("ablation_no_sample_attn")
        ablation_model = run_ablation_no_sample_attn(
            device, num_epochs, datasets=datasets, seed=seed
        )

        if save_dir is not None:
            save_model(ablation_model, save_dir / "model_ablation_no_sample_attn.pt")


# ==========================================================================================
# 主函数
# ==========================================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="DA-DGP多任务实验（只保存数据和模型）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main_experiment.py                      # 运行全部实验(1-N_RUNS)
  python main_experiment.py --start-run 5 --end-run 10  # 运行第5-10次
  python main_experiment.py --methods da_dgp    # 只运行DA-DGP
  python main_experiment.py --overwrite         # 强制覆盖已有结果

可用方法:
  da_dgp, baseline_equal, baseline_pure_dgp, baseline_dwa, baseline_uw,
  baseline_mgda, baseline_indep_dgp, baseline_indep_hetgp, baseline_lmc_dgp,
  ablation_no_sample_attn
        """
    )
    parser.add_argument("--start-run", type=int, default=1, help="起始运行编号（默认: 1）")
    parser.add_argument("--end-run", type=int, default=N_RUNS, help=f"结束运行编号（默认: {N_RUNS}）")
    parser.add_argument("--methods", type=str, default=None, help="要运行的方法列表（逗号分隔）")
    parser.add_argument("--base-seed", type=int, default=42, help="基础随机种子（默认: 42）")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已有结果，不询问")
    parser.add_argument("--skip-all", action="store_true", help="跳过所有已有结果，只运行缺失的")
    return parser.parse_args()


def main():
    """主函数，运行完整的实验。"""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = NUM_EPOCHS
    start_run = args.start_run
    end_run = args.end_run

    if start_run > end_run:
        print(f"错误: start-run ({start_run}) 大于 end-run ({end_run})")
        return
    if start_run < 1:
        print(f"错误: start-run ({start_run}) 必须大于等于 1")
        return

    n_runs = end_run - start_run + 1
    BASE_SEED = args.base_seed

    # 解析方法
    if args.methods is None:
        methods_to_run = VALID_METHODS.copy()
    else:
        methods_to_run = [m.strip() for m in args.methods.split(",")]

    # 验证方法
    for method in methods_to_run:
        if method not in VALID_METHODS:
            print(f"警告: 方法 '{method}' 不是有效方法，将被跳过")
    methods_to_run = [m for m in methods_to_run if m in VALID_METHODS]

    if not methods_to_run:
        print("错误: 没有有效的方法可运行")
        return

    # 配置信息
    print(f"\n{'='*60}")
    print("实验配置:")
    print(f"  - 运行范围: {start_run} - {end_run} (共 {n_runs} 次)")
    print(f"  - 基础种子: {BASE_SEED}")
    print(f"  - 方法列表: {', '.join(methods_to_run)}")
    print(f"  - 训练轮数: {num_epochs}")
    print(f"  - 覆盖模式: {'强制覆盖' if args.overwrite else '询问确认' if not args.skip_all else '跳过已有'}")
    print(f"{'='*60}\n")

    achievements_dir = BASE_DIR / "Achievements"
    achievements_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start_run, end_run + 1):
        run_id = i
        run_seed = BASE_SEED + run_id
        run_dir = achievements_dir / f"{run_id:02d}"

        print(f"\n{'='*60}")
        print(f"Experiment {run_id} ({run_id - start_run + 1}/{n_runs}) (seed={run_seed})")
        print(f"目录: {run_dir}")
        print(f"{'='*60}")

        # 检查已有结果
        if run_dir.exists() and not args.overwrite:
            existing = check_existing_results(run_dir, methods_to_run)

            if args.skip_all:
                # 跳过已有，只运行缺失
                overwrite_data = False
                overwrite_models = False
                to_run = existing["missing_methods"]
                if to_run:
                    print(f"[+] 跳过已有方法，运行: {', '.join(to_run)}")
                elif existing["data_exists"] and existing["existing_methods"]:
                    print(f"[+] 所有数据和模型已完成，跳过")
                    continue
                else:
                    print(f"[+] 跳过此实验")
                    continue
            else:
                # 询问用户
                overwrite_data, overwrite_models = prompt_overwrite(existing, run_dir)
                if not overwrite_data and not overwrite_models and not existing["missing_methods"]:
                    print("[+] 跳过此实验")
                    continue
        else:
            overwrite_data = True
            overwrite_models = True

        run_single_experiment(
            device, num_epochs, methods=methods_to_run,
            save_dir=run_dir, seed=run_seed,
            overwrite_data=overwrite_data, overwrite_models=overwrite_models
        )

        print(f"\n--- Finished Experiment {run_id} ({run_id - start_run + 1}/{n_runs}) ---")

    # 保存配置
    config_info = {
        "start_run": start_run,
        "end_run": end_run,
        "n_runs": n_runs,
        "num_epochs": num_epochs,
        "base_seed": BASE_SEED,
        "methods": methods_to_run,
        "device": str(device),
    }
    config_path = achievements_dir / f"experiment_config_{start_run:02d}-{end_run:02d}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"\n保存配置: '{config_path}'")

    print("\n" + "="*60)
    print("所有实验完成!")
    print("数据、模型和权重历史已保存。")
    print("="*60)


if __name__ == "__main__":
    main()
