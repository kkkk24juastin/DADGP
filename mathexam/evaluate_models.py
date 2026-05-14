# -*- coding: utf-8 -*-
"""
模型评估脚本：加载训练好的模型，在测试集上评估性能指标

本脚本用于在实验完成后独立评估模型性能：
- 加载已保存的模型和数据
- 先筛选局部测试点，再对局部测试点进行预测
- 计算局部指标（RMSE、NLPD、质量损失）
- 保存评估结果和 mean/std/95% CI 汇总文件

命令行参数：
- --run-id: 运行编号（单个评估）
- --start-run: 起始运行编号
- --end-run: 结束运行编号
- --methods: 要评估的方法列表（逗号分隔）
- --use-existing-metrics: 复用已有 local_metrics.xlsx 直接汇总
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from config import (
    BASE_DIR, NUM_HIDDEN_DGP_DIMS, TARGET_VALUES,
    PREDICT_BATCH_SIZE,
)
from models_dgp import (
    IndependentDeepGP,
    IndependentHeteroscedasticGP,
    LMCDeepGP,
    MultitaskDeepGP,
)
from metrics import evaluate_metrics, select_local_data


# ==========================================================================================
# 模型和数据加载
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

METRIC_TYPES = ["rmse", "nlpd", "quality_loss"]
NUM_TASKS = 3
CI_Z = 1.96


def format_pm(mean_value: float, spread_value: float, digits: int = 6) -> str:
    """格式化 mean ± spread，便于直接写入论文补充表。"""
    if not np.isfinite(mean_value) or not np.isfinite(spread_value):
        return ""
    return f"{mean_value:.{digits}g} ± {spread_value:.{digits}g}"


def summarize_values(values) -> dict:
    """计算均值、样本标准差和95%置信区间半宽。"""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    count = int(arr.size)

    if count == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "mean±std": "",
            "mean±ci95": "",
        }

    mean_value = float(np.mean(arr))
    std_value = float(np.std(arr, ddof=1)) if count > 1 else np.nan
    ci95 = float(CI_Z * std_value / np.sqrt(count)) if count > 1 else np.nan

    return {
        "n": count,
        "mean": mean_value,
        "std": std_value,
        "ci95": ci95,
        "ci95_lower": mean_value - ci95 if np.isfinite(ci95) else np.nan,
        "ci95_upper": mean_value + ci95 if np.isfinite(ci95) else np.nan,
        "mean±std": format_pm(mean_value, std_value),
        "mean±ci95": format_pm(mean_value, ci95),
    }


def build_metric_summary_tables(
    grouped_results: dict,
    group_order: list,
    group_column: str = "method",
):
    """为多组实验结果构建宽表和长表统计汇总。"""
    summary_wide_dfs = {}
    summary_long_dfs = {}
    task_index = [f"task{i}" for i in range(1, NUM_TASKS + 1)]

    for metric in METRIC_TYPES:
        wide_data = {}
        long_records = []

        for group_name in group_order:
            group_metrics = grouped_results.get(group_name, [])
            means, stds, mean_pm_stds = [], [], []
            ci95s, ci95_lowers, ci95_uppers, counts = [], [], [], []
            mean_pm_ci95s = []

            for task in range(1, NUM_TASKS + 1):
                key = f"local_{metric}_task{task}"
                values = [item.get(key, float("nan")) for item in group_metrics]
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
                    group_column: group_name,
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

            # 保留原方法名列作为均值列，兼容已有阅读习惯。
            wide_data[group_name] = means
            wide_data[f"{group_name}_std"] = stds
            wide_data[f"{group_name}_mean±std"] = mean_pm_stds
            wide_data[f"{group_name}_ci95"] = ci95s
            wide_data[f"{group_name}_ci95_lower"] = ci95_lowers
            wide_data[f"{group_name}_ci95_upper"] = ci95_uppers
            wide_data[f"{group_name}_mean±ci95"] = mean_pm_ci95s
            wide_data[f"{group_name}_n"] = counts

        summary_wide_dfs[metric] = pd.DataFrame(wide_data, index=task_index)
        summary_long_dfs[metric] = pd.DataFrame(long_records)

    return summary_wide_dfs, summary_long_dfs


def create_model_for_method(
    method_name: str,
    train_data: dict,
    num_tasks: int,
    device: torch.device,
):
    """按方法名重建对应的模型结构。"""
    train_x_shape = train_data["x"].shape

    if method_name == "baseline_indep_dgp":
        model = IndependentDeepGP(
            train_x_shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        )
    elif method_name == "baseline_indep_hetgp":
        model = IndependentHeteroscedasticGP(train_data["x"], train_data["y"])
    elif method_name == "baseline_lmc_dgp":
        model = LMCDeepGP(
            train_x_shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        )
    else:
        model = MultitaskDeepGP(
            train_x_shape, num_hidden_dgp_dims=NUM_HIDDEN_DGP_DIMS, num_tasks=num_tasks
        )

    return model.to(device)


def load_model(
    model_path: Path,
    method_name: str,
    train_data: dict,
    num_tasks: int,
    device: torch.device,
):
    """加载保存的模型状态字典。

    Args:
        model_path: 模型文件路径
        method_name: 方法名称
        train_data: 训练数据（用于确定输入维度，Indep-HetGP还用于重建固定噪声）
        num_tasks: 任务数量
        device: 计算设备

    Returns:
        加载后的模型实例
    """
    model = create_model_for_method(method_name, train_data, num_tasks, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_data(data_path: Path, device: torch.device):
    """加载保存的数据文件（xlsx格式）。

    Args:
        data_path: 数据文件路径
        device: 计算设备

    Returns:
        数据字典 {"x": tensor, "y": tensor}
    """
    df = pd.read_excel(data_path)

    # 提取输入和输出列
    x_cols = [col for col in df.columns if col.startswith("x")]
    y_cols = [col for col in df.columns if col.startswith("y")]

    x_np = df[x_cols].values
    y_np = df[y_cols].values

    return {
        "x": torch.from_numpy(x_np).float().to(device),
        "y": torch.from_numpy(y_np).float().to(device)
    }


def evaluate_local_predictions(
    model,
    local_test_data: dict,
    target_values,
    num_tasks: int,
):
    """只在已筛选的局部测试点上预测并计算局部指标。"""
    if local_test_data["x"].size(0) == 0:
        mean_pred = local_test_data["y"].new_empty((0, num_tasks))
        var_pred = local_test_data["y"].new_empty((0, num_tasks))
    else:
        mean_pred, var_pred = model.predict(
            local_test_data["x"], batch_size=PREDICT_BATCH_SIZE
        )

    return evaluate_metrics(mean_pred, var_pred, local_test_data["y"], target_values)


def load_weights_from_history(weight_history_path: Path):
    """从权重历史文件加载最终权重。

    Args:
        weight_history_path: 权重历史文件路径

    Returns:
        最终权重字典 {"weight_task1": value, "weight_task2": value, "weight_task3": value}
    """
    if not weight_history_path.exists():
        return None

    df = pd.read_excel(weight_history_path)

    # 获取最后一行的权重
    last_row = df.iloc[-1]

    weights = {}
    task_columns = {
        "task1": ["weight_task1", "weight_local_A"],
        "task2": ["weight_task2", "weight_local_B"],
        "task3": ["weight_task3", "weight_local_C"],
    }
    for task_id, candidate_cols in task_columns.items():
        for col_name in candidate_cols:
            if col_name in df.columns:
                weights[f"weight_{task_id}"] = last_row[col_name]
                break

    return weights


def load_weights_from_file(weights_path: Path):
    """从单次运行的 weights.xlsx 读取最终任务权重。"""
    if not weights_path.exists():
        return {}

    df = pd.read_excel(weights_path)
    index_col = df.columns[0]
    if "da_dgp" not in df.columns:
        return {}

    weights = {}
    for task in range(1, NUM_TASKS + 1):
        row = df[df[index_col] == f"task{task}"]
        if not row.empty:
            weights[f"weight_task{task}"] = row["da_dgp"].values[0]
    return weights


def load_saved_run_results(run_dir: Path, methods: list):
    """从已有 local_metrics.xlsx/weights.xlsx 读取单次运行结果。"""
    local_metrics_path = run_dir / "local_metrics.xlsx"
    if not local_metrics_path.exists():
        return None

    results = {}
    try:
        for metric in METRIC_TYPES:
            df = pd.read_excel(local_metrics_path, sheet_name=metric)
            index_col = df.columns[0]
            for method in methods:
                if method not in df.columns:
                    continue
                method_metrics = results.setdefault(method, {})
                for task in range(1, NUM_TASKS + 1):
                    row = df[df[index_col] == f"task{task}"]
                    if not row.empty:
                        method_metrics[f"local_{metric}_task{task}"] = row[method].values[0]
    except Exception as exc:
        print(f"[!] 读取已有指标失败: {local_metrics_path}: {exc}")
        return None

    if not results:
        return None

    missing_methods = [method for method in methods if method not in results]
    if missing_methods:
        print(f"[!] 已有指标缺少方法: {', '.join(missing_methods)}")

    if "da_dgp" in results:
        weights = load_weights_from_history(run_dir / "weight_history_da_dgp.xlsx")
        if not weights:
            weights = load_weights_from_file(run_dir / "weights.xlsx")
        results["da_dgp"].update(weights)

    return results


# ==========================================================================================
# 模型评估
# ==========================================================================================

def evaluate_single_method(
    model_path: Path,
    local_test_data: dict,
    train_data: dict,
    num_tasks: int,
    device: torch.device,
    method_name: str,
    run_dir: Path = None
):
    """评估单个方法的模型性能。

    Args:
        model_path: 模型文件路径
        local_test_data: 已筛选的局部测试数据字典
        train_data: 训练数据字典
        num_tasks: 任务数量
        device: 计算设备
        method_name: 方法名称
        run_dir: 运行目录（用于加载权重历史）

    Returns:
        结果字典，包含指标和权重信息
    """
    if not model_path.exists():
        print(f"[!] 模型文件不存在: {model_path}")
        return None

    # 加载模型
    model = load_model(model_path, method_name, train_data, num_tasks, device)

    # 只在局部测试点上预测并计算指标
    metrics = evaluate_local_predictions(
        model, local_test_data, TARGET_VALUES, num_tasks
    )

    # 如果是DA-DGP方法，加载权重历史中的最终权重
    if method_name == "da_dgp" and run_dir is not None:
        weight_history_path = run_dir / "weight_history_da_dgp.xlsx"
        weights = load_weights_from_history(weight_history_path)
        if weights:
            metrics.update(weights)

    return metrics


def evaluate_run(run_dir: Path, methods: list, device: torch.device):
    """评估单个运行的所有方法。

    Args:
        run_dir: 运行目录
        methods: 要评估的方法列表
        device: 计算设备

    Returns:
        结果字典 {method_name: metrics_dict}
    """
    # 检查数据文件是否存在
    test_data_path = run_dir / "test_data.xlsx"
    train_data_path = run_dir / "train_data.xlsx"

    if not test_data_path.exists():
        print(f"[!] 测试数据不存在: {test_data_path}")
        return None

    # 加载测试数据
    test_data = load_data(test_data_path, device)

    # 加载训练数据以获取输入形状
    train_data = load_data(train_data_path, device)
    num_tasks = test_data["y"].shape[-1]
    local_test_data, local_mask = select_local_data(test_data, TARGET_VALUES)
    local_count = int(local_mask.sum().item())
    total_count = int(test_data["y"].shape[0])
    print(f"  局部测试点: {local_count}/{total_count}")

    results = {}

    for method in methods:
        model_file = METHOD_MODEL_FILES.get(method)
        if model_file is None:
            print(f"[!] 未知方法: {method}")
            continue

        model_path = run_dir / model_file
        print(f"  评估方法: {method}")

        metrics = evaluate_single_method(
            model_path, local_test_data, train_data, num_tasks,
            device, method, run_dir
        )

        if metrics:
            results[method] = metrics

    return results


# ==========================================================================================
# 结果保存
# ==========================================================================================

def save_results(results: dict, save_dir: Path):
    """保存评估结果到xlsx文件。

    保存格式：
    - local_metrics.xlsx: 局部指标，包含3个sheet (rmse, nlpd, quality_loss)
    - weights.xlsx: 任务权重信息（仅DA-DGP方法）
    """
    methods = list(results.keys())

    # 构建局部指标DataFrame
    local_dfs = {}
    for metric in METRIC_TYPES:
        data = {}
        for method in methods:
            row = []
            for task in range(1, NUM_TASKS + 1):
                key = f"local_{metric}_task{task}"
                value = results[method].get(key, float("nan"))
                row.append(value)
            data[method] = row
        local_dfs[metric] = pd.DataFrame(
            data,
            index=[f"task{i}" for i in range(1, NUM_TASKS + 1)],
        )

    # 构建权重DataFrame（如果有权重信息）
    weights_data = {}
    for method in methods:
        if method == "da_dgp":
            row = []
            for task_id in ["task1", "task2", "task3"]:
                key = f"weight_{task_id}"
                value = results[method].get(key, float("nan"))
                row.append(value)
            weights_data[method] = row
    if weights_data:
        weights_df = pd.DataFrame(
            weights_data,
            index=[f"task{i}" for i in range(1, NUM_TASKS + 1)],
        )

    # 保存局部指标
    local_path = save_dir / "local_metrics.xlsx"
    with pd.ExcelWriter(local_path) as writer:
        for metric, df in local_dfs.items():
            df.to_excel(writer, sheet_name=metric)

    # 保存权重
    if weights_data:
        weights_path = save_dir / "weights.xlsx"
        weights_df.to_excel(weights_path)

    print(f"[+] 结果已保存到: {save_dir}")


# ==========================================================================================
# 汇总结果
# ==========================================================================================

def aggregate_results(all_results: list, start_run: int, end_run: int, achievements_dir: Path):
    """汇总多次运行的评估结果。

    Args:
        all_results: 所有运行的结果列表
        start_run: 起始运行编号
        end_run: 结束运行编号
        achievements_dir: 汇总结果保存目录
    """
    all_methods = set()
    for run_result in all_results:
        all_methods.update(run_result.keys())
    ordered_known_methods = [method for method in VALID_METHODS if method in all_methods]
    ordered_extra_methods = sorted(all_methods - set(VALID_METHODS))
    all_methods = ordered_known_methods + ordered_extra_methods

    # 构建每次运行的详细数据
    local_all_dfs = {}
    weights_all_dfs = {}

    for metric in METRIC_TYPES:
        # 局部指标汇总
        records = []
        for run_idx, run_result in enumerate(all_results, start=start_run):
            for method in all_methods:
                if method in run_result:
                    row = {"run": run_idx, "method": method}
                    for task in range(1, NUM_TASKS + 1):
                        key = f"local_{metric}_task{task}"
                        row[f"task{task}"] = run_result[method].get(key, float("nan"))
                    records.append(row)
        local_all_dfs[metric] = pd.DataFrame(records)

    # 权重汇总（仅DA-DGP）
    if "da_dgp" in all_methods:
        records = []
        for run_idx, run_result in enumerate(all_results, start=start_run):
            if "da_dgp" in run_result:
                row = {"run": run_idx}
                for task in range(1, NUM_TASKS + 1):
                    key = f"weight_task{task}"
                    row[f"task{task}"] = run_result["da_dgp"].get(key, float("nan"))
                records.append(row)
        weights_all_dfs["weights"] = pd.DataFrame(records)

    # 保存每次运行的详细汇总
    local_all_path = achievements_dir / f"local_metrics_all_runs_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(local_all_path) as writer:
        for metric, df in local_all_dfs.items():
            df.to_excel(writer, sheet_name=metric, index=False)
    print(f"保存局部指标汇总: '{local_all_path}'")

    if weights_all_dfs:
        weights_all_path = achievements_dir / f"weights_all_runs_{start_run:02d}-{end_run:02d}.xlsx"
        with pd.ExcelWriter(weights_all_path) as writer:
            for name, df in weights_all_dfs.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print(f"保存权重汇总: '{weights_all_path}'")

    # 计算均值、标准差和置信区间汇总
    method_results = {}
    for run_result in all_results:
        for method, metrics in run_result.items():
            method_results.setdefault(method, []).append(metrics)

    local_mean_dfs, local_summary_long_dfs = build_metric_summary_tables(
        method_results,
        all_methods,
        group_column="method",
    )
    weights_mean_df = None

    # 权重均值、标准差和置信区间
    if "da_dgp" in method_results:
        data = {
            "da_dgp": [],
            "da_dgp_std": [],
            "da_dgp_mean±std": [],
            "da_dgp_ci95": [],
            "da_dgp_ci95_lower": [],
            "da_dgp_ci95_upper": [],
            "da_dgp_mean±ci95": [],
            "da_dgp_n": [],
        }
        for task in range(1, NUM_TASKS + 1):
            key = f"weight_task{task}"
            values = [m.get(key, float("nan")) for m in method_results["da_dgp"]]
            summary = summarize_values(values)
            data["da_dgp"].append(summary["mean"])
            data["da_dgp_std"].append(summary["std"])
            data["da_dgp_mean±std"].append(summary["mean±std"])
            data["da_dgp_ci95"].append(summary["ci95"])
            data["da_dgp_ci95_lower"].append(summary["ci95_lower"])
            data["da_dgp_ci95_upper"].append(summary["ci95_upper"])
            data["da_dgp_mean±ci95"].append(summary["mean±ci95"])
            data["da_dgp_n"].append(summary["n"])
        weights_mean_df = pd.DataFrame(
            data,
            index=[f"task{i}" for i in range(1, NUM_TASKS + 1)],
        )

    # 保存统计汇总
    local_mean_path = achievements_dir / f"local_metrics_mean_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(local_mean_path) as writer:
        for metric, df in local_mean_dfs.items():
            df.to_excel(writer, sheet_name=metric)
    print(f"保存局部指标统计汇总: '{local_mean_path}'")

    local_summary_path = achievements_dir / f"local_metrics_summary_{start_run:02d}-{end_run:02d}.xlsx"
    with pd.ExcelWriter(local_summary_path) as writer:
        for metric, df in local_summary_long_dfs.items():
            df.to_excel(writer, sheet_name=metric, index=False)
    print(f"保存局部指标长表统计: '{local_summary_path}'")

    if weights_mean_df is not None:
        weights_mean_path = achievements_dir / f"weights_mean_{start_run:02d}-{end_run:02d}.xlsx"
        weights_mean_df.to_excel(weights_mean_path)
        print(f"保存权重统计汇总: '{weights_mean_path}'")

    # 显示局部指标统计值
    print("\nSummary - Local Metrics (mean/std/95% CI over runs):")
    for metric in METRIC_TYPES:
        print(f"\n{metric}:")
        print(local_mean_dfs[metric])


# ==========================================================================================
# 主函数
# ==========================================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate_models.py --run-id 5              # 评估单个运行
  python evaluate_models.py --start-run 1 --end-run 20  # 评估多次运行
  python evaluate_models.py --start-run 1 --end-run 20 --use-existing-metrics  # 只重新汇总
  python evaluate_models.py --methods da_dgp        # 只评估DA-DGP方法
        """
    )
    parser.add_argument("--run-id", type=int, default=None, help="单个运行编号")
    parser.add_argument("--start-run", type=int, default=None, help="起始运行编号")
    parser.add_argument("--end-run", type=int, default=None, help="结束运行编号")
    parser.add_argument("--methods", type=str, default=None, help="要评估的方法列表（逗号分隔）")
    parser.add_argument(
        "--use-existing-metrics",
        action="store_true",
        help="若 run 目录已有 local_metrics.xlsx，则直接读取并汇总，不重新预测。",
    )
    return parser.parse_args()


def main():
    """主函数，执行模型评估。"""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    achievements_dir = BASE_DIR / "Achievements"

    # 确定运行范围
    if args.run_id is not None:
        start_run = args.run_id
        end_run = args.run_id
    elif args.start_run is not None and args.end_run is not None:
        start_run = args.start_run
        end_run = args.end_run
    else:
        # 默认评估所有运行
        # 查找Achievements目录下的所有运行目录
        run_dirs = sorted([d for d in achievements_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if not run_dirs:
            print("[!] 没有找到任何运行目录")
            return
        start_run = int(run_dirs[0].name)
        end_run = int(run_dirs[-1].name)

    # 确定方法
    if args.methods is None:
        methods = VALID_METHODS.copy()
    else:
        methods = [m.strip() for m in args.methods.split(",")]
        # 验证方法
        methods = [m for m in methods if m in VALID_METHODS]

    if not methods:
        print("[!] 没有有效的方法可评估")
        return

    print(f"\n{'='*60}")
    print("评估配置:")
    print(f"  - 运行范围: {start_run} - {end_run}")
    print(f"  - 方法列表: {', '.join(methods)}")
    print(f"  - 复用已有指标: {args.use_existing_metrics}")
    print(f"{'='*60}\n")

    all_results = []

    for run_id in range(start_run, end_run + 1):
        run_dir = achievements_dir / f"{run_id:02d}"

        if not run_dir.exists():
            print(f"[!] 运行目录不存在: {run_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"评估运行 {run_id}")
        print(f"目录: {run_dir}")
        print(f"{'='*60}")

        loaded_existing = False
        results = None
        if args.use_existing_metrics:
            results = load_saved_run_results(run_dir, methods)
            if results:
                loaded_existing = True
                print("  已复用已有 local_metrics.xlsx")
            else:
                print("  未找到可复用的已有指标，改为重新评估")

        if results is None:
            results = evaluate_run(run_dir, methods, device)

        if results:
            # 保存单次运行的结果
            if not loaded_existing:
                save_results(results, run_dir)
            all_results.append(results)
        else:
            print(f"[!] 运行 {run_id} 没有可用的评估结果")
            all_results.append({})

        print(f"\n--- 完成评估运行 {run_id} ---")

    # 汇总结果
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("--- 汇总评估结果 ---")
        print("="*60)
        aggregate_results(all_results, start_run, end_run, achievements_dir)

    print("\n" + "="*60)
    print("所有评估完成!")
    print("="*60)


if __name__ == "__main__":
    main()
