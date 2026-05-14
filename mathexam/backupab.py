# 消融实验：补充两个缺失的基线
# -Sample Attn: AutoLambda任务调权，但无样本加权
# -Both: 无样本加权 + 任务等权

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 从主文件导入必要的组件
from backup import (
    MultitaskDeepGP,
    AutoLambda,
    IndexedTensorDataset,
    setup_experiment_data,
    evaluate_metrics,
    create_loss_function,
)

# ==========================================================================================
# Ablation 1: -Sample Attn (AutoLambda without sample weighting)
# ==========================================================================================

def run_autolambda_no_sample_weights(model, autol, datasets, num_epochs):
    """执行AutoLambda训练，但不使用样本加权（消融样本注意力）。"""
    train_data, val_data = datasets['train'], datasets['val']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    train_x_full, train_y_full, train_indices_full = next(iter(full_train_loader))

    val_dataset = IndexedTensorDataset(val_data['x'], val_data['y'])
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    meta_optimizer = torch.optim.Adam([autol.meta_weights], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=0.999)

    history = {'weights': [], 'losses': []}

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"Ablation(-SampleAttn) Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            # 模型参数更新
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = autol.model_fit(output, train_y_batch, train_indices, is_val=False)
            normalized_weights = autol.get_normalized_weights()
            total_loss = sum(w * l for w, l in zip(normalized_weights, task_losses))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 元更新步骤
            meta_optimizer.zero_grad()
            autol.unrolled_backward(
                train_x_full, train_y_full, train_indices_full,
                val_x_full, val_y_full, val_indices_full,
                optimizer.param_groups[0]['lr'], optimizer
            )
            meta_optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            history['weights'].append(autol.get_normalized_weights().detach().cpu().numpy().copy())
            epochs_iter.set_postfix(loss=total_loss.item())

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
        history['losses'].append(avg_epoch_loss)
        scheduler.step()
        meta_scheduler.step()

        weights_str = ', '.join(f"{w:.4f}" for w in autol.get_normalized_weights().detach().cpu().numpy())
        print(f"Ablation(-SampleAttn) Epoch {epoch+1} finished. Loss: {avg_epoch_loss:.4f}, Weights: [{weights_str}]")

    return history


# ==========================================================================================
# Ablation 2: -Both (No sample weighting + Equal task weights)
# ==========================================================================================

def run_baseline_no_both(model, datasets, num_epochs, task_ids):
    """执行无样本加权+任务等权的基线训练（消融两层注意力）。"""
    train_data = datasets['train']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # 关键：不使用样本权重
    loss_fn = create_loss_function(model, sample_weights=None)

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"Ablation(-Both) Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            # 任务等权重
            task_losses = [loss_fn(output, train_y_batch, task_id, train_indices, is_val=False) for task_id in task_ids]
            total_loss = torch.mean(torch.stack(task_losses))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            epochs_iter.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Ablation(-Both) Epoch {epoch+1} finished. Loss: {avg_epoch_loss:.4f}")


# ==========================================================================================
# Main Experiment Runner
# ==========================================================================================

def run_ablation_experiments(device, num_epochs, methods=None):
    """运行消融实验。

    Args:
        device: 计算设备
        num_epochs: 训练轮数
        methods: 要运行的方法列表，可选值: ['ablation_no_sample_attn', 'ablation_no_both']
    """
    if methods is None:
        methods = ['ablation_no_sample_attn', 'ablation_no_both']

    # 设置数据（与主实验保持一致）
    datasets = setup_experiment_data(device, samples=(400, 400, 5000))
    train_data, test_data = datasets['train'], datasets['test']
    num_tasks = train_data['y'].size(-1)

    # 任务定义（与主实验保持一致）
    train_tasks = {'local_A': 1, 'local_B': 1, 'local_C': 1}
    pri_tasks = {'local_A': 1, 'local_B': 1, 'local_C': 1}
    target_values = (0.0, 1.5, 0.8)

    results_by_method = {}

    # Ablation 1: -Sample Attn (AutoLambda without sample weighting)
    if 'ablation_no_sample_attn' in methods:
        print("\n=== Running Ablation: -Sample Attn (AutoLambda, no sample weights) ===")
        model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)

        # 关键：sample_weights=None
        autol = AutoLambda(model, device, train_tasks, pri_tasks, weight_init=0.1, sample_weights=None)
        autol.loss_fn = create_loss_function(model, sample_weights=None)

        history = run_autolambda_no_sample_weights(model, autol, datasets, num_epochs)

        mean_pred, var_pred = model.predict(test_data['x'])
        metrics = evaluate_metrics(mean_pred, var_pred, test_data['y'], target_values)
        final_weights = autol.get_normalized_weights().detach().cpu().numpy()
        results_by_method['ablation_no_sample_attn'] = {
            **metrics,
            **{f'weight_{task_id}': final_weights[i] for i, task_id in enumerate(train_tasks)}
        }

    # Ablation 2: -Both (No sample weighting + Equal task weights)
    if 'ablation_no_both' in methods:
        print("\n=== Running Ablation: -Both (no sample weights, equal task weights) ===")
        model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)

        run_baseline_no_both(model, datasets, num_epochs, list(train_tasks.keys()))

        mean_pred, var_pred = model.predict(test_data['x'])
        metrics = evaluate_metrics(mean_pred, var_pred, test_data['y'], target_values)
        results_by_method['ablation_no_both'] = metrics

    return results_by_method


def main():
    """主函数，运行消融实验。"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 500
    N_RUNS = 20

    # 配置要运行的消融实验
    methods_to_run = ['ablation_no_sample_attn', 'ablation_no_both']

    all_results = []

    print(f"Running {N_RUNS} ablation experiments...")
    for i in range(N_RUNS):
        print(f"\n{'='*60}")
        print(f"--- Running Ablation Experiment {i+1}/{N_RUNS} ---")
        print(f"{'='*60}")

        results = run_ablation_experiments(device, num_epochs, methods=methods_to_run)
        all_results.append(results)

        print(f"--- Finished Ablation Experiment {i+1}/{N_RUNS} ---")

    # 结果汇总
    print("\n" + "="*60)
    print("--- Ablation Experiment Results ---")
    print("="*60)

    # 构建所有运行的详细结果DataFrame
    all_runs_records = []
    for run_idx, run_result in enumerate(all_results, start=1):
        for method, metrics in run_result.items():
            record = {'run': run_idx, 'method': method}
            record.update(metrics)
            all_runs_records.append(record)

    # 保存详细结果
    all_runs_df = pd.DataFrame(all_runs_records)
    all_runs_df.to_csv('ablation_results_3task.csv', index=False)
    print(f"\nSaved detailed ablation results to 'ablation_results_3task.csv'")
    print(f"Total records: {len(all_runs_df)}")

    # 计算并显示平均值汇总
    method_results = {}
    for run_result in all_results:
        for method, metrics in run_result.items():
            method_results.setdefault(method, []).append(metrics)

    method_mean_records = {}
    for method, metrics_list in method_results.items():
        metrics_df = pd.DataFrame(metrics_list)
        method_mean_records[method] = metrics_df.mean(numeric_only=True)

    methods_summary_df = pd.DataFrame(method_mean_records).T
    methods_summary_df.index.name = 'method'

    # 只显示局部指标
    local_columns = [col for col in methods_summary_df.columns if col.startswith('local_')]
    methods_summary_local_df = methods_summary_df[local_columns]

    print("\nSummary of ablation experiments - Local Metrics Only (mean over runs):")
    print(methods_summary_local_df)

    # 保存汇总结果
    methods_summary_df.to_csv('ablation_summary_3task.csv')
    print("\nSaved summary to 'ablation_summary_3task.csv'")


if __name__ == '__main__':
    main()
