# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OFDM 链路质量代理模型的多任务深度高斯过程（DGP）实验框架。核心方法是 DADGP（Dual-Attention Deep Gaussian Process），通过双层优化动态学习任务权重，并对比多种多任务学习基线方法。

研究场景：4 维设计变量 → 3 个任务目标（local_A / local_B / local_C），用于 OFDM 无线通信链路质量预测。

## Common Commands

```bash
# 交互式运行训练实验（推荐，有菜单选择方法）
./run_experiment.sh

# 直接运行全部方法训练
python main_experiment.py

# 只运行指定方法
python main_experiment.py --methods dadgp,baseline_equal,baseline_dwa

# 指定随机种子
python main_experiment.py --base-seed 123

# 多目标优化（NSGA-II）
python moo_nsga2.py

# 分析 Pareto 前沿
python analyze_real_simulation_pareto.py
python analyze_robust_ofdm_pareto.py

# 监控实验状态
./monitor.sh
./monitor.sh stop
```

## Architecture

### 核心模块

- [config.py](config.py) — 全局配置（超参数、任务定义、路径、方法列表）。修改训练参数或任务定义在此文件操作。
- [models_dgp.py](models_dgp.py) — 模型定义：`MultitaskDeepGP`（主模型）、`IndependentDeepGP`、`IndependentHeteroscedasticGP`、`LMCDeepGP`。均基于 GPyTorch 实现。
- [common.py](common.py) — `BackupStyleLoss` 损失函数（每个任务一个完整 ELBO）、`set_seed`、`IndexedTensorDataset`。
- [data_loading.py](data_loading.py) — 数据加载（支持 csv/xlsx）、归一化（X: min-max, Y: 标准化）、高斯样本权重计算。

### 算法实现

- [algo_dadgp.py](algo_dadgp.py) — **DADGP 核心**：双层优化（virtual step + unrolled backward + Hessian 近似），动态学习 meta_weights。
- [algo_equal.py](algo_equal.py) — Equal Weight 基线（等权重）。
- [algo_dwa.py](algo_dwa.py) — DWA（Dynamic Weight Average）基线。
- [algo_uw.py](algo_uw.py) — Uncertainty Weighting 基线。
- [algo_mgda.py](algo_mgda.py) — MGDA（Multiple Gradient Descent Algorithm）基线。
- [algo_supplemental_baselines.py](algo_supplemental_baselines.py) — Indep-DGP 和 Indep-HetGP 基线的训练循环。
- [algo_ablation_no_sample_attn.py](algo_ablation_no_sample_attn.py) — 消融实验：去掉 Sample Attention。

### 实验编排

- [main_experiment.py](main_experiment.py) — 主入口：编排所有方法的训练，保存模型 checkpoint 和训练曲线。
- [experiment_utils.py](experiment_utils.py) — `build_model`、`save_model_checkpoint`、`load_model_checkpoint` 等工具。
- [moo_nsga2.py](moo_nsga2.py) — NSGA-II 多目标优化，使用已训练模型进行 Pareto 搜索。

### 分析与可视化

- [analyze_real_simulation_pareto.py](analyze_real_simulation_pareto.py) — 真实仿真 Pareto 前沿可视化和指标计算（HV/IGD/GD）。
- [analyze_robust_ofdm_pareto.py](analyze_robust_ofdm_pareto.py) — 鲁棒 OFDM Pareto 分析。
- [plot_result_boxplots.py](plot_result_boxplots.py) — 箱线图绘制。
- [analyze_timealy.py](analyze_timealy.py) — 时间复杂度分析。

### MATLAB 辅助

- [generate_ofdm_train_val_dataset.m](generate_ofdm_train_val_dataset.m) — 生成 OFDM 训练/验证数据集。
- [generate_moo_real_simulation_results.m](generate_moo_real_simulation_results.m) — 生成 MOO 真实仿真结果。
- [ofdm_link_quality_ex.m](ofdm_link_quality_ex.m) — OFDM 链路质量评估。
- [robust_select_ofdm_existing_noise.m](robust_select_ofdm_existing_noise.m) — 鲁棒选择含噪声 OFDM。

### 目录结构

- `data/` — 训练数据（train.xlsx, val.xlsx）
- `model/` — 训练好的模型 checkpoint (.pt) 和训练曲线图
- `moo/` — 多目标优化结果（每方法一个 .xlsx）
- `fig/` — 分析图表
- `logs/` — 实验日志

## Key Design Decisions

- **样本注意力（Sample Attention）**：DADGP 及部分基线使用基于目标值的高斯样本权重，通过 `TARGET_VALUES` 和 `SIGMA_VALUES` 配置。`SAMPLE_ATTENTION_METHODS` 定义了哪些方法启用此机制。
- **Checkpoint 格式**：模型以 state_dict + 元数据方式保存，`experiment_utils.py` 的 `load_model_checkpoint` 可根据 `model_type` 自动重建模型。
- **归一化**：默认启用，X 使用 min-max，Y 使用标准化。归一化参数保存在 checkpoint 中供推理时反归一化。
- **随机种子**：每个方法训练前重置种子（`reset_method_seed`），确保公平对比。

## Dependencies

- PyTorch、GPyTorch（核心 ML 框架）
- pandas（数据加载）
- matplotlib（绘图）
- tqdm（训练进度条）
- pymoo（NSGA-II 多目标优化，moo_nsga2.py 使用）

部分约定
配置内嵌：不要使用命令行参数来设置配置，而是将配置内嵌，可以直接修改
拒绝冒烟测试：不要使用冒烟测试，也不要静态编译文件

