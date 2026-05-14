# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是 **DA-DGP（双注意力深度高斯过程）** 多任务学习研究项目，在 5D 数学函数验证案例上对比 DA-DGP 与多种基线方法。核心创新是双层优化（bilevel optimization）：内层优化模型参数，外层通过验证集元目标动态学习任务权重，同时使用样本级高斯核加权实现局部拟合。

## 常用命令

### 运行实验
```bash
# 使用交互式 Shell 脚本启动（推荐，后台运行 + 日志）
./run_experiment.sh

# 直接用 Python 运行全部实验（1-20 run）
python main_experiment.py

# 运行指定范围的实验
python main_experiment.py --start-run 1 --end-run 5

# 只运行特定方法（逗号分隔）
python main_experiment.py --methods da_dgp,baseline_equal

# 跳过已有结果 / 强制覆盖
python main_experiment.py --skip-all
python main_experiment.py --overwrite
```

可用方法名: `da_dgp`, `baseline_equal`, `baseline_pure_dgp`, `baseline_dwa`, `baseline_uw`, `baseline_mgda`, `baseline_indep_dgp`, `baseline_indep_hetgp`, `baseline_lmc_dgp`, `ablation_no_sample_attn`

### 评估模型
```bash
# 评估单次运行
python evaluate_models.py --run-id 5

# 评估多次运行并汇总（均值表）
python evaluate_models.py --start-run 1 --end-run 20

# 只评估特定方法
python evaluate_models.py --methods da_dgp
```

### 监控实验状态
```bash
./monitor.sh status    # 检查运行状态
./monitor.sh log       # 查看实时日志
./monitor.sh results   # 查看已有结果概览
./monitor.sh stop      # 终止实验
./monitor.sh           # 交互式菜单
```

### 绘图
```bash
python plot.py           # 箱线图（fig/box/）
python plot_metrics.py   # 统计折线图（fig/metrics/）
python plotweight.py     # 权重演化图
```

## 架构概览

### 核心模块

| 文件 | 职责 |
|------|------|
| `config.py` | 所有可配置参数统一管理（路径、超参、模型结构、数据生成） |
| `models_dgp.py` | DGP 模型定义：`WeightedVariationalELBO`（样本级加权 ELBO）、`DGPHiddenLayer`、`MultitaskDeepGP`（两层 DGP + 多任务似然） |
| `common.py` | `BackupStyleLoss`（每个任务损失 = 完整 ELBO，KL 已包含在内）、`IndexedTensorDataset` |
| `algo_da_dgp.py` | **DA-DGP 核心算法**：`DADGP` 类实现双层优化（虚拟 Adam 步骤 + 有限差分 Hessian 近似）、`run_training_loop` |
| `algo_equal.py` | 等权重基线（1/n 固定权重） |
| `algo_dwa.py` | DWA 基线（基于损失变化率的动态权重，温度 T=2.0） |
| `algo_uw.py` | Uncertainty Weighting 基线（可学习 log-variance 自动权衡） |
| `algo_mgda.py` | MGDA 基线（Frank-Wolfe 求解 Pareto 最优梯度方向） |
| `algo_ablation_no_sample_attn.py` | 消融实验：DA-DGP 去掉样本加权（sample_weights=None） |
| `algo_supplemental_baselines.py` | 补充基线训练循环：Indep-DGP 与 Indep-HetGP |
| `data_generation.py` | 5D 三任务测试函数 + 拉丁超立方采样 + 样本高斯核权重计算 |
| `metrics.py` | 局部区域指标：RMSE、NLPD、质量损失（基于 LOCAL_THRESHOLD=0.3 定义"局部"） |
| `evaluate_models.py` | 加载已保存模型，在测试集上评估并汇总结果 |

### 数据流

1. **生成数据**: `data_generation.py` 在 [-1,1]^5 上用 LHS 采样，通过 `three_task_function` 生成 (y1,y2,y3)，并计算高斯核样本权重
2. **训练**: `main_experiment.py` 为每种方法创建独立模型实例，调用对应训练循环。每个 run 有独立随机种子（base_seed + run_id）
3. **保存**:  `Achievements/{run_id:02d}/` 下保存 `train_data.xlsx / val_data.xlsx / test_data.xlsx`、`model_*.pt`、DA-DGP 额外保存 `weight_history_da_dgp.xlsx`
4. **评估**: `evaluate_models.py` 加载模型和测试数据，计算局部 RMSE/NLPD/QL，按 run 汇总生成 `Achievements/local_metrics_mean_*.xlsx`

### 损失函数语义

项目使用"备份版语义"（`BackupStyleLoss`）：每个任务损失自身已是完整 ELBO（包含 KL 散度），**不再**额外添加共享 KL 项。`split_sizes` 用于 ELBO 中的 `num_data` 参数。

### 关键机制

- **样本注意力**：通过 `compute_sample_weights()` 计算高斯加权，目标值附近的样本获得更高权重，控制参数为 σ
- **任务注意力**：`DADGP.meta_weights` 通过 softmax 归一化，在验证集元目标上通过展开梯度反向传播（unrolled gradient）更新
- **Hessian 近似**：`compute_hessian()` 使用中心有限差分（eps=0.01/||d_model||）近似 Hessian 向量积，避免显式计算二阶导数
