# Dual-Attention Deep Gaussian Processes for Multi-Objective Robust Parameter Design

本仓库整理了论文 **Dual-Attention Deep Gaussian Processes for Multi-Objective Robust Parameter Design** 的实验代码。核心方法是 DADGP/DA-DGP: 通过任务级注意力动态学习多任务权重，并通过样本级高斯权重强化目标区域附近的局部拟合能力。

仓库包含两套实验:

- `mathexam/`: 5D 三任务数学函数验证实验，用于方法对比、消融和敏感性分析。
- `wirelessexam/`: 4D OFDM 链路质量工程案例，用于鲁棒参数设计、多目标 Pareto 搜索和真实仿真分析。

## 方法概览

主要比较方法包括:

- `dadgp` / `da_dgp`: Dual-Attention Deep Gaussian Process。
- `baseline_equal`: 等权重多任务 DGP。
- `baseline_pure_dgp`: 不使用样本注意力的纯多任务 DGP。
- `baseline_dwa`: Dynamic Weight Average。
- `baseline_uw`: Uncertainty Weighting。
- `baseline_mgda`: Multi-Gradient Descent Algorithm。
- `baseline_indep_dgp`: 独立任务 DGP。
- `baseline_indep_hetgp`: 独立异方差 GP/DGP 基线。
- `baseline_lmc_dgp`: LMC-DGP 基线。
- `ablation_no_sample_attn`: 去除样本注意力的消融实验。

## 目录结构

```text
.
├── mathexam/                 # 5D synthetic benchmark
│   ├── main_experiment.py    # 多次训练实验入口
│   ├── evaluate_models.py    # 指标评估与汇总
│   ├── data_generation.py    # 数学函数、LHS 采样、样本权重
│   ├── models_dgp.py         # DGP/GP 模型定义
│   ├── algo_*.py             # DADGP 与各类基线训练逻辑
│   ├── run_*sensitivity.py   # 样本量、sigma、深度复杂度敏感性实验
│   └── plot*.py              # 绘图脚本
└── wirelessexam/             # OFDM robust parameter design case
    ├── main_experiment.py    # OFDM 代理模型训练入口
    ├── data_loading.py       # train/val 表格数据读取与归一化
    ├── models_dgp.py         # DGP/GP 模型定义
    ├── experiment_utils.py   # checkpoint 保存/加载与模型构建
    ├── moo_nsga2.py          # 基于 pymoo 的 NSGA-II Pareto 搜索
    ├── bo_compare.py         # BoTorch BO 对比候选生成
    ├── analyze_*.py          # Pareto/鲁棒性/耗时分析
    └── *.m                   # OFDM 数据生成与真实仿真辅助脚本
```

## 环境准备

建议使用 Python 3.10 或更高版本。GPU 不是必需的，代码会在 CUDA 可用时自动使用 `cuda:0`，否则回退到 CPU。

核心 Python 依赖:

```bash
pip install torch gpytorch numpy pandas matplotlib seaborn tqdm pyDOE2 openpyxl pymoo botorch
```

说明:

- `torch` 与 `gpytorch` 是 DGP 模型训练的核心依赖。
- `pymoo` 用于 `wirelessexam/moo_nsga2.py` 中的 NSGA-II 多目标优化。
- `botorch` 用于 `wirelessexam/bo_compare.py` 中的 BO 对比实验。
- MATLAB 脚本用于生成 OFDM 数据和真实仿真结果；`bo_compare.py` 若要直接调用 MATLAB 仿真，还需要 MATLAB Engine for Python。

## 快速开始: 数学函数实验

```bash
cd mathexam

# 运行一次轻量实验，只训练 DA-DGP 与等权重基线
python main_experiment.py --start-run 1 --end-run 1 --methods da_dgp,baseline_equal --skip-all

# 评估第 1 次实验
python evaluate_models.py --run-id 1 --methods da_dgp,baseline_equal
```

完整实验默认运行 `config.py` 中的 `N_RUNS=20`，每次实验会重新生成训练、验证和测试数据，并保存到 `mathexam/Achievements/{run_id}/`。

常用命令:

```bash
# 交互式后台运行，带日志与方法选择
./run_experiment.sh

# 运行指定范围
python main_experiment.py --start-run 1 --end-run 5 --methods da_dgp,baseline_dwa

# 汇总多次实验结果
python evaluate_models.py --start-run 1 --end-run 20

# 绘图
python plot.py
python plot_metrics.py
python plotweight.py
```

## 快速开始: OFDM 工程案例

`wirelessexam/main_experiment.py` 默认读取:

- `wirelessexam/data/train.xlsx`
- `wirelessexam/data/val.xlsx`

如果数据文件不存在，可在 MATLAB 中进入 `wirelessexam/` 后运行:

```matlab
generate_ofdm_train_val_dataset
```

训练代理模型:

```bash
cd wirelessexam

# 训练全部方法
python main_experiment.py

# 或只训练指定方法
python main_experiment.py --methods dadgp,baseline_equal,baseline_dwa --base-seed 42
```

训练输出位于 `wirelessexam/model/`，包括各方法 checkpoint、训练曲线和 DADGP 权重轨迹。

执行 Pareto 搜索与分析:

```bash
# 默认按 moo_nsga2.py 内 SELECTED_METHODS 配置执行
python moo_nsga2.py

# 基于 MOO/真实仿真结果做 Pareto 分析
python analyze_real_simulation_pareto.py
python analyze_robust_ofdm_pareto.py

# 鲁棒 Pareto 指标汇总与绘图
python calc_robust_pareto_14methods.py
python plot_robust_pareto_indicator_boxplots.py
```

若要运行 BO 对比:

```bash
python bo_compare.py
```

该脚本默认从 `wirelessexam/data/train.xlsx` 初始化，并将 BO 候选结果写入 `wirelessexam/moo/`。

## 输出文件

主要生成目录如下，已在 `.gitignore` 中排除，避免将大体积实验产物提交到开源仓库:

- `mathexam/Achievements/`: 数学函数实验的数据、模型与评估表。
- `mathexam/SupplementalModels/`: 敏感性分析模型和结果。
- `mathexam/fig/`: 数学函数实验图表。
- `wirelessexam/model/`: OFDM 训练 checkpoint、训练曲线和权重轨迹。
- `wirelessexam/moo/`: NSGA-II 与 BO 候选结果 workbook。
- `wirelessexam/result/`: 真实仿真和鲁棒 Pareto 指标结果。
- `wirelessexam/fig/`: OFDM 案例图表。
- `wirelessexam/logs/`, `wirelessexam/timealy/`: 日志和耗时分析结果。

`wirelessexam/data/train.xlsx` 与 `wirelessexam/data/val.xlsx` 是 OFDM 训练入口需要的输入数据，小规模版本可以随代码发布；如需重新生成，请使用 MATLAB 脚本。

## 配置说明

实验配置集中在各子目录的 `config.py`:

- `mathexam/config.py`: 训练轮数、样本量、任务目标、sigma、DGP 结构和绘图参数。
- `wirelessexam/config.py`: OFDM 输入维度、目标值、训练方法列表、MOO 搜索边界和 NSGA-II 参数。

多数脚本采用“修改配置常量后运行”的方式。例如 `wirelessexam/moo_nsga2.py` 中的 `SELECTED_METHODS` 控制待优化方法，`bo_compare.py` 中的 `BO_METHODS` 和 `BO_BUDGET` 控制 BO 对比设置。

## 复现实验建议

1. 先确认 Python 依赖和可选 MATLAB 环境。
2. 对 `mathexam`，从小范围 run 开始验证流程，再扩大到完整 `N_RUNS=20`。
3. 对 `wirelessexam`，先生成或保留 `data/train.xlsx`、`data/val.xlsx`，再训练代理模型。
4. 完成 OFDM 训练后，再运行 `moo_nsga2.py`、真实仿真脚本和 Pareto 分析脚本。
5. 大体积结果建议通过 release、网盘或归档数据集单独发布，不建议直接提交到 Git。

## 引用

论文正式发表后，可在此补充 BibTeX。当前推荐在使用本仓库时引用论文题目:

```bibtex
@article{dadgp_robust_parameter_design,
  title = {Dual-Attention Deep Gaussian Processes for Multi-Objective Robust Parameter Design},
  author = {TBD},
  journal = {TBD},
  year = {TBD}
}
```
