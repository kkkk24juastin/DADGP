# DADGP 论文修订计划（按当前仓库状态更新）

## Summary

修订目标：基于 `实验.md` 的 12 条审稿意见，系统整理 `mathexam/` 与 `wirelessexam/` 两个案例里已经完成的代码、图表和结果文件，优先完成论文正文与补充材料的重构，而不是默认重跑全部实验。

当前判断：

- `mathexam/` 的主实验、强 GP 基线、`Pure DGP`、消融、`sigma` 敏感性、网络深度敏感性、样本量敏感性、`20` 次统计汇总都已经落地。
- `wirelessexam/` 的 OFDM 主模型、强 GP 基线、`Pure DGP`、`BO` 对比、`20` 次 `MOO`、`16` 个噪声 realization 的 robust simulation、Pareto 指标、星座图和训练耗时分析都已经落地。
- 因此，当前 `Plan` 的重心应从“补做实验”切换为“提炼现有证据、补写方法解释、重构表图和组织审稿回复”。

边界说明：

- OFDM 目前已有的 `std/95% CI` 主要来自 `20` 次 `moo_run` 与 robust Pareto 指标汇总，不是 `20` 次独立代理重训。
- 若正文仍想强调 OFDM 代理预测误差本身的统计显著性，再考虑补做多次独立训练；默认计划先不重跑。

## 已有证据清单

### 数值案例 `mathexam/`

- 主实验 `20` 次结果已在 `mathexam/Achievements/01-20/`，并已有总表：
  - `mathexam/Achievements/local_metrics_summary_01-20.xlsx`
  - `mathexam/Achievements/local_metrics_mean_01-20.xlsx`
  - `mathexam/Achievements/weights_mean_01-20.xlsx`
- 已包含的主方法与基线共有 `10` 个：
  - `da_dgp`
  - `baseline_equal`
  - `baseline_pure_dgp`
  - `baseline_dwa`
  - `baseline_uw`
  - `baseline_mgda`
  - `baseline_indep_dgp`
  - `baseline_indep_hetgp`
  - `baseline_lmc_dgp`
  - `ablation_no_sample_attn`
- 三类补充实验与汇总结果已完整生成：
  - `mathexam/SupplementalModels/sigma_sensitivity/`
  - `mathexam/SupplementalModels/depth_complexity/`
  - `mathexam/SupplementalModels/sample_size/`
- 对应误差棒图已存在：
  - `mathexam/fig/sensitivity/sigma_sensitivity/`
  - `mathexam/fig/sensitivity/depth_complexity/`
  - `mathexam/fig/sensitivity/sample_size/`
- `mathexam/evaluate_models.py` 已经支持导出 `mean/std/95% CI`，这一项不是缺代码，而是缺论文呈现。

### OFDM 案例 `wirelessexam/`

- `wirelessexam/config.py` 与 `wirelessexam/models_dgp.py` 已支持：
  - `dadgp`
  - `baseline_equal`
  - `baseline_pure_dgp`
  - `baseline_dwa`
  - `baseline_uw`
  - `baseline_mgda`
  - `baseline_indep_dgp`
  - `baseline_indep_hetgp`
  - `baseline_lmc_dgp`
  - `ablation_no_sample_attn`
- 单次代理训练结果与 checkpoint 已在 `wirelessexam/model/`。
- `20` 次 `MOO` 候选结果已在 `wirelessexam/moo/`，并且已经包含 `BO-qParEGO`、`BO-qEHVI`、`BO-qNEHVI`。
- 基于现有噪声机制的 robust simulation 已在 `wirelessexam/result/robust_simulation_*.xlsx`，噪声重复使用 `SIMULATION_SEEDS = 42:57`。
- 汇总 Pareto 指标、`TOPSIS`、`CI` 与 boxplot 已存在：
  - `wirelessexam/result/robust_pareto_13methods_indicators.xlsx`
  - `wirelessexam/fig/robust_pareto_indicator_boxplots.pdf`
  - `wirelessexam/fig/paper_4obj_pareto/`
- 训练耗时和复杂度分析已存在：
  - `wirelessexam/timealy/training_time.csv`
  - `wirelessexam/timealy/time_complexity.csv`
  - `wirelessexam/timealy/summary.txt`

## 全文修订总策略

### 摘要与关键词

- 删除或弱化 “small sample sizes”“small sample conditions”“极小样本条件” 等过强表述，改成 “limited engineering samples” 或 “finite training samples”，重点回应意见 `#12`。
- 摘要结论不能再写成 “consistently outperforms all baselines” 这一类绝对化措辞，因为当前 OFDM 的 `BO-qNEHVI` 在个别 Pareto 指标上已接近或略优于 `DADGP`；摘要应改写为“在综合鲁棒 Pareto 质量、局部建模精度和工程效率权衡上表现最优或最均衡”。
- 摘要中补入四类修订后的核心信息：
  - `sigma` 与网络深度敏感性分析；
  - 强 GP 基线与消融结果；
  - OFDM 显式噪声与 transmitted variance；
  - Pareto/BO 对比与统计不确定性。
- 关键词原则上保持稳定，仅在需要时把 `Pareto trade-off` 或 `robust multi-objective optimization` 纳入关键词体系。

### Section 1: Introduction

- 第一段保留 RPD 与 OFDM 工程背景，但删除把当前设置描述为“极小样本”的表达。
- 文献综述部分需要更清楚地区分三条技术主线：
  - 非平稳 surrogate modeling；
  - 多任务/多目标冲突处理；
  - 鲁棒参数设计中的均值-方差与 Pareto trade-off。
- 引言末尾的贡献列表应按修订后论文内容重写，不再只强调“预测更准”，而是明确包括：
  - ROI-oriented sample attention；
  - meta-learning driven task balancing；
  - robust OFDM with explicit noise realizations；
  - strong GP baselines, ablation, sensitivity, Pareto and BO comparisons。
- “本文结构”这一段要与实际章节严格一致：
  - Section 2 理论背景；
  - Section 3 方法与优化；
  - Section 4 数值案例与 OFDM 工程验证；
  - Section 5 结论。

### Section 2: Theoretical Backgrounds

- 本节整体保持精炼，不把大量审稿回复直接塞进理论背景。
- 在 `2.3 Multi-output Deep Gaussian Process Framework` 末尾补一句过渡说明：
  - 正文实验默认采用 `2-layer` 多输出 DGP；
  - 更深结构仅作为 Section `4.1` 的复杂度敏感性分析对象。
- 适度补一层定位说明：本文方法与 `LMC`、独立 GP/DGP 的差异在于共享深层潜在表示与双注意力训练机制，便于后文强基线对比衔接。

### Section 3: DA-DGP Modeling and Optimization

- `3.1 Target-Oriented Sample-Level Attention`
  - 明确 `y^*` 的工程来源；
  - 增加 `y^*` 误设定鲁棒性的文字讨论；
  - 对 `sigma_k` 写清楚“当前为预设启发式、可学习化属于未来工作、现有敏感性结果支持其稳定性”，回应 `#1/#10`。
- `3.2 Meta-Learning Driven Task-Level Dynamic Attention`
  - 保留双层优化推导；
  - 补入训练复杂度的文字说明，并和普通共享 DGP、MGDA、Indep-DGP 做复杂度层面的相对定位，回应 `#5`。
- `3.3 Multi-Objective Robust Optimization Model`
  - 需要重新表述 OFDM 中的鲁棒目标，不再让读者把 surrogate posterior variance 和真实噪声传递方差混为一谈；
  - 写清楚：单点 `DGP` 后验方差是 surrogate uncertainty，而 OFDM robust simulation 中跨 realization 的经验方差才对应 transmitted variance；
  - 为 Section `4.2` 的 robust selection、Pareto analysis 和 BO 对比埋好符号接口。

### Section 4: Numerical and Engineering Validation

- 本节开头先统一补 Experimental Setup 层面的关键信息：
  - 默认网络结构为 `2-layer DGP`；
  - `num_hidden_dgp_dims = 5`；
  - `num_inducing_points = 128`；
  - 数值案例主实验为 `20` 次独立运行；
  - OFDM 当前为单次 surrogate 训练 + `20` 次 `MOO` + `16` 个噪声 realization 的 robust simulation。
- 本节所有主表原则上统一为 `mean ± std` 或 `mean [95% CI]` 风格，回应 `#11`。
- 本节总体叙事从“预测精度比较”升级为“局部建模能力 + 鲁棒优化能力 + Pareto trade-off + 训练代价”。

### Section 4.1: Numerical Example

- `Evaluation Metrics and Baselines` 需要重写：
  - 不能只保留 `EQUAL/UW/DWA/MGDA`；
  - 必须纳入 `Pure DGP`、`Indep-DGP`、`Indep-HetGP`、`LMC-DGP`；
  - 消融链要明确写成 `DA-DGP / No Sample Attn / Equal / Pure DGP`，回应 `#4/#6/#7`。
- `Experimental Setup` 要明确默认结构、运行次数与数据规模，直接回应 `#3/#11/#12`。
- Tables `1-3` 建议从原来的 `Max/Min/Median/Mean/Q1/Q3` 统计改成更贴近审稿需求的 `mean ± std` 或 `95% CI` 主表；原六数概括如需保留，可转入补充材料。
- 在 `4.1` 末尾或单独小段落中按顺序加入三类补充分析：
  - `sigma` 敏感性；
  - 深度/复杂度敏感性；
  - 样本量敏感性。
- 这一节的结论句要明确：`2L-D5` 是默认且最优的稳定配置，`400` 样本不是“极小样本”，而是“有限样本但足以支撑稳定建模”。

### Section 4.2: Robust Parameter Design for OFDM Systems

- 先重写问题定义，把 `x1-x4` 的物理含义和噪声来源写透：
  - `x1 = Ptx_dBm`
  - `x2 = d_m`
  - `x3 = tauRMS_ns`
  - `x4 = cfo_ppm`
  - bits / channel taps / AWGN 为噪声 realization。
- OFDM 的 `Experimental Setup` 里要写清楚：
  - surrogate 训练数据来自 `400` 个 LHS 样本；
  - robust simulation 使用 `42:57` 共 `16` 个种子；
  - `20` 次 `MOO` 用于 Pareto 指标统计。
- Tables `4-5` 与相关文字需要从“单点最优解/EVM”导向，升级为三层结果组织：
  - surrogate-based optimization result；
  - robust simulation result；
  - robust Pareto indicator result。
- 本节必须显式加入四块内容：
  - transmitted variance 与 robust loss 定义，回应 `#2`；
  - 训练耗时和时间复杂度，回应 `#5`；
  - Pareto frontier / trade-off / `TOPSIS` 分析，回应 `#8`；
  - `BO-qParEGO / BO-qEHVI / BO-qNEHVI` 数值对比，回应 `#9`。
- 写法上避免宣称 `DADGP` 在所有指标上都最优，更准确的叙述应是：
  - `DADGP` 在综合 `TOPSIS` 排名、Pareto 稳健性和理想点接近度上最均衡；
  - `BO-qNEHVI` 在部分 `HV` 指标上非常接近甚至略优。

### Section 5: Conclusion

- 结论不能只重复“DA-DGP 精度更高”，而要按修订后主线总结：
  - 局部建模有效；
  - 多任务冲突缓解有效；
  - OFDM robust Pareto 设计有效；
  - 代价是更高训练开销。
- 未来工作建议与审稿意见对应：
  - 从数据中学习 `sigma_k`；
  - 研究 `y^*` 不确定下的自适应 ROI；
  - 做 OFDM 多次独立 surrogate retraining 的统计验证；
  - 扩展到更一般的多响应 heteroscedastic robust design。

### Supplementary Materials / Appendices

- 主文聚焦核心结论，完整统计与扩展图表转入补充材料。
- 当前 Supplementary 已有 `Appendix A-D`，计划新增：
  - `Appendix E`: `sigma`、深度、样本量敏感性；
  - `Appendix F`: 强 GP 基线与消融完整表；
  - `Appendix G`: OFDM robust simulation、Pareto 指标、BO 对比、耗时分析。
- 正文中每个新增主结论都要明确指向对应附录，避免主文过重而又不给证据落点。

## Section 3 与方法层修订

- `#1/#10`：在样本注意力定义处补入 `y^*` 设定逻辑和误设定讨论。
  - 写法上强调：`y^*` 来自工程规格时可直接使用；若规格存在不确定性，可用邻近候选目标做敏感性分析。
  - 对 `sigma_k` 明确说明：当前采用预设启发式；从数据中学习 `sigma_k` 可作为未来工作；但现有数值敏感性结果显示该参数在较宽范围内是稳定的。
- `#5`：在双层优化小节补入训练复杂度表达，直接引用 `wirelessexam/timealy/time_complexity.csv` 的符号体系。
  - `DADGP` 与 `No Sample Attn` 的主阶仍由变分 DGP 的 `B*M^2 + M^3` 主导。
  - 相比普通共享 DGP，`DADGP` 的额外代价来自 unrolled validation backward 与 Hessian-vector 近似，常数因子约为 `4x`。
- 避免额外引入大量新公式，优先用文字解释“surrogate posterior variance” 与 “transmitted variance across noise realizations” 是两个不同层面的不确定性。

## Section 4.1 数值案例修订

### `#1/#10` `sigma` 敏感性

- 直接使用：
  - `mathexam/run_sigma_sensitivity.py`
  - `mathexam/SupplementalModels/sigma_sensitivity/local_metrics_summary_01-20.xlsx`
- 已有结果表明：
  - 平均 `quality_loss` 以 `sigma = 0.5` 最优，三任务均值约为 `0.0348`；
  - 平均 `RMSE` 以 `sigma = 0.3` 最优，`sigma = 0.5` 非常接近；
  - `sigma = 0.3-1.0` 整体稳定；
  - `sigma = 0.1` 明显变差，说明带宽过窄会造成过度局部化。
- 论文动作：
  - 主文给一段结论性文字；
  - 补充材料给完整误差棒图和 `mean ± std / 95% CI` 表。

### `#3` 网络深度与复杂度敏感性

- 直接使用：
  - `mathexam/run_depth_complexity_sensitivity.py`
  - `mathexam/SupplementalModels/depth_complexity/local_metrics_summary_01-20.xlsx`
- 已有结果表明：
  - `2L-D5` 是明显最佳配置，三任务平均 `quality_loss = 0.0347`；
  - `2L-D2` 明显欠拟合，平均 `quality_loss = 0.1860`；
  - `3L-D5`、`4L-D5`、`5L-D5` 快速恶化，平均 `quality_loss` 分别约为 `0.2821`、`0.6505`、`1.1925`；
  - 对当前 `4D/5D`、`400` 样本的场景，更深网络并不带来收益，反而增加不稳定和过拟合风险。
- 论文动作：
  - 在 Experimental Setup 明确默认就是 `2-layer DGP`；
  - 用敏感性结果论证“为什么不用更深的结构”。

### `#4/#6/#7` 强 GP 基线与消融

- 数值案例当前已经不是“缺基线”状态。
- `mathexam/Achievements/01-20/` 中已经同时包含：
  - `Indep-DGP`
  - `Indep-HetGP`
  - `LMC-DGP`
  - `Pure DGP`
  - `No Sample Attn`
- 也就是说，审稿意见要求的两类补充已经齐备：
  - 强 GP 专用基线；
  - 区分深层结构、任务注意力、样本注意力的消融链。
- 论文动作：
  - 更新 baseline 段落，不再把这些方法写成 future extension；
  - 明确四个关键变体：
    - `DA-DGP`
    - `No Sample Attn`
    - `Equal`
    - `Pure DGP`

### `#11` 标准差与置信区间

- `mathexam/evaluate_models.py` 已经输出：
  - `mean`
  - `std`
  - `95% CI`
  - `mean ± std`
  - `mean ± ci95`
- 主实验总表已经在：
  - `mathexam/Achievements/local_metrics_summary_01-20.xlsx`
- 论文动作：
  - 主文表格使用 `mean ± std` 或 `mean [95% CI]`；
  - 完整宽表放补充材料。

### `#12` 样本量敏感性与“小样本”表述修订

- 直接使用：
  - `mathexam/run_sample_size_sensitivity.py`
  - `mathexam/SupplementalModels/sample_size/local_metrics_summary_01-20.xlsx`
- 已有结果表明：
  - `n_train = 400` 时三任务平均 `quality_loss = 0.0347`；
  - `n_train = 200` 虽有改善，但平均 `quality_loss = 0.2851`，与 `400` 仍有显著差距；
  - `n_train = 100/50/20` 的平均 `quality_loss` 分别约为 `1.0331 / 1.1290 / 1.4397`；
  - 因此，当前稿件不能再把 `400` 样本表述为“小样本”，更适合写成“有限样本但并非极小样本”。
- 论文动作：
  - 删除“小样本条件”相关强表述；
  - 把样本量曲线作为“最低可用样本量”证据放补充材料。

### 数值案例主结果写法

- 主实验 `20` 次结果显示 `DA-DGP` 在三任务局部 `quality_loss` 上明显优于 `Equal` 与 `Pure DGP`：
  - `DA-DGP`: `0.0290 / 0.0278 / 0.0475`
  - `Equal`: `0.0600 / 0.0577 / 0.0830`
  - `Pure DGP`: `0.0598 / 0.0600 / 0.1019`
- 这一组数字非常适合放在主文核心表，用来支撑双注意力与联合建模的必要性。

## Section 4.2 OFDM 工程案例修订

### `#2` 显式噪声因素与传递方差

- 当前代码已经给出足够明确的物理定义：
  - `x1 = Ptx_dBm`
  - `x2 = d_m`
  - `x3 = tauRMS_ns`
  - `x4 = cfo_ppm`
- 对应实现见：
  - `wirelessexam/evaluate_ofdm_point.m`
  - `wirelessexam/ofdm_link_quality_ex.m`
- 现有噪声来源并非缺失，而是此前论文没有写清楚：
  - 随机比特流
  - Rayleigh channel taps
  - AWGN
  - 固定 `NF = 5 dB`
  - 通过 `rngseed` 控制 realization
- `wirelessexam/robust_select_ofdm_existing_noise.m` 已经对每个候选点在 `42:57` 共 `16` 个 realization 上重复仿真，并导出：
  - `robust_mean_*`
  - `robust_var_*`
- `wirelessexam/analyze_robust_ofdm_pareto.py` 已经按
  - `robust_loss = (robust_mean - target)^2 + robust_var`
 计算鲁棒损失并做 Pareto 选择。
- 论文动作：
  - 在问题定义里明确区分 design factors 与 noise realizations；
  - 把当前 Eq. (32) 周边文字改写为“单点后验方差是 surrogate uncertainty，robust variance 才是跨噪声 realization 的传递方差”；
  - 最终鲁棒参数设计结论应以后者为主。

### `#4/#6/#7` 强 GP 基线与消融

- OFDM 侧这部分也已经实现完毕，不需要再规划为新增开发。
- 现有代码、模型、候选结果和 robust simulation 已覆盖：
  - `Pure DGP`
  - `Indep-DGP`
  - `Indep-HetGP`
  - `LMC-DGP`
  - `No Sample Attn`
- 论文动作：
  - 把 OFDM baseline 段落从“通用深度学习基线”升级为“共享 DGP / 独立 DGP / 异方差 GP / LMC-DGP / BO”的完整对比框架；
  - 在主文只保留最关键方法，完整 `13` 方法表放补充材料。

### `#5` 时间复杂度与训练耗时

- 现有文件：
  - `wirelessexam/timealy/training_time.csv`
  - `wirelessexam/timealy/time_complexity.csv`
  - `wirelessexam/timealy/summary.txt`
- 已有 wall-clock 结果可直接写入论文：
  - `DADGP`: `397.79s`
  - `No Sample Attn`: `387.78s`
  - `MGDA`: `164.87s`
  - `Indep-DGP`: `163.40s`
  - `Equal`: `100.57s`
  - `Pure DGP`: `99.56s`
  - `Indep-HetGP`: `10.51s`
- 论文动作：
  - 正文给一张耗时表和一句复杂度总结；
  - 重点说明 `DADGP` 的计算开销真实存在，但换来的是更稳健的 Pareto 综合表现，而不是只强调精度。

### `#8` Pareto 前沿与 trade-off 分析

- 现有结果已经完全足够支撑这一条意见：
  - `wirelessexam/result/robust_pareto_13methods_indicators.xlsx`
  - `wirelessexam/fig/paper_4obj_pareto/`
  - `wirelessexam/fig/robust_pareto_indicator_boxplots.pdf`
- 当前 robust `4-objective` 指标显示：
  - `DADGP` 的 `TOPSIS` 排名为第 `1`
  - `Ci = 0.9322`
  - `HV(norm) = 0.3326`
  - `HV Ratio = 0.7764`
- 论文动作：
  - 把 OFDM 章节叙事从“谁预测更准”转成“谁给出的 Pareto 集更优、更稳、更贴近理想点”；
  - 明确讨论以下 trade-off：
    - throughput vs BER
    - throughput vs PAPR
    - throughput / BER / PAPR 与 energy efficiency 的四目标冲突

### `#9` 与 Bayesian Optimization 的数值比较

- `wirelessexam/bo_compare.py` 已经实现：
  - `BO-qParEGO`
  - `BO-qEHVI`
  - `BO-qNEHVI`
- 对应 `moo/`、`result/`、Pareto 指标和星座图都已生成。
- 当前汇总结果说明：
  - `BO-qNEHVI` 的 `HV(norm) = 0.3330`，略高于 `DADGP`
  - 但 `DADGP` 的 `TOPSIS` 综合排名仍是第 `1`
  - `BO-qEHVI` 排名中上
  - `BO-qParEGO` 明显较弱
- 论文动作：
  - 不要把 BO 对比写成“全面碾压”；
  - 更准确的叙述是：`DADGP` 的 one-shot surrogate 在综合 Pareto 稳健性和理想点接近度上最均衡，而 `BO-qNEHVI` 在部分指标上非常接近甚至略优。

### `#11` 标准差与置信区间

- OFDM 现有 robust Pareto 工作簿已经包含：
  - `indicators`
  - `topsis`
  - `indicators_with_ci`
  - `indicator_stats_long`
  - `per_run_indicators`
- 因此，Pareto 指标层面的 `std/95% CI` 已经可直接用于论文和补充材料。
- 但要写清楚：
  - 这些区间来自 `20` 次 `moo_run` 的候选集与 robust 指标统计；
  - 不是 `20` 次独立 surrogate retraining 的预测误差统计。

## 补充材料重构建议

- **附录 E：Sensitivity Analyses**
  - 放 `sigma`、深度、样本量三组误差棒图与完整统计表。
  - 主要来源：
    - `mathexam/SupplementalModels/*/local_metrics_summary_01-20.xlsx`
    - `mathexam/fig/sensitivity/*`

- **附录 F：Additional Baselines and Ablation**
  - 放数值案例 `10` 方法完整统计表。
  - 明确 `DA-DGP / No Sample Attn / Equal / Pure DGP` 四段消融链。
  - 必要时补充 OFDM 星座图：
    - `wirelessexam/fig/ofdm_constellation/`

- **附录 G：Robust OFDM and Pareto Analysis**
  - 放噪声 realizations 说明、robust variance 定义、`13` 方法 Pareto 指标、`TOPSIS`、boxplot 和 `BO` 对比。
  - 主要来源：
    - `wirelessexam/result/robust_pareto_13methods_indicators.xlsx`
    - `wirelessexam/fig/paper_4obj_pareto/`
    - `wirelessexam/fig/robust_pareto_indicator_boxplots.pdf`
    - `wirelessexam/timealy/*`

## 审稿意见到证据的映射

- `#1`：`mathexam` 的 `sigma` 敏感性结果 + 方法部分对 `y^*` 误设定的文字讨论。
- `#2`：`wirelessexam/ofdm_link_quality_ex.m`、`robust_select_ofdm_existing_noise.m`、`analyze_robust_ofdm_pareto.py`。
- `#3`：`mathexam` 深度敏感性结果 + 默认两层结构说明。
- `#4`：`Indep-DGP`、`Indep-HetGP` 已在两个案例实现并有结果。
- `#5`：`wirelessexam/timealy/training_time.csv` 与 `time_complexity.csv`。
- `#6`：`LMC-DGP`、`Indep-HetGP`、`Indep-DGP` 已有。
- `#7`：`Pure DGP`、`No Sample Attn`、`Equal`、`DA-DGP` 四段消融链已齐全。
- `#8`：`wirelessexam/result/robust_pareto_13methods_indicators.xlsx` + `paper_4obj_pareto` 图组。
- `#9`：`wirelessexam/bo_compare.py` 与对应 `moo/result` 文件。
- `#10`：`mathexam` 的 `sigma` 敏感性汇总表。
- `#11`：`mathexam/evaluate_models.py` 输出统计表 + `wirelessexam` 的 Pareto `CI` 工作簿。
- `#12`：`mathexam` 样本量敏感性结果 + 全文措辞收缩。

## 执行顺序

1. 先改正文总叙事，删除“小样本”强表述，并明确默认网络结构与 OFDM 噪声定义。
2. 再重写 Section 4.1，把数值案例组织成“主结果 + 三类敏感性 + 强 GP 基线 + 消融”。
3. 接着重写 Section 4.2，把 OFDM 组织成“鲁棒建模定义 + 训练复杂度 + robust Pareto + BO 对比”。
4. 然后整理补充材料，把完整统计表、误差棒图、`13` 方法 Pareto 指标和耗时表移入附录。
5. 最后逐条核对 `#1-#12` 是否都在正文或附录中有明确证据落点。

## 当前不建议默认执行的动作

- 不建议一上来重跑全部实验。
- 不建议先扩充新模型或新指标。
- 只有在以下情况下，才考虑新增计算：
  - 期刊明确要求 OFDM 代理预测误差也给出多次独立训练的 `CI`；
  - 现有图表版面无法支持主文叙事，需要额外导出更适合排版的图。
