# -*- coding: utf-8 -*-
"""
wireless 实验配置。

参考 3tasks 的组织方式，将训练超参数、任务定义、文件路径和输出文件名集中管理。
"""

from pathlib import Path

# ==========================================================================================
# 路径配置
# ==========================================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MOO_DIR = BASE_DIR / "moo"
MOO_DIR.mkdir(parents=True, exist_ok=True)

ACHIEVEMENTS_DIR = BASE_DIR / "Achievements"
ACHIEVEMENTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================================================
# 训练与模型配置
# ==========================================================================================

NUM_EPOCHS = 500

BATCH_SIZE = 128
LEARNING_RATE = 0.01
META_LEARNING_RATE = 0.01
LR_GAMMA = 0.99
META_LR_GAMMA = 0.999
MAX_GRAD_NORM = 1.0

NUM_HIDDEN_DGP_DIMS = 5
NUM_INDUCING_POINTS = 128
PREDICT_BATCH_SIZE = 50
LMC_NUM_LATENTS = 3
HETGP_NOISE_K = 20
HETGP_MIN_NOISE = 1e-4


# ==========================================================================================
# 任务与数据配置
# ==========================================================================================

INPUT_DIMENSIONS = 4
NUM_TASKS = 3

TRAIN_TASKS = {"local_A": 1, "local_B": 1, "local_C": 1}
PRI_TASKS = {"local_A": 1, "local_B": 1, "local_C": 1}
TASK_IDS = list(TRAIN_TASKS.keys())
TASK_TO_INDEX = {task_id: idx for idx, task_id in enumerate(TASK_IDS)}

TARGET_VALUES = (5.7131, 0.2532, 6.8794)
SIGMA_VALUES = (1.20, 0.10, 0.85)



WEIGHT_INIT = 0.1

TRAIN_SPLIT_NAME = "train"
VAL_SPLIT_NAME = "val"


# ==========================================================================================
# 方法配置
# ==========================================================================================

VALID_METHODS = [
    "dadgp",
    "baseline_equal",
    "baseline_pure_dgp",
    "baseline_dwa",
    "baseline_uw",
    "baseline_mgda",
    "baseline_indep_dgp",
    "baseline_indep_hetgp",
    "baseline_lmc_dgp",
    "ablation_no_sample_attn",
]

SAMPLE_ATTENTION_METHODS = (
    "dadgp",
    "baseline_equal",
    "baseline_dwa",
    "baseline_uw",
    "baseline_mgda",
)

DWA_TEMPERATURE = 2.0
DWA_RATIO_EPS = 1e-8
DWA_LOGIT_SCALE = 25.0
DWA_LOGIT_CLAMP = 4.0

MGDA_GRAD_EPS = 1e-12
MGDA_MAX_ITER = 250
MGDA_WEIGHT_TOL = 1e-6
MGDA_WEIGHT_EMA = 0.85

UW_LOG_VAR_LR_SCALE = 0.1
UW_LOG_VAR_CLAMP = 0.2

if len(TASK_IDS) != NUM_TASKS:
    raise ValueError(
        "TASK_IDS 的长度必须与 NUM_TASKS 一致，"
        f"当前长度: {len(TASK_IDS)}, NUM_TASKS: {NUM_TASKS}"
    )
if len(TARGET_VALUES) != NUM_TASKS:
    raise ValueError(
        "TARGET_VALUES 的长度必须与 NUM_TASKS 一致，"
        f"当前长度: {len(TARGET_VALUES)}, NUM_TASKS: {NUM_TASKS}"
    )
if len(SIGMA_VALUES) != NUM_TASKS:
    raise ValueError(
        "SIGMA_VALUES 的长度必须与 NUM_TASKS 一致，"
        f"当前长度: {len(SIGMA_VALUES)}, NUM_TASKS: {NUM_TASKS}"
    )
if any(sigma <= 0 for sigma in SIGMA_VALUES):
    raise ValueError(f"SIGMA_VALUES 必须全部大于 0，当前收到: {SIGMA_VALUES}")

_invalid_sample_attention_methods = [
    method for method in SAMPLE_ATTENTION_METHODS if method not in VALID_METHODS
]
if _invalid_sample_attention_methods:
    raise ValueError(
        "SAMPLE_ATTENTION_METHODS 中存在无效方法: "
        + ", ".join(_invalid_sample_attention_methods)
    )


# ==========================================================================================
# 多目标优化配置
# ==========================================================================================

MOO_LOWER_BOUND = [5.0, 50.0, 50.0, 0.0]
MOO_UPPER_BOUND = [30.0, 500.0, 500.0, 10.0]
MOO_TARGET_VALUES = (7.0, 0.2, 6.5)
DEFAULT_MOO_RUNS = 20
DEFAULT_MOO_POP_SIZE = 20
DEFAULT_MOO_N_GEN = 100
MOO_EVAL_BATCH_SIZE = DEFAULT_MOO_POP_SIZE
MOO_CPU_EVAL_BATCH_SIZE = 20
MOO_LIKELIHOOD_SAMPLES = 16


# ==========================================================================================
# 输出配置
# ==========================================================================================
