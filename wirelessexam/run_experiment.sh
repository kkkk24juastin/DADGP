#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# wireless 实验交互式运行脚本
# 基于上级 3tasks/run_experiment.sh 的交互风格，适配当前 unified main_experiment.py
# 只负责训练并保存模型，不自动执行 MOO
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$LOG_DIR/last_pid.txt"
CONFIG_EPOCHS=$(python -c "from config import NUM_EPOCHS; print(NUM_EPOCHS)")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

ALL_METHODS=(
    "dadgp"
    "baseline_equal"
    "baseline_pure_dgp"
    "baseline_dwa"
    "baseline_uw"
    "baseline_mgda"
    "baseline_indep_dgp"
    "baseline_indep_hetgp"
    "baseline_lmc_dgp"
    "ablation_no_sample_attn"
)

print_separator() {
    echo "============================================================"
}

find_split_file() {
    local split_name=$1
    local candidate
    for candidate in \
        "$DATA_DIR/${split_name}.csv" \
        "$DATA_DIR/${split_name}.xlsx" \
        "$DATA_DIR/${split_name}_data.xlsx"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

select_methods() {
    print_separator
    echo "选择要运行的方法"
    print_separator

    echo "可用方法:"
    for i in "${!ALL_METHODS[@]}"; do
        printf "  %2d) %s\n" $((i + 1)) "${ALL_METHODS[$i]}"
    done
    echo ""
    echo "选项:"
    echo "  a) 全选（所有方法）"
    echo "  d) 默认（不含消融）"
    echo "  n) 输入方法名（逗号分隔）"
    echo "  q) 取消"
    echo ""

    read -p "请选择 (输入数字序号，如: 1,2,3 或 a/d/n/q): " choice

    case $choice in
        a|A)
            SELECTED_METHODS="${ALL_METHODS[*]}"
            ;;
        d|D)
            SELECTED_METHODS="dadgp baseline_equal baseline_pure_dgp baseline_dwa baseline_uw baseline_mgda baseline_indep_dgp baseline_indep_hetgp baseline_lmc_dgp"
            ;;
        q|Q)
            echo "已取消"
            exit 0
            ;;
        n|N)
            read -p "请输入方法名（逗号分隔，如: dadgp,baseline_equal): " custom_methods
            SELECTED_METHODS=$(echo "$custom_methods" | tr ',' ' ')
            ;;
        *)
            SELECTED_METHODS=""
            for num in $(echo "$choice" | tr ',' ' '); do
                if [[ $num =~ ^[0-9]+$ ]] && [ "$num" -ge 1 ] && [ "$num" -le ${#ALL_METHODS[@]} ]; then
                    idx=$((num - 1))
                    SELECTED_METHODS="$SELECTED_METHODS ${ALL_METHODS[$idx]}"
                fi
            done
            ;;
    esac

    if [ -z "$SELECTED_METHODS" ]; then
        echo -e "${RED}未选择任何方法${NC}"
        exit 1
    fi

    METHODS_STR=$(echo "$SELECTED_METHODS" | tr ' ' ',' | sed 's/^,//;s/,$//')
    echo -e "${GREEN}已选择方法:${NC} $METHODS_STR"
}

main() {
    print_separator
    echo -e "${CYAN}wireless 实验运行脚本${NC}"
    print_separator

    if [ ! -d "$DATA_DIR" ]; then
        echo -e "${RED}未找到数据目录: $DATA_DIR${NC}"
        exit 1
    fi

    TRAIN_FILE=$(find_split_file train || true)
    VAL_FILE=$(find_split_file val || true)

    if [ -z "$TRAIN_FILE" ] || [ -z "$VAL_FILE" ]; then
        echo -e "${RED}data 目录下缺少单次实验所需的数据文件${NC}"
        echo "期望文件之一:"
        echo "  - data/train.xlsx 或 data/train_data.xlsx 或 data/train.csv"
        echo "  - data/val.xlsx 或 data/val_data.xlsx 或 data/val.csv"
        exit 1
    fi

    echo ""
    echo "单次实验数据文件:"
    echo "  训练集: ${TRAIN_FILE#$SCRIPT_DIR/}"
    echo "  验证集: ${VAL_FILE#$SCRIPT_DIR/}"
    echo ""

    select_methods

    read -p "基础随机种子 (默认 42): " BASE_SEED
    BASE_SEED=${BASE_SEED:-42}

    echo ""
    print_separator
    echo "实验配置确认"
    print_separator
    echo "  训练数据: ${TRAIN_FILE#$SCRIPT_DIR/}"
    echo "  验证数据: ${VAL_FILE#$SCRIPT_DIR/}"
    echo "  方法列表: $METHODS_STR"
    echo "  基础种子: $BASE_SEED"
    echo "  训练轮数: $CONFIG_EPOCHS (来自 config.py)"
    print_separator
    echo ""

    read -p "确认启动实验? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 0
    fi

    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    METHOD_TAG=$(echo "$METHODS_STR" | tr ',' '_' | cut -c1-40)
    LOG_FILE="$LOG_DIR/experiment_single_${METHOD_TAG}_${TIMESTAMP}.log"

    PYTHON_CMD=(
        python main_experiment.py
        --methods "$METHODS_STR"
        --base-seed "$BASE_SEED"
    )

    echo ""
    echo -e "${CYAN}启动实验...${NC}"
    echo "日志文件: $LOG_FILE"
    echo "执行命令: ${PYTHON_CMD[*]}"

    nohup "${PYTHON_CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    sleep 1
    if ps -p "$PID" > /dev/null; then
        echo ""
        print_separator
        echo -e "${GREEN}[✓] 实验已成功启动!${NC}"
        print_separator
        echo "进程ID: $PID"
        echo "查看状态: ./monitor.sh"
        echo "查看日志: tail -f $LOG_FILE"
        echo "终止实验: ./monitor.sh stop"
        print_separator
    else
        echo -e "${RED}[✗] 进程启动失败，请检查日志${NC}"
        head -30 "$LOG_FILE"
        exit 1
    fi
}

main
