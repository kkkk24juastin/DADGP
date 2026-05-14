#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# DA-DGP实验交互式运行脚本
# ============================================================================

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACHIEVEMENTS_DIR="$SCRIPT_DIR/Achievements"
LOG_DIR="$SCRIPT_DIR/logs"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 可用方法列表
ALL_METHODS=(
    "da_dgp"
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

# 检查已有结果
check_existing_runs() {
    local start=$1
    local end=$2
    local existing=""
    local missing=""
    local partial=""

    for i in $(seq $start $end); do
        RUN_DIR="$ACHIEVEMENTS_DIR/$(printf '%02d' $i)"
        if [ -d "$RUN_DIR" ]; then
            if [ -f "$RUN_DIR/train_data.xlsx" ] \
                && [ -f "$RUN_DIR/val_data.xlsx" ] \
                && [ -f "$RUN_DIR/test_data.xlsx" ] \
                && compgen -G "$RUN_DIR/model_*.pt" > /dev/null; then
                existing="$existing $i"
            elif [ -f "$RUN_DIR/train_data.xlsx" ]; then
                partial="$partial $i"
            else
                missing="$missing $i"
            fi
        else
            missing="$missing $i"
        fi
    done

    echo "existing=$existing"
    echo "missing=$missing"
    echo "partial=$partial"
}

# 显示已有结果状态
show_existing_status() {
    local start=$1
    local end=$2

    print_separator
    echo "已有实验结果状态"
    print_separator

    local result=$(check_existing_runs $start $end)
    local existing=$(echo "$result" | grep "existing=" | cut -d'=' -f2)
    local missing=$(echo "$result" | grep "missing=" | cut -d'=' -f2)
    local partial=$(echo "$result" | grep "partial=" | cut -d'=' -f2)

    if [ -n "$existing" ]; then
        echo -e "${GREEN}[✓] 已完成:${NC} $existing"
    fi
    if [ -n "$partial" ]; then
        echo -e "${YELLOW}[!] 部分:${NC} $partial (有数据但缺少模型)"
    fi
    if [ -n "$missing" ]; then
        echo -e "${BLUE}[+] 待运行:${NC} $missing"
    fi

    if [ -z "$existing" ] && [ -z "$missing" ] && [ -z "$partial" ]; then
        echo -e "${CYAN}无已有结果${NC}"
    fi

    print_separator
}

# 交互式选择方法
select_methods() {
    print_separator
    echo "选择要运行的方法"
    print_separator

    echo "可用方法:"
    for i in "${!ALL_METHODS[@]}"; do
        printf "  %2d) %s\n" $((i+1)) "${ALL_METHODS[$i]}"
    done
    echo ""
    echo "选项:"
    echo "  a) 全选（所有方法）"
    echo "  d) 默认（不含消融实验）"
    echo "  n) 输入方法名（逗号分隔）"
    echo "  q) 取消"
    echo ""

    read -p "请选择 (输入数字序号，如: 1,2,3 或 a/d/n/q): " choice

    case $choice in
        a|A)
            SELECTED_METHODS="${ALL_METHODS[*]}"
            ;;
        d|D)
            SELECTED_METHODS="da_dgp baseline_equal baseline_pure_dgp baseline_dwa baseline_uw baseline_mgda baseline_indep_dgp baseline_indep_hetgp baseline_lmc_dgp"
            ;;
        q|Q)
            echo "已取消"
            exit 0
            ;;
        n|N)
            read -p "请输入方法名（逗号分隔，如: da_dgp,baseline_equal): " custom_methods
            SELECTED_METHODS=$(echo "$custom_methods" | tr ',' ' ')
            ;;
        *)
            # 解析数字选择
            SELECTED_METHODS=""
            for num in $(echo "$choice" | tr ',' ' '); do
                if [[ $num =~ ^[0-9]+$ ]] && [ $num -ge 1 ] && [ $num -le ${#ALL_METHODS[@]} ]; then
                    idx=$((num-1))
                    SELECTED_METHODS="$SELECTED_METHODS ${ALL_METHODS[$idx]}"
                fi
            done
            ;;
    esac

    # 验证选择
    if [ -z "$SELECTED_METHODS" ]; then
        echo -e "${RED}未选择任何方法${NC}"
        exit 1
    fi

    # 转换为逗号分隔格式
    METHODS_STR=$(echo "$SELECTED_METHODS" | tr ' ' ',' | sed 's/^,//;s/,$//')
    echo -e "${GREEN}已选择方法:${NC} $METHODS_STR"
}

# 主交互流程
main() {
    print_separator
    echo -e "${CYAN}DA-DGP 实验运行脚本${NC}"
    print_separator

    # 1. 选择运行范围
    echo ""
    read -p "起始运行编号 (默认 1): " START_RUN
    START_RUN=${START_RUN:-1}

    read -p "结束运行编号 (默认 20): " END_RUN
    END_RUN=${END_RUN:-20}

    echo ""
    echo -e "${CYAN}运行范围: $START_RUN - $END_RUN (共 $((END_RUN-START_RUN+1)) 次)${NC}"

    # 2. 显示已有结果
    mkdir -p "$ACHIEVEMENTS_DIR"
    show_existing_status $START_RUN $END_RUN

    # 3. 选择方法
    select_methods

    # 4. 选择运行模式
    echo ""
    print_separator
    echo "运行模式"
    print_separator
    echo "  1) 跳过已有结果，只运行缺失的"
    echo "  2) 强制覆盖所有已有结果"
    echo "  3) 取消运行"
    echo ""

    read -p "请选择 (1/2/3): " mode_choice

    case $mode_choice in
        1)
            RUN_MODE="--skip-all"
            echo -e "${GREEN}模式: 跳过已有，只运行缺失${NC}"
            ;;
        2)
            RUN_MODE="--overwrite"
            echo -e "${YELLOW}模式: 强制覆盖所有${NC}"
            ;;
        3)
            echo "已取消"
            exit 0
            ;;
        *)
            RUN_MODE="--skip-all"
            echo -e "${GREEN}模式: 跳过已有（默认）${NC}"
            ;;
    esac

    # 5. 其他参数
    echo ""
    read -p "基础随机种子 (默认 42): " BASE_SEED
    BASE_SEED=${BASE_SEED:-42}

    # 6. 确认并启动
    echo ""
    print_separator
    echo "实验配置确认"
    print_separator
    echo "  运行范围: $START_RUN - $END_RUN"
    echo "  方法列表: $METHODS_STR"
    echo "  运行模式: $RUN_MODE"
    echo "  基础种子: $BASE_SEED"
    print_separator
    echo ""

    read -p "确认启动实验? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "已取消"
        exit 0
    fi

    # 创建日志目录
    mkdir -p "$LOG_DIR"

    # 生成时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # 构建日志文件名
    METHOD_TAG=$(echo "$METHODS_STR" | tr ',' '_' | cut -c1-30)
    LOG_FILE="$LOG_DIR/experiment_${START_RUN}-${END_RUN}_${METHOD_TAG}_${TIMESTAMP}.log"

    # 构建Python命令
    PYTHON_CMD="python main_experiment.py --start-run $START_RUN --end-run $END_RUN --base-seed $BASE_SEED --methods $METHODS_STR $RUN_MODE"

    # 启动实验
    echo ""
    echo -e "${CYAN}启动实验...${NC}"
    echo "日志文件: $LOG_FILE"

    nohup $PYTHON_CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$LOG_DIR/last_pid.txt"

    sleep 1
    if ps -p $PID > /dev/null; then
        echo ""
        print_separator
        echo -e "${GREEN}[✓] 实验已成功启动!${NC}"
        print_separator
        echo ""
        echo "进程ID: $PID"
        echo ""
        echo "后续操作:"
        echo "  查看状态: ./monitor.sh"
        echo "  查看日志: tail -f $LOG_FILE"
        echo "  终止实验: ./monitor.sh stop"
        echo ""
        print_separator
    else
        echo -e "${RED}[✗] 进程启动失败，请检查日志${NC}"
        echo "日志内容:"
        head -30 "$LOG_FILE"
    fi
}

# 运行主流程
main
