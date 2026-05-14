#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# DA-DGP实验交互式监控脚本
# ============================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$BASE_DIR/logs"
PID_FILE="$LOGS_DIR/last_pid.txt"
ACHIEVEMENTS_DIR="$BASE_DIR/Achievements"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_separator() {
    echo "============================================================"
}

# 解析实验进度
parse_progress() {
    LATEST_LOG=$1

    if [ ! -f "$LATEST_LOG" ]; then
        return
    fi

    echo -e "\n${CYAN}=== 实验进度 ===${NC}"
    echo "------------------------------------------------------------"

    # 提取配置信息
    RUN_RANGE=$(grep -o "运行范围: [0-9]* - [0-9]*" "$LATEST_LOG" 2>/dev/null | head -1)
    METHODS=$(grep -o "方法列表:.*" "$LATEST_LOG" 2>/dev/null | head -1)

    if [ -n "$RUN_RANGE" ]; then
        echo "$RUN_RANGE"
    fi
    if [ -n "$METHODS" ]; then
        echo "$METHODS"
    fi

    echo ""

    # 提取当前实验编号
    CURRENT_EXP=$(grep -o "Experiment [0-9]* ([0-9]*/[0-9]*" "$LATEST_LOG" 2>/dev/null | tail -1)
    if [ -n "$CURRENT_EXP" ]; then
        EXP_NUM=$(echo "$CURRENT_EXP" | grep -o "Experiment [0-9]*" | grep -o "[0-9]*")
        EXP_PROGRESS=$(echo "$CURRENT_EXP" | grep -o "[0-9]*/[0-9]*")
        echo -e "${GREEN}当前: 第 $EXP_NUM 次实验 ($EXP_PROGRESS)${NC}"
    fi

    # 提取当前运行的方法
    CURRENT_METHOD=$(grep -E "=== Running .+ ===" "$LATEST_LOG" 2>/dev/null | tail -1)
    if [ -n "$CURRENT_METHOD" ]; then
        METHOD_NAME=$(echo "$CURRENT_METHOD" | sed 's/=== Running //' | sed 's/ ===//')
        echo -e "${BLUE}正在运行方法: $METHOD_NAME${NC}"
    fi

    echo "------------------------------------------------------------"
}

# 检查状态
check_status() {
    print_separator
    echo -e "${CYAN}实验运行状态检查${NC}"
    print_separator

    # 检查日志目录
    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在，可能还没有运行过实验${NC}"
        return
    fi

    # 找到最新日志
    LATEST_LOG=$(find "$LOGS_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    # 检查PID文件
    if [ ! -f "$PID_FILE" ]; then
        echo -e "\n${YELLOW}[!] 未找到PID记录文件${NC}"
        if [ -n "$LATEST_LOG" ]; then
            parse_progress "$LATEST_LOG"
        fi
        return
    fi

    PID=$(cat "$PID_FILE")
    echo -e "\n记录的进程PID: $PID"

    # 检查进程是否在运行
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "\n${GREEN}[✓] 实验正在运行中!${NC}"
        echo "------------------------------------------------------------"
        ps -p "$PID" -o pid,ppid,cmd,%cpu,%mem,etime,stat 2>/dev/null | head -2

        # 检查是否为Python进程
        CMD=$(ps -p "$PID" -o comm= 2>/dev/null)
        if [[ "$CMD" == *"python"* ]]; then
            echo -e "\n进程类型: Python实验进程"
        else
            echo -e "\n${YELLOW}[!] 警告: 该PID不是Python进程，可能是旧记录${NC}"
        fi
    else
        echo -e "\n${RED}[✗] 实验未在运行 (进程已结束或不存在)${NC}"
    fi

    # 显示进度
    if [ -n "$LATEST_LOG" ]; then
        parse_progress "$LATEST_LOG"
    fi

    # 日志统计
    LOG_COUNT=$(find "$LOGS_DIR" -name "*.log" -type f 2>/dev/null | wc -l)
    echo -e "\n日志文件数量: $LOG_COUNT"

    if [ "$LOG_COUNT" -gt 0 ]; then
        echo "日志总大小: $(du -sh "$LOGS_DIR"/*.log 2>/dev/null | tail -1 | cut -f1)"
    fi

    print_separator
}

# 查看实时日志
view_log() {
    print_separator
    echo -e "${CYAN}查看实时日志${NC}"
    print_separator

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在${NC}"
        return
    fi

    # 找到最新日志
    LATEST_LOG=$(find "$LOGS_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$LATEST_LOG" ]; then
        echo -e "\n${YELLOW}[!] 未找到日志文件${NC}"
        return
    fi

    echo -e "\n${GREEN}日志文件: $(basename "$LATEST_LOG")${NC}"
    echo "文件大小: $(du -h "$LATEST_LOG" | cut -f1)"
    echo ""
    echo -e "${CYAN}按 Ctrl+C 退出日志查看${NC}"
    echo "------------------------------------------------------------"
    echo ""

    tail -f "$LATEST_LOG"
}

# 查看已有实验结果
view_results() {
    print_separator
    echo -e "${CYAN}已有实验结果概览${NC}"
    print_separator

    if [ ! -d "$ACHIEVEMENTS_DIR" ]; then
        echo -e "\n${YELLOW}[!] Achievements目录不存在${NC}"
        return
    fi

    # 统计已完成的实验
    COMPLETED=0
    PARTIAL=0
    EMPTY=0

    for dir in "$ACHIEVEMENTS_DIR"/*; do
        if [ -d "$dir" ]; then
            if [ -f "$dir/train_data.xlsx" ] \
                && [ -f "$dir/val_data.xlsx" ] \
                && [ -f "$dir/test_data.xlsx" ] \
                && compgen -G "$dir/model_*.pt" > /dev/null; then
                COMPLETED=$((COMPLETED + 1))
            elif [ -f "$dir/train_data.xlsx" ]; then
                PARTIAL=$((PARTIAL + 1))
            else
                EMPTY=$((EMPTY + 1))
            fi
        fi
    done

    echo -e "\n${GREEN}[✓] 已完成: $COMPLETED 个实验${NC}"
    echo -e "${YELLOW}[!] 部分完成: $PARTIAL 个实验${NC}"
    echo -e "${BLUE}[+] 空/未开始: $EMPTY 个实验${NC}"

    # 显示最近完成的实验
    if [ "$COMPLETED" -gt 0 ]; then
        echo -e "\n${CYAN}最近完成的实验:${NC}"
        find "$ACHIEVEMENTS_DIR" -name "model_*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -5 | while read -r line; do
            dir_path=$(echo "$line" | cut -d' ' -f2-)
            run_name=$(basename "$(dirname "$dir_path")")
            time_str=$(date -d "@$(echo "$line" | cut -d' ' -f1)" "+%Y-%m-%d %H:%M" 2>/dev/null || echo "未知")
            echo "  - Run $run_name ($time_str)"
        done
    fi

    print_separator
}

# 停止实验
stop_experiment() {
    print_separator
    echo -e "${CYAN}停止实验${NC}"
    print_separator

    if [ ! -f "$PID_FILE" ]; then
        echo -e "\n${YELLOW}[!] 未找到PID记录文件，无法停止${NC}"
        return
    fi

    PID=$(cat "$PID_FILE")
    echo -e "\n目标进程PID: $PID"

    # 检查进程是否在运行
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${GREEN}[✓] 进程已不在运行${NC}"
        return
    fi

    # 显示进程信息
    echo -e "\n进程信息:"
    ps -p "$PID" -o pid,ppid,cmd,%cpu,%mem,etime,stat 2>/dev/null | head -2

    # 确认
    echo ""
    read -p "确认要停止该进程吗? (y/n): " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "操作已取消"
        return
    fi

    # 发送终止信号
    if kill "$PID" 2>/dev/null; then
        echo -e "\n${GREEN}[✓] 已发送终止信号到进程 $PID${NC}"
        echo "请稍后检查进程是否已停止"
    else
        echo -e "\n${RED}[✗] 终止失败，可能需要更高权限${NC}"
    fi
}

# 清除日志
clear_logs() {
    CLEAR_PID=$1
    print_separator
    echo -e "${CYAN}清除日志${NC}"
    print_separator

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在${NC}"
        return
    fi

    # 统计
    LOG_FILES=$(find "$LOGS_DIR" -name "*.log" -type f 2>/dev/null)
    LOG_COUNT=$(echo "$LOG_FILES" | grep -c . 2>/dev/null || echo 0)

    if [ "$LOG_COUNT" -eq 0 ] && [ "$CLEAR_PID" != "true" ]; then
        echo "没有日志文件需要清除"
        return
    fi

    echo -e "\n找到 $LOG_COUNT 个日志文件"
    if [ "$LOG_COUNT" -gt 0 ]; then
        echo "日志文件:"
        echo "$LOG_FILES" | while read -r f; do
            [ -n "$f" ] && echo "  - $(basename "$f") ($(du -h "$f" | cut -f1))"
        done
    fi

    if [ "$CLEAR_PID" = "true" ] && [ -f "$PID_FILE" ]; then
        echo -e "\nPID文件: $(basename "$PID_FILE")"
    fi

    # 确认
    echo ""
    read -p "确认要删除这些文件吗? (y/n): " CONFIRM
    if [ "$CONFIRM" != "y" ]; then
        echo "操作已取消"
        return
    fi

    # 删除日志
    DELETED=0
    echo "$LOG_FILES" | while read -r f; do
        if [ -n "$f" ] && [ -f "$f" ]; then
            rm -f "$f" && DELETED=$((DELETED + 1))
        fi
    done

    if [ "$CLEAR_PID" = "true" ] && [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
        echo -e "${GREEN}[✓] 已删除 PID 文件${NC}"
    fi

    echo -e "\n${GREEN}[✓] 已删除 $LOG_COUNT 个日志文件${NC}"
}

# 主菜单
show_menu() {
    echo ""
    print_separator
    echo -e "${CYAN}DA-DGP 实验监控 - 主菜单${NC}"
    print_separator
    echo ""
    echo "  1) 检查运行状态"
    echo "  2) 查看实时日志 (tail -f)"
    echo "  3) 查看已有实验结果"
    echo "  4) 停止正在运行的实验"
    echo "  5) 清除日志文件"
    echo "  6) 清除日志和PID记录"
    echo "  q) 退出"
    echo ""
    print_separator
}

# 主交互流程
main() {
    # 如果带参数运行，直接执行对应功能
    if [ $# -gt 0 ]; then
        case "$1" in
            status)
                check_status
                exit 0
                ;;
            log)
                view_log
                exit 0
                ;;
            results)
                view_results
                exit 0
                ;;
            stop)
                stop_experiment
                exit 0
                ;;
            clear)
                clear_logs false
                exit 0
                ;;
            clear-all)
                clear_logs true
                exit 0
                ;;
            *)
                echo "用法: $0 [status|log|results|stop|clear|clear-all]"
                exit 1
                ;;
        esac
    fi

    # 交互式菜单
    while true; do
        show_menu
        read -p "请选择操作 (1-6/q): " choice

        case "$choice" in
            1)
                check_status
                ;;
            2)
                view_log
                ;;
            3)
                view_results
                ;;
            4)
                stop_experiment
                ;;
            5)
                clear_logs false
                ;;
            6)
                clear_logs true
                ;;
            q|Q)
                echo ""
                echo -e "${GREEN}已退出监控脚本${NC}"
                exit 0
                ;;
            *)
                echo -e "\n${RED}[!] 无效选择，请重新输入${NC}"
                ;;
        esac

        echo ""
        read -p "按回车键继续..."
    done
}

# 运行主流程
main "$@"
