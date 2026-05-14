#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# wireless 实验监控脚本
# 基于上级 3tasks/monitor.sh 的交互风格，适配当前 wireless/logs 与 model 目录
# ============================================================================

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="$BASE_DIR/logs"
PID_FILE="$LOGS_DIR/last_pid.txt"
MODEL_DIR="$BASE_DIR/model"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_separator() {
    echo "============================================================"
}

parse_progress() {
    local latest_log=$1

    if [ ! -f "$latest_log" ]; then
        return
    fi

    echo -e "\n${CYAN}=== 实验进度 ===${NC}"
    echo "------------------------------------------------------------"

    local methods
    local current_method

    methods=$(grep -o "Running single experiment with methods (.*" "$latest_log" 2>/dev/null | tail -1)
    current_method=$(grep -E "=== Running .+ ===" "$latest_log" 2>/dev/null | tail -1)

    [ -n "$methods" ] && echo "$methods"
    [ -n "$methods" ] && echo -e "${GREEN}当前模式: 单次实验${NC}"

    if [ -n "$current_method" ]; then
        local method_name
        method_name=$(echo "$current_method" | sed 's/=== Running //' | sed 's/ ===//')
        echo -e "${BLUE}当前方法: $method_name${NC}"
    fi

    echo "------------------------------------------------------------"
}

check_status() {
    print_separator
    echo -e "${CYAN}实验运行状态检查${NC}"
    print_separator

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在${NC}"
        return
    fi

    local latest_log
    latest_log=$(find "$LOGS_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ ! -f "$PID_FILE" ]; then
        echo -e "\n${YELLOW}[!] 未找到 PID 记录文件${NC}"
        [ -n "$latest_log" ] && parse_progress "$latest_log"
        return
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo -e "\n记录的进程 PID: $pid"

    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "\n${GREEN}[✓] 实验正在运行中!${NC}"
        ps -p "$pid" -o pid,ppid,cmd,%cpu,%mem,etime,stat 2>/dev/null | head -2
    else
        echo -e "\n${RED}[✗] 实验未在运行 (进程已结束或不存在)${NC}"
    fi

    [ -n "$latest_log" ] && parse_progress "$latest_log"
    print_separator
}

view_log() {
    print_separator
    echo -e "${CYAN}查看实时日志${NC}"
    print_separator

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在${NC}"
        return
    fi

    local latest_log
    latest_log=$(find "$LOGS_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$latest_log" ]; then
        echo -e "\n${YELLOW}[!] 未找到日志文件${NC}"
        return
    fi

    echo -e "\n${GREEN}日志文件: $(basename "$latest_log")${NC}"
    echo "文件大小: $(du -h "$latest_log" | cut -f1)"
    echo -e "${CYAN}按 Ctrl+C 退出日志查看${NC}"
    echo ""

    tail -f "$latest_log"
}

view_results() {
    print_separator
    echo -e "${CYAN}已保存模型概览${NC}"
    print_separator

    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "\n${YELLOW}[!] model 目录不存在${NC}"
        return
    fi

    local model_count
    model_count=$(find "$MODEL_DIR" -type f -name "*.pt" 2>/dev/null | wc -l)
    local plot_count
    plot_count=$(find "$MODEL_DIR" -type f -name "*.png" 2>/dev/null | wc -l)

    echo -e "\n${GREEN}[✓] 已保存模型: $model_count 个${NC}"
    echo -e "${BLUE}[+] 训练曲线图: $plot_count 个${NC}"

    if [ "$model_count" -gt 0 ]; then
        echo -e "\n${CYAN}最近保存的模型:${NC}"
        find "$MODEL_DIR" -type f -name "*.pt" -printf '%T@ %p\n' 2>/dev/null | \
            sort -rn | head -10 | while read -r line; do
                local_path=$(echo "$line" | cut -d' ' -f2-)
                time_str=$(date -d "@$(echo "$line" | cut -d' ' -f1)" "+%Y-%m-%d %H:%M" 2>/dev/null || echo "未知")
                echo "  - ${local_path#$BASE_DIR/} ($time_str)"
            done
    fi

    print_separator
}

stop_experiment() {
    print_separator
    echo -e "${CYAN}停止实验${NC}"
    print_separator

    if [ ! -f "$PID_FILE" ]; then
        echo -e "\n${YELLOW}[!] 未找到 PID 记录文件${NC}"
        return
    fi

    local pid
    pid=$(cat "$PID_FILE")
    echo -e "\n目标进程 PID: $pid"

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}[✓] 进程已不在运行${NC}"
        return
    fi

    ps -p "$pid" -o pid,ppid,cmd,%cpu,%mem,etime,stat 2>/dev/null | head -2
    echo ""
    read -p "确认要停止该进程吗? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "操作已取消"
        return
    fi

    if kill "$pid" 2>/dev/null; then
        echo -e "\n${GREEN}[✓] 已发送终止信号到进程 $pid${NC}"
    else
        echo -e "\n${RED}[✗] 终止失败${NC}"
    fi
}

clear_logs() {
    local clear_pid=$1
    print_separator
    echo -e "${CYAN}清除日志${NC}"
    print_separator

    if [ ! -d "$LOGS_DIR" ]; then
        echo -e "\n${YELLOW}[!] 日志目录不存在${NC}"
        return
    fi

    local log_files
    log_files=$(find "$LOGS_DIR" -name "*.log" -type f 2>/dev/null)
    local log_count
    log_count=$(echo "$log_files" | grep -c . 2>/dev/null || echo 0)

    if [ "$log_count" -eq 0 ] && [ "$clear_pid" != "true" ]; then
        echo "没有日志文件需要清除"
        return
    fi

    echo -e "\n找到 $log_count 个日志文件"
    [ "$clear_pid" = "true" ] && [ -f "$PID_FILE" ] && echo "将同时清除 PID 记录"
    echo ""
    read -p "确认要删除这些文件吗? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        echo "操作已取消"
        return
    fi

    if [ "$log_count" -gt 0 ]; then
        find "$LOGS_DIR" -name "*.log" -type f -delete
    fi
    if [ "$clear_pid" = "true" ] && [ -f "$PID_FILE" ]; then
        rm -f "$PID_FILE"
    fi

    echo -e "\n${GREEN}[✓] 日志清理完成${NC}"
}

show_menu() {
    echo ""
    print_separator
    echo -e "${CYAN}wireless 实验监控 - 主菜单${NC}"
    print_separator
    echo "  1) 检查运行状态"
    echo "  2) 查看实时日志"
    echo "  3) 查看已保存模型"
    echo "  4) 停止正在运行的实验"
    echo "  5) 清除日志文件"
    echo "  6) 清除日志和 PID 记录"
    echo "  q) 退出"
    echo ""
}

main() {
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

    while true; do
        show_menu
        read -p "请选择操作 (1-6/q): " choice
        case "$choice" in
            1) check_status ;;
            2) view_log ;;
            3) view_results ;;
            4) stop_experiment ;;
            5) clear_logs false ;;
            6) clear_logs true ;;
            q|Q)
                echo -e "\n${GREEN}已退出监控脚本${NC}"
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

main "$@"
