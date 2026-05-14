#!/bin/bash
# -*- coding: utf-8 -*-
# ============================================================================
# LaTeX compile script for the IISE Transactions manuscript
# ============================================================================

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TEX="IISE Transactions Latex Template.tex"

ENGINE="${LATEX_ENGINE:-xelatex}"
BIBTEX="${BIBTEX:-bibtex}"

usage() {
    cat <<'EOF'
用法:
  ./compile_latex.sh [tex文件]

示例:
  ./compile_latex.sh
  ./compile_latex.sh "IISE Transactions Latex Template.tex"
  LATEX_ENGINE=lualatex ./compile_latex.sh

说明:
  默认使用 xelatex，适合当前包含 ctex 中文内容的论文。
  如果 tex 文件包含 \bibliography{...}，脚本会自动运行 bibtex。
EOF
}

print_separator() {
    echo "============================================================"
}

require_command() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[错误] 未找到命令: $cmd"
        echo "请先安装对应的 TeX 发行版，或设置 LATEX_ENGINE 指向可用引擎。"
        exit 1
    fi
}

run_latex() {
    local pass_name="$1"
    echo ""
    echo "[$pass_name] $ENGINE"
    "$ENGINE" -interaction=nonstopmode -file-line-error -synctex=1 "$TEX_FILE"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

TEX_FILE="${1:-$DEFAULT_TEX}"

cd "$SCRIPT_DIR"

if [ ! -f "$TEX_FILE" ]; then
    echo "[错误] 找不到 TeX 文件: $TEX_FILE"
    echo "当前目录: $SCRIPT_DIR"
    exit 1
fi

if [[ "$TEX_FILE" != *.tex ]]; then
    echo "[错误] 输入文件必须以 .tex 结尾: $TEX_FILE"
    exit 1
fi

JOB_NAME="${TEX_FILE%.tex}"
PDF_FILE="${JOB_NAME}.pdf"

require_command "$ENGINE"

if grep -q '\\bibliography{' "$TEX_FILE"; then
    NEED_BIBTEX=1
    require_command "$BIBTEX"
else
    NEED_BIBTEX=0
fi

print_separator
echo "开始编译 LaTeX"
echo "工作目录: $SCRIPT_DIR"
echo "源文件: $TEX_FILE"
echo "编译引擎: $ENGINE"
print_separator

run_latex "第 1 次编译"

if [ "$NEED_BIBTEX" -eq 1 ]; then
    echo ""
    echo "[参考文献] $BIBTEX"
    "$BIBTEX" "$JOB_NAME"
else
    echo ""
    echo "[参考文献] 未检测到 \\bibliography{...}，跳过 bibtex"
fi

run_latex "第 2 次编译"
run_latex "第 3 次编译"

print_separator
echo "[完成] PDF 已生成:"
echo "$SCRIPT_DIR/$PDF_FILE"
print_separator
