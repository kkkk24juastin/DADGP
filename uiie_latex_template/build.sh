#!/bin/bash
# LaTeX 编译脚本 - IISE Transactions 论文
# 用法: ./build.sh [选项]
#   无参数: 编译主文件和补充材料
#   main:   仅编译主文件
#   sup:    仅编译补充材料
#   clean:  清理临时文件
#   help:   显示帮助信息

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 文件名
MAIN_TEX="IISE Transactions Latex Template"
SUP_TEX="Supplementary Materials"
BIB_FILE="IISE-Trans"

# 检查必要工具
check_tools() {
    for cmd in xelatex bibtex; do
        if ! command -v "$cmd" &> /dev/null; then
            echo -e "${RED}错误: 未找到 $cmd，请先安装 TeX Live 或 MiKTeX${NC}"
            exit 1
        fi
    done
}

# 编译主文件
compile_main() {
    echo -e "${YELLOW}编译主文件: ${MAIN_TEX}.tex${NC}"

    echo "  [1/4] 第一次 xelatex 编译..."
    xelatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1

    echo "  [2/4] 运行 bibtex 处理参考文献..."
    bibtex "$MAIN_TEX" > /dev/null 2>&1 || true

    echo "  [3/4] 第二次 xelatex 编译..."
    xelatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1

    echo "  [4/4] 第三次 xelatex 编译..."
    xelatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1

    echo -e "${GREEN}主文件编译完成: ${MAIN_TEX}.pdf${NC}"
}

# 编译补充材料
compile_sup() {
    echo -e "${YELLOW}编译补充材料: ${SUP_TEX}.tex${NC}"

    echo "  [1/2] 第一次 xelatex 编译..."
    xelatex -interaction=nonstopmode "$SUP_TEX.tex" > /dev/null 2>&1

    echo "  [2/2] 第二次 xelatex 编译..."
    xelatex -interaction=nonstopmode "$SUP_TEX.tex" > /dev/null 2>&1

    echo -e "${GREEN}补充材料编译完成: ${SUP_TEX}.pdf${NC}"
}

# 清理临时文件
clean() {
    echo -e "${YELLOW}清理临时文件...${NC}"

    # 主文件临时文件
    rm -f "${MAIN_TEX}.aux" "${MAIN_TEX}.bbl" "${MAIN_TEX}.blg"
    rm -f "${MAIN_TEX}.fdb_latexmk" "${MAIN_TEX}.fls"
    rm -f "${MAIN_TEX}.log" "${MAIN_TEX}.out"
    rm -f "${MAIN_TEX}.synctex.gz" "${MAIN_TEX}.xdv"

    # 补充材料临时文件
    rm -f "${SUP_TEX}.aux" "${SUP_TEX}.log"
    rm -f "${SUP_TEX}.synctex.gz"

    echo -e "${GREEN}清理完成${NC}"
}

# 显示帮助
show_help() {
    echo "LaTeX 编译脚本 - IISE Transactions 论文"
    echo ""
    echo "用法: ./build.sh [选项]"
    echo ""
    echo "选项:"
    echo "  (无参数)  编译主文件和补充材料"
    echo "  main      仅编译主文件"
    echo "  sup       仅编译补充材料"
    echo "  clean     清理临时文件"
    echo "  help      显示此帮助信息"
}

# 主逻辑
check_tools

case "${1:-all}" in
    main)
        compile_main
        ;;
    sup)
        compile_sup
        ;;
    clean)
        clean
        ;;
    help)
        show_help
        ;;
    all)
        compile_main
        echo ""
        compile_sup
        ;;
    *)
        echo -e "${RED}未知选项: $1${NC}"
        show_help
        exit 1
        ;;
esac
