#!/bin/bash
# 检查GA搜索的结果文件

echo "========================================================================"
echo "GA搜索 - 结果文件检查"
echo "========================================================================"
echo ""

echo "📁 ga_quick (快速测试) - results/real_test/"
echo "----------------------------------------"
if [ -d "results/real_test" ]; then
    echo "日志文件:"
    ls -lth results/real_test/*.txt 2>/dev/null | head -3 || echo "  暂无日志文件"
    echo ""
    echo "检查点:"
    ls -lth results/real_test/checkpoints_*/*.json 2>/dev/null | head -5 || echo "  暂无检查点"
    echo ""
    echo "最新检查点:"
    [ -d "results/real_test/checkpoints_"* ] && python view_checkpoint.py results/real_test/checkpoints_*/ 2>/dev/null || echo "  暂无检查点"
else
    echo "  目录不存在"
fi

echo ""
echo "========================================================================"
echo ""

echo "📁 ga_full_fast (完整搜索) - results/real_results_fast/"
echo "----------------------------------------"
if [ -d "results/real_results_fast" ]; then
    echo "日志文件:"
    ls -lth results/real_results_fast/*.txt 2>/dev/null | head -3 || echo "  暂无日志文件"
    echo ""
    echo "检查点:"
    ls -lth results/real_results_fast/checkpoints_*/*.json 2>/dev/null | head -5 || echo "  暂无检查点"
    echo ""
    echo "最新检查点:"
    [ -d "results/real_results_fast/checkpoints_"* ] && python view_checkpoint.py results/real_results_fast/checkpoints_*/ 2>/dev/null || echo "  暂无检查点"
else
    echo "  目录不存在"
fi

echo ""
echo "========================================================================"
echo "提示:"
echo "  - 查看详细检查点: ./monitor_progress.sh results/real_test"
echo "  - 查看详细检查点: ./monitor_progress.sh results/real_results_fast"
echo "  - 查看tmux状态: ./check_tmux.sh"
echo "========================================================================"

