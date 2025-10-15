#!/bin/bash
# 监控GA搜索进度

if [ -z "$1" ]; then
    echo "用法: $0 <结果目录>"
    echo "例如: $0 results/real_results"
    exit 1
fi

RESULT_DIR="$1"

# 找到最新的检查点目录
CHECKPOINT_DIR=$(ls -td ${RESULT_DIR}/checkpoints_* 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "在 ${RESULT_DIR} 中没有找到检查点目录"
    exit 1
fi

echo "========================================================================"
echo "GA搜索进度监控"
echo "========================================================================"
echo ""
echo "检查点目录: $CHECKPOINT_DIR"
echo ""

# 列出所有检查点
python view_checkpoint.py "$CHECKPOINT_DIR" --list

echo ""
echo "========================================================================"
echo "最新检查点详情:"
echo "========================================================================"

# 显示最新检查点详情
python view_checkpoint.py "$CHECKPOINT_DIR"

echo ""
echo "========================================================================"
echo "提示:"
echo "  - 实时查看日志: tail -f ${RESULT_DIR}/search_log_*.txt"
echo "  - 重新运行监控: $0 $RESULT_DIR"
echo "========================================================================"

