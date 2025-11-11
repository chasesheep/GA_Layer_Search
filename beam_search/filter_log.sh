#!/bin/bash
# 从beam search log中提取关键信息，过滤警告

LOG_FILE="$1"

if [ -z "$LOG_FILE" ]; then
    echo "用法: bash filter_log.sh <log_file>"
    echo "示例: bash filter_log.sh beam_aligned_log_*.txt"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 文件不存在: $LOG_FILE"
    exit 1
fi

# 生成输出文件名
OUTPUT_FILE="${LOG_FILE%.txt}_clean.txt"

echo "📄 过滤log文件: $(basename $LOG_FILE)"
echo "📝 输出文件: $(basename $OUTPUT_FILE)"
echo "========================================================================"

# 提取关键信息到文件
grep -E "Beam Search|Parameters:|Depth|Testing \[|✅ MMLU completed|❌ MMLU|Top|Best|Checkpoint saved|Evaluation count|replaced_layers|score" "$LOG_FILE" | \
    grep -v "trust_remote_code\|Parquet\|loading script\|lm_eval" > "$OUTPUT_FILE"

# 显示统计信息
TOTAL_LINES=$(wc -l < "$LOG_FILE")
CLEAN_LINES=$(wc -l < "$OUTPUT_FILE")
COMPLETED=$(grep -c "✅ MMLU completed" "$OUTPUT_FILE")
FAILED=$(grep -c "❌ MMLU" "$OUTPUT_FILE")

echo "✅ 过滤完成！"
echo ""
echo "📊 统计信息:"
echo "  - 原始行数: $TOTAL_LINES"
echo "  - 清洁行数: $CLEAN_LINES (压缩 $(( (TOTAL_LINES - CLEAN_LINES) * 100 / TOTAL_LINES ))%)"
echo "  - 成功评估: $COMPLETED"
echo "  - 失败评估: $FAILED"
echo ""
echo "💡 查看清洁log: less $OUTPUT_FILE"
echo "💡 查看原始log: less $LOG_FILE"

# 显示前50行预览
echo ""
echo "========================================================================"
echo "📋 前50行预览:"
echo "========================================================================"
head -50 "$OUTPUT_FILE"

