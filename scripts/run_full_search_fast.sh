#!/bin/bash
# 完整搜索 - 快速版本（降低粗搜索limit）

echo "========================================================================"
echo "完整GA搜索 - 快速版本（limit=20/None）"
echo "========================================================================"
echo ""
echo "优化配置:"
echo "  - GPU: 3"
echo "  - 快速limit: 20 (降低以加速)"
echo "  - 完整limit: None (完整MMLU)"
echo "  - 种群: 40"
echo "  - 最大代数: 20"
echo "  - 无改进阈值: 6"
echo ""
echo "预计评估次数:"
echo "  - 阶段1 (GA粗搜索):      ~400次 (limit=20)"
echo "  - 阶段2 (完整评估):      ~20次  (limit=None)"
echo "  - 阶段3 (局部优化-粗):   ~600次 (limit=20)"
echo "  - 阶段3 (局部优化-完整): ~30次  (limit=None)"
echo ""
echo "预计总时间:"
echo "  - 阶段1: ~22小时  (400次 × 200s)"
echo "  - 阶段2: ~11小时  (20次 × 2000s)"
echo "  - 阶段3: ~40小时  (600次 × 200s + 30次 × 2000s)"
echo "  - 总计:  ~73小时 (约3天)"
echo ""
echo "  注: limit=20比limit=50快很多，但仍能有效筛选"
echo ""
echo "⚠️  这是一个长时间运行的任务！"
echo ""
read -p "确认开始？按Enter继续，或Ctrl+C取消..."

# 激活环境
export PATH="/data/huzhuangfei/conda_envs/ganda_new/bin:$PATH"

# 进入目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"
cd "${GA_CODE_DIR}"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/real_results/full_search_fast_${TIMESTAMP}.log"

echo ""
echo "日志文件: $LOG_FILE"
echo ""

# 运行搜索（输出到终端和日志文件）
python run_ga_search_real.py \
  --gpu 3 \
  --fast-limit 20 \
  --full-limit None \
  --population 40 \
  --generations 20 \
  --no-improve 6 \
  --output-dir results/real_results_fast \
  --verbose 2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================================================"
echo "搜索完成！"
echo "========================================================================"
echo "日志文件: $LOG_FILE"

