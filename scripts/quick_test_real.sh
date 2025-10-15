#!/bin/bash
# 快速测试真实MMLU搜索（使用小参数）

echo "========================================================================"
echo "快速测试 - 真实MMLU搜索"
echo "========================================================================"
echo ""
echo "配置:"
echo "  - GPU: 3"
echo "  - 快速limit: 10 (非常快)"
echo "  - 完整limit: 50 (中等)"
echo "  - 种群: 20 (小)"
echo "  - 最大代数: 15"
echo "  - 无改进阈值: 8"
echo ""
echo "预计时间: ~4-5小时"
echo ""
read -p "按Enter开始测试，或Ctrl+C取消..."

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# 进入GA代码目录
cd "${GA_CODE_DIR}"

# 运行搜索
python run_ga_search_real.py \
  --gpu 3 \
  --fast-limit 10 \
  --full-limit 50 \
  --population 20 \
  --generations 15 \
  --no-improve 8 \
  --output-dir results/real_test \
  --verbose

echo ""
echo "========================================================================"
echo "测试完成！"
echo "========================================================================"

