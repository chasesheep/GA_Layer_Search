#!/bin/bash
# 批量创建最优层组合的checkpoint
# 基于GA搜索发现的最佳层组合

echo "========================================================================"
echo "创建最优层组合的模型checkpoint"
echo "========================================================================"
echo ""
echo "这将创建多个最优层组合的checkpoint，每个约16GB"
echo "确保有足够的磁盘空间（至少100GB）"
echo ""
read -p "按Enter继续，或Ctrl+C取消..."

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../config.sh"

# 进入model_preparation目录
cd "${SCRIPT_DIR}"

# 设置输出目录
CHECKPOINTS_DIR="${PROJECT_ROOT}/model_checkpoints"
mkdir -p "${CHECKPOINTS_DIR}"

echo ""
echo "输出目录: ${CHECKPOINTS_DIR}"
echo ""

# GPU设置
GPU_ID=0
echo "使用GPU: ${GPU_ID}"
echo ""

# 定义最优层组合（根据GA搜索结果）
# 格式: "layers|description|score"
BEST_COMBINATIONS=(
    "11 13 17 21|4层最优组合 - GA搜索发现|0.5700"
    "13 16 17|3层最优组合|0.6542"
    "13 17|2层最优组合|0.5544"
    "17|1层最优组合|0.5144"
    "10 14 17 30|4层备选组合|0.5600"
)

echo "将创建 ${#BEST_COMBINATIONS[@]} 个checkpoint..."
echo ""

# 循环创建checkpoint
for combo in "${BEST_COMBINATIONS[@]}"; do
    # 解析组合信息
    IFS='|' read -r layers description score <<< "$combo"
    
    # 创建输出目录名（使用下划线连接层号）
    layers_str=$(echo $layers | tr ' ' '_')
    output_name="llamba_replaced_${layers_str}"
    output_dir="${CHECKPOINTS_DIR}/${output_name}"
    
    echo "========================================================================"
    echo "创建: ${output_name}"
    echo "  层: ${layers}"
    echo "  描述: ${description}"
    echo "  MMLU分数: ${score}"
    echo "========================================================================"
    echo ""
    
    # 运行创建脚本
    python create_replaced_model_checkpoint.py \
        --layers ${layers} \
        --llama_layers_dir "${LLAMA_LAYERS_DIR}" \
        --output_dir "${output_dir}" \
        --gpu ${GPU_ID} \
        --description "${description}" \
        --score ${score}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ ${output_name} 创建成功!"
        echo ""
    else
        echo ""
        echo "❌ ${output_name} 创建失败!"
        echo ""
        read -p "按Enter继续下一个，或Ctrl+C退出..."
    fi
    
    echo ""
    echo "清理GPU缓存..."
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null
    sleep 2
    echo ""
done

echo "========================================================================"
echo "✅ 所有checkpoint创建完成！"
echo "========================================================================"
echo ""
echo "📁 Checkpoint位置: ${CHECKPOINTS_DIR}"
echo ""
echo "📂 创建的checkpoint:"
ls -lh "${CHECKPOINTS_DIR}"
echo ""
echo "💾 总大小:"
du -sh "${CHECKPOINTS_DIR}"
echo ""
echo "🎯 使用方法:"
echo "  1. 测试checkpoint:"
echo "     python test_checkpoint.py --checkpoint ${CHECKPOINTS_DIR}/llamba_replaced_11_13_17_21"
echo ""
echo "  2. 加载使用:"
echo "     model = torch.load('${CHECKPOINTS_DIR}/llamba_replaced_11_13_17_21/model.pt')"
echo ""

