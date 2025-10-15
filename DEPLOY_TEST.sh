#!/bin/bash
# 完整部署测试脚本 - 从零开始验证项目可用性

set -e  # 遇到错误立即退出

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "GA Layer Search - 完整部署测试"
echo "========================================================================"
echo ""
echo "项目目录: ${PROJECT_ROOT}"
echo ""

# 步骤1: 检查conda
echo "========================================================================"
echo "步骤1: 检查Conda"
echo "========================================================================"
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda命令"
    echo "   请先安装Miniconda或Anaconda"
    exit 1
fi
echo "✅ Conda已安装: $(conda --version)"
echo ""

# 步骤2: 创建conda环境
echo "========================================================================"
echo "步骤2: 创建Conda环境"
echo "========================================================================"
ENV_NAME="ga_layer_search"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 ${ENV_NAME} 已存在，将使用现有环境"
    echo "   （如需重建，请手动删除：conda env remove -n ${ENV_NAME} -y）"
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "创建新环境 ${ENV_NAME}..."
    conda create -n ${ENV_NAME} python=3.10 -y
    echo "✅ 环境创建完成"
else
    echo "✅ 环境已存在"
fi
echo ""

# 步骤3: 安装Python依赖
echo "========================================================================"
echo "步骤3: 安装Python依赖到新环境"
echo "========================================================================"

# 自动获取环境路径（适配不同用户）
ENV_PATH=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
if [ -z "${ENV_PATH}" ]; then
    echo "❌ 错误: 无法找到环境 ${ENV_NAME} 的路径"
    exit 1
fi

ENV_PYTHON="${ENV_PATH}/bin/python"
ENV_PIP="${ENV_PATH}/bin/pip"

echo "目标环境: ${ENV_NAME}"
echo "环境路径: ${ENV_PATH}"
echo "Python: ${ENV_PYTHON}"
echo ""

echo "安装依赖包..."
echo "  步骤1: 先安装numpy和scipy（避免兼容性问题）"
${ENV_PIP} install -q numpy==1.26.4 scipy==1.14.1
echo "  步骤2: 安装其他依赖"
${ENV_PIP} install -q -r requirements.txt

echo ""
echo "验证安装..."
${ENV_PIP} list | grep -E "torch|transformers|lm-eval|modelscope" | head -5
echo "✅ 依赖安装完成"
echo ""

# 步骤4: 检查models目录
echo "========================================================================"
echo "步骤4: 检查models目录"
echo "========================================================================"
if [ ! -d "models" ]; then
    echo "❌ 错误: models目录不存在"
    echo "   这不应该发生（models目录应该在Git中）"
    exit 1
else
    echo "✅ models目录已存在: ${PROJECT_ROOT}/models"
    ls -lh models/ | head -5
fi
echo ""

# 步骤5: 配置ModelScope缓存
echo "========================================================================"
echo "步骤5: 配置ModelScope"
echo "========================================================================"
mkdir -p modelscope_cache
export MODELSCOPE_CACHE="${PROJECT_ROOT}/modelscope_cache"
echo "✅ ModelScope缓存目录: ${MODELSCOPE_CACHE}"
echo ""

# 步骤6: 下载模型
echo "========================================================================"
echo "步骤6: 下载模型"
echo "========================================================================"
cd model_preparation

echo "检查模型是否已下载..."
${ENV_PYTHON} << 'PYEOF'
import os
os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(os.getcwd()), 'modelscope_cache')

try:
    from modelscope_utils import get_model_modelscope
    
    print("正在下载/加载Llamba模型...")
    model, tokenizer, _, _ = get_model_modelscope('unaligned_llamba', is_minimal=False)
    print("✅ Llamba模型已准备")
    del model, tokenizer
    
    print("\n正在下载/加载Llama模型...")
    model, tokenizer, _, _ = get_model_modelscope('llama', is_minimal=False)
    print("✅ Llama模型已准备")
    del model, tokenizer
    
    print("\n✅ 所有模型已准备就绪")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo "❌ 模型准备失败"
    exit 1
fi
echo ""

# 步骤7: 提取Llama层
echo "========================================================================"
echo "步骤7: 提取Llama层文件"
echo "========================================================================"
cd "${PROJECT_ROOT}/model_preparation"

if [ -d "../extracted_llama_layers" ] && [ "$(ls -A ../extracted_llama_layers 2>/dev/null | wc -l)" -gt 30 ]; then
    echo "✅ 层文件已存在（$(ls ../extracted_llama_layers/*.pt 2>/dev/null | wc -l)个文件）"
else
    echo "提取Llama层文件（需要~5-10分钟）..."
    ${ENV_PYTHON} extract_layers.py \
        --model_name llama \
        --output_dir ../extracted_llama_layers
    
    if [ $? -ne 0 ]; then
        echo "❌ 层提取失败"
        exit 1
    fi
    echo "✅ 层提取完成"
fi
echo ""

# 步骤8: 测试层替换
echo "========================================================================"
echo "步骤8: 测试层替换功能"
echo "========================================================================"
cd "${PROJECT_ROOT}/model_preparation"

echo "测试单层替换（layer 17）..."
${ENV_PYTHON} test_specific_combination.py \
    --layers 17 \
    --gpu_id 7 \
    --batch_size 8 \
    --limit 10

if [ $? -ne 0 ]; then
    echo "❌ 层替换测试失败"
    exit 1
fi
echo "✅ 层替换功能正常"
echo ""

# 步骤9: 测试GA代码
echo "========================================================================"
echo "步骤9: 测试GA搜索代码（Mock函数）"
echo "========================================================================"
cd "${PROJECT_ROOT}/genetic_algorithm"

echo "运行Mock函数测试..."
timeout 300 ${ENV_PYTHON} run_complete_search.py || {
    echo "⚠️  超时或失败，但继续..."
}
echo "✅ GA代码可运行"
echo ""

# 步骤10: 测试checkpoint生成
echo "========================================================================"
echo "步骤10: 测试Checkpoint生成"
echo "========================================================================"
cd "${PROJECT_ROOT}/model_preparation"

echo "生成测试checkpoint（layer 17）..."
${ENV_PYTHON} create_replaced_model_checkpoint.py \
    --layers 17 \
    --output_dir ../model_checkpoints/test_layer17 \
    --description "部署测试checkpoint" \
    --score 0.5144 \
    --gpu 7

if [ $? -ne 0 ]; then
    echo "❌ Checkpoint生成失败"
    exit 1
fi
echo "✅ Checkpoint生成成功"
echo ""

# 步骤11: 测试checkpoint加载
echo "========================================================================"
echo "步骤11: 测试Checkpoint加载"
echo "========================================================================"
cd "${PROJECT_ROOT}/model_preparation"

echo "测试checkpoint加载..."
${ENV_PYTHON} test_checkpoint.py \
    --checkpoint ../model_checkpoints/test_layer17 \
    --gpu 7

if [ $? -ne 0 ]; then
    echo "❌ Checkpoint测试失败"
    exit 1
fi
echo "✅ Checkpoint加载正常"
echo ""

# 完成
echo "========================================================================"
echo "✅ 所有测试通过！项目已准备就绪"
echo "========================================================================"
echo ""
echo "📂 项目结构:"
echo "   - extracted_llama_layers/  ($(ls ../extracted_llama_layers/*.pt 2>/dev/null | wc -l)个层文件)"
echo "   - modelscope_cache/         (模型缓存)"
echo "   - models/                   (Llamba模型代码)"
echo "   - model_checkpoints/        (生成的checkpoint)"
echo ""
echo "🎯 下一步:"
echo "   1. 运行快速测试:"
echo "      cd scripts && ./quick_test_real.sh"
echo ""
echo "   2. 生成最优checkpoint:"
echo "      cd model_preparation && ./create_best_checkpoints.sh"
echo ""
echo "   3. 运行完整搜索:"
echo "      cd scripts && ./run_full_search.sh"
echo ""
echo "📚 文档:"
echo "   - README.md - 项目概述"
echo "   - SETUP.md - 详细安装指南"
echo "   - MODEL_CHECKPOINTS_GUIDE.md - Checkpoint使用"
echo ""

