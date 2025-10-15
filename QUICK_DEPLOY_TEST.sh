#!/bin/bash
# 快速部署测试 - 验证环境和代码，不下载大文件

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "快速部署测试（不下载大文件）"
echo "========================================================================"
echo ""

ENV_NAME="ga_layer_search"

# 1. 创建环境
echo "1️⃣  创建Conda环境..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "   ⚠️  环境已存在，删除并重建..."
    conda env remove -n ${ENV_NAME} -y
fi

conda create -n ${ENV_NAME} python=3.10 -y
echo "   ✅ 环境创建完成"
echo ""

# 2. 获取环境路径
ENV_PATH=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
ENV_PYTHON="${ENV_PATH}/bin/python"
ENV_PIP="${ENV_PATH}/bin/pip"

echo "2️⃣  安装依赖..."
echo "   环境: ${ENV_PATH}"
echo "   先装numpy和scipy（避免兼容性问题）..."
${ENV_PIP} install -q numpy==1.26.4 scipy==1.14.1
echo "   安装其他依赖..."
${ENV_PIP} install -q -r requirements.txt
echo "   ✅ 依赖安装完成"
echo ""

# 3. 验证安装
echo "3️⃣  验证安装..."
${ENV_PYTHON} -c "import torch; print(f'   ✅ PyTorch {torch.__version__}')"
${ENV_PYTHON} -c "import transformers; print(f'   ✅ Transformers {transformers.__version__}')"
${ENV_PYTHON} -c "import modelscope; print(f'   ✅ ModelScope {modelscope.__version__}')"
echo ""

# 4. 测试代码导入
echo "4️⃣  测试代码导入..."
cd genetic_algorithm
${ENV_PYTHON} -c "from config import GAConfig; from population import Population; from ga_core import GeneticAlgorithm; print('   ✅ GA模块可导入')"
cd ../model_preparation
${ENV_PYTHON} -c "from modelscope_utils import get_model_modelscope; print('   ✅ 模型工具可导入')"
cd ..
echo ""

# 5. 检查models目录
echo "5️⃣  检查models目录..."
if [ -f "models/llamba.py" ]; then
    ${ENV_PYTHON} -c "import sys; sys.path.insert(0, '.'); from models.llamba import LlambaLMHeadModel; print('   ✅ models.llamba可导入')"
else
    echo "   ❌ models/llamba.py不存在"
    exit 1
fi
echo ""

echo "========================================================================"
echo "✅ 快速部署测试通过！"
echo "========================================================================"
echo ""
echo "📋 测试结果:"
echo "   [√] Conda环境创建"
echo "   [√] Python依赖安装"
echo "   [√] 代码模块导入"
echo "   [√] models目录正确"
echo ""
echo "🎯 环境已就绪，可以:"
echo "   1. 提取Llama层（需GPU，~10分钟）:"
echo "      cd model_preparation"
echo "      ${ENV_PYTHON} extract_layers.py --model_name llama --output_dir ../extracted_llama_layers"
echo ""
echo "   2. 生成checkpoint（需GPU + 层文件，~5分钟）:"
echo "      ${ENV_PYTHON} create_replaced_model_checkpoint.py --layers 11 13 17 29 --output_dir ../model_checkpoints/best"
echo ""
echo "   3. 运行搜索（需GPU + 层文件，~数小时到数天）:"
echo "      cd ../scripts && ./quick_test_real.sh"
echo ""

