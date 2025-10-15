#!/bin/bash
# 轻量级安装验证脚本 - 只检查环境和代码，不下载大文件

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "GA Layer Search - 安装验证"
echo "========================================================================"
echo ""

# 1. 检查文件完整性
echo "1️⃣  检查项目文件..."
REQUIRED_DIRS=("genetic_algorithm" "model_preparation" "scripts" "models")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir/"
    else
        echo "   ❌ 缺少 $dir/"
        exit 1
    fi
done
echo ""

# 2. 检查conda
echo "2️⃣  检查Conda环境..."
if conda env list | grep -q "^ga_layer_search "; then
    ENV_PATH=$(conda env list | grep "^ga_layer_search " | awk '{print $NF}')
    echo "   ✅ 环境已存在: ${ENV_PATH}"
else
    echo "   ❌ 环境不存在，请运行: ./DEPLOY_TEST.sh"
    exit 1
fi
echo ""

# 3. 检查Python包
echo "3️⃣  检查Python依赖..."
ENV_PYTHON="${ENV_PATH}/bin/python"
REQUIRED_PACKAGES=("torch" "transformers" "modelscope")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ${ENV_PYTHON} -c "import ${pkg}" 2>/dev/null; then
        version=$(${ENV_PYTHON} -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "unknown")
        echo "   ✅ ${pkg} (${version})"
    else
        echo "   ❌ 缺少 ${pkg}"
        echo "      运行: ${ENV_PATH}/bin/pip install ${pkg}"
    fi
done
echo ""

# 4. 检查models目录
echo "4️⃣  检查models目录..."
if [ -f "models/llamba.py" ]; then
    echo "   ✅ models/llamba.py 存在"
else
    echo "   ❌ models/llamba.py 不存在"
    exit 1
fi
echo ""

# 5. 测试Python import
echo "5️⃣  测试代码导入..."
cd genetic_algorithm
if ${ENV_PYTHON} -c "from config import GAConfig; print('✅ GAConfig')" 2>/dev/null; then
    echo "   ✅ genetic_algorithm 模块可导入"
else
    echo "   ❌ genetic_algorithm 导入失败"
    exit 1
fi
cd ..
echo ""

echo "========================================================================"
echo "✅ 验证通过！项目配置正确"
echo "========================================================================"
echo ""
echo "📋 下一步:"
echo "   - 提取Llama层: cd model_preparation && python extract_layers.py ..."
echo "   - 运行搜索: cd scripts && ./quick_test_real.sh"
echo "   - 生成checkpoint: cd model_preparation && python create_replaced_model_checkpoint.py ..."
echo ""
echo "📚 详细说明: 查看 README.md"
echo ""

