#!/bin/bash
# 项目配置文件 - 在所有脚本中source此文件

# 项目根目录（自动检测）
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python环境（修改为新环境名）
export CONDA_ENV_NAME="ga_layer_search"
export CONDA_ENV_PATH="${HOME}/.conda/envs/${CONDA_ENV_NAME}"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"

# 模型文件路径（都在项目内）
export LLAMA_LAYERS_DIR="${PROJECT_ROOT}/extracted_llama_layers"
export MODELSCOPE_CACHE="${PROJECT_ROOT}/modelscope_cache"

# 目录结构
export GA_CODE_DIR="${PROJECT_ROOT}/genetic_algorithm"
export MODEL_PREP_DIR="${PROJECT_ROOT}/model_preparation"
export SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
export RESULTS_DIR="${PROJECT_ROOT}/results"
export CHECKPOINTS_DIR="${PROJECT_ROOT}/model_checkpoints"

# 设置ModelScope缓存目录（避免污染全局）
export MODELSCOPE_CACHE="${PROJECT_ROOT}/modelscope_cache"

echo "✅ 配置已加载"
echo "   项目根目录: ${PROJECT_ROOT}"
echo "   Conda环境: ${CONDA_ENV_NAME}"
echo "   层文件目录: ${LLAMA_LAYERS_DIR}"
echo "   ModelScope缓存: ${MODELSCOPE_CACHE}"

