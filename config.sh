#!/bin/bash
# 项目配置文件 - 在所有脚本中source此文件

# 项目根目录（自动检测）
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python环境
export CONDA_ENV_PATH="/data/huzhuangfei/conda_envs/ganda_new"
export PATH="${CONDA_ENV_PATH}/bin:$PATH"

# 模型文件路径（需要根据实际情况修改）
# 如果已经有提取好的层文件，设置此路径
export LLAMA_LAYERS_DIR="${PROJECT_ROOT}/extracted_llama_layers"

# 如果还没有提取层文件，可以设置为原始项目路径
# export LLAMA_LAYERS_DIR="/home/huzhuangfei/Code/GandA/Gather-and-Aggregate/extracted_llama_layers"

# GA代码目录
export GA_CODE_DIR="${PROJECT_ROOT}/genetic_algorithm"

# 脚本目录
export SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

# 输出目录
export RESULTS_DIR="${PROJECT_ROOT}/results"

echo "✅ 配置已加载"
echo "   项目根目录: ${PROJECT_ROOT}"
echo "   层文件目录: ${LLAMA_LAYERS_DIR}"

