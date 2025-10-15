"""
路径配置模块 - 管理所有文件路径
"""
import os
from pathlib import Path

# 项目根目录（genetic_algorithm的父目录）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 获取层文件路径（优先从环境变量读取，否则使用默认值）
LLAMA_LAYERS_DIR = os.environ.get(
    'LLAMA_LAYERS_DIR',
    str(PROJECT_ROOT / 'extracted_llama_layers')
)

# 如果默认路径不存在，尝试原始项目路径
if not Path(LLAMA_LAYERS_DIR).exists():
    fallback_path = "/home/huzhuangfei/Code/GandA/Gather-and-Aggregate/extracted_llama_layers"
    if Path(fallback_path).exists():
        LLAMA_LAYERS_DIR = fallback_path
        print(f"⚠️  使用原始项目的层文件: {LLAMA_LAYERS_DIR}")
    else:
        print(f"❌ 警告: 层文件目录不存在: {LLAMA_LAYERS_DIR}")
        print(f"   请运行 model_preparation/extract_layers.py 提取层文件")

# 数据文件路径
SINGLE_LAYER_RESULTS_FILE = PROJECT_ROOT / 'scripts' / 'single_layer_results.json'

# 输出目录
RESULTS_DIR = PROJECT_ROOT / 'results'

# 确保输出目录存在
RESULTS_DIR.mkdir(exist_ok=True)

def get_llama_layers_dir():
    """获取Llama层文件目录"""
    return LLAMA_LAYERS_DIR

def get_single_layer_results_file():
    """获取单层结果文件路径"""
    return str(SINGLE_LAYER_RESULTS_FILE)

def get_results_dir():
    """获取结果输出目录"""
    return str(RESULTS_DIR)

