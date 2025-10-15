# 安装和配置指南

本文档说明如何从零开始设置和运行GA层替换搜索系统。

## 📋 前置要求

### 硬件要求

- **GPU**: NVIDIA GPU，推荐18GB+显存（如RTX A6000）
- **CPU**: 多核CPU（用于Ray并行评估）
- **内存**: 32GB+ RAM
- **硬盘**: 50GB+可用空间（用于模型和层文件）

### 软件要求

- **操作系统**: Linux（Ubuntu 18.04+推荐）
- **Python**: 3.8-3.10
- **CUDA**: 11.x或12.x（匹配PyTorch版本）
- **Conda或venv**: 推荐使用Conda管理环境

## 🚀 安装步骤

### 第1步：克隆仓库

```bash
git clone <repository_url> GA_Layer_Search
cd GA_Layer_Search
```

### 第2步：创建Python环境

使用Conda（推荐）:

```bash
# 创建新环境
conda create -n ga_layer_search python=3.10
conda activate ga_layer_search

# 安装依赖
pip install -r requirements.txt
```

或使用venv:

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 第3步：配置路径

编辑 `config.sh` 文件，修改conda环境路径：

```bash
# 修改为你的conda环境路径
export CONDA_ENV_PATH="/path/to/your/conda/envs/ga_layer_search"
```

### 第4步：准备模型文件

这是**最重要**的步骤！需要下载模型并提取层文件。

#### 4.1 下载模型

```bash
cd model_preparation/

# 下载Llamba模型（未对齐版本，用于替换）
python -c "from modelscope_utils import get_model_modelscope; get_model_modelscope('unaligned_llamba', is_minimal=False)"

# 下载Llama模型（基础模型，提取层）
python -c "from modelscope_utils import get_model_modelscope; get_model_modelscope('llama', is_minimal=False)"
```

**注意**：
- 首次下载需要时间（每个模型约16GB）
- 模型会保存在 `~/.cache/modelscope/` 或 `modelscope_cache/`
- 需要稳定的网络连接

#### 4.2 提取Llama层文件

```bash
cd model_preparation/

# 提取所有层和rotary_emb
python extract_layers.py \
  --model_name llama \
  --output_dir ../extracted_llama_layers

# 验证提取结果
python extract_layers.py \
  --verify \
  --output_dir ../extracted_llama_layers
```

**预期输出**：

```
extracted_llama_layers/
├── metadata.json          # 元数据
├── rotary_emb.pt         # 旋转位置编码 (~1MB)
├── layer_00.pt           # 第0层 (~1.2GB)
├── layer_01.pt
├── ...
└── layer_31.pt           # 第31层
```

- **总大小**: 约40GB
- **层数**: 32层（0-31）
- **额外文件**: rotary_emb.pt 和 metadata.json

#### 4.3 配置层文件路径

编辑 `config.sh`，设置层文件路径：

```bash
# 设置为提取的层文件目录
export LLAMA_LAYERS_DIR="${PROJECT_ROOT}/extracted_llama_layers"
```

#### 4.4 测试层替换功能

```bash
cd model_preparation/

# 测试单层替换（替换第17层）
python test_specific_combination.py --layers 17 --gpu_id 0

# 测试多层替换
python test_specific_combination.py --layers 13 17 --gpu_id 0
```

**预期输出**：

```
✅ MMLU completed: 0.5570
```

如果看到MMLU分数，说明模型准备成功！

## ✅ 验证安装

### 测试1: 环境检查

```bash
# 检查Python和依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('transformers OK')"
python -c "from lm_eval import evaluator; print('lm-eval OK')"
```

### 测试2: 路径检查

```bash
# 加载配置
source config.sh

# 检查路径
echo "Project root: ${PROJECT_ROOT}"
echo "Llama layers: ${LLAMA_LAYERS_DIR}"
ls -lh "${LLAMA_LAYERS_DIR}" | head -5
```

### 测试3: Mock函数测试

```bash
cd genetic_algorithm/

# 运行Mock测试（不需要模型文件）
python run_complete_search.py
```

应该在几分钟内完成，输出JSON结果和日志。

### 测试4: 真实MMLU测试（小规模）

```bash
cd genetic_algorithm/

# 快速测试（10个个体，5代）
python run_ga_search_real.py \
  --gpu 0 \
  --population 10 \
  --generations 5 \
  --fast-limit 10 \
  --verbose
```

应该在1-2小时内完成。

## 🔧 故障排除

### 问题1: 找不到模型文件

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: '...extracted_llama_layers/layer_00.pt'
```

**解决方案**：
1. 确认已运行 `extract_layers.py`
2. 检查 `config.sh` 中的 `LLAMA_LAYERS_DIR` 路径
3. 验证层文件确实存在：`ls extracted_llama_layers/`

### 问题2: CUDA内存不足

**症状**：
```
CUDA out of memory. Tried to allocate XXX GB
```

**解决方案**：
1. 使用显存更大的GPU
2. 确保没有其他进程占用GPU：`nvidia-smi`
3. 减小batch_size或limit参数
4. 检查是否有多个模型同时加载

### 问题3: ImportError

**症状**：
```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**：
1. 确认已激活正确的Python环境
2. 重新安装依赖：`pip install -r requirements.txt`
3. 对于特定包：`pip install xxx`

### 问题4: ModelScope下载失败

**症状**：
```
Connection timeout / Network error
```

**解决方案**：
1. 检查网络连接
2. 使用代理（如果需要）
3. 尝试多次重试
4. 手动下载模型文件

### 问题5: 层提取失败

**症状**：
```
Layer extraction failed: ...
```

**解决方案**：
1. 确保有足够的GPU显存（18GB+）
2. 确保有足够的磁盘空间（50GB+）
3. 检查GPU是否可用：`nvidia-smi`
4. 清理GPU缓存后重试

## 📚 下一步

安装完成后，可以：

1. **阅读文档**：查看 `README.md` 了解系统概述
2. **运行快速测试**：`cd scripts && ./quick_test_real.sh`
3. **运行完整搜索**：`cd scripts && ./run_full_search.sh`
4. **查看USAGE_GUIDE.md**：详细使用指南

## 💡 性能优化建议

### 1. 使用SSD

将提取的层文件存储在SSD上，可以显著加快加载速度。

### 2. 启用Ray缓存

系统会自动缓存MMLU评估结果，避免重复评估。确保缓存目录有足够空间。

### 3. 选择合适的GPU

- **开发测试**: RTX 3090 (24GB) 可用
- **生产运行**: RTX A6000 (48GB) 推荐
- **大规模实验**: A100 (80GB) 理想

### 4. 后台运行

对于长时间任务，使用tmux或screen：

```bash
# 创建tmux会话
tmux new-session -s ga_search

# 在会话中运行
cd GA_Layer_Search/scripts
./run_full_search.sh

# 断开会话：Ctrl+B, 然后按 D
# 重新连接：tmux attach -t ga_search
```

## 🆘 获取帮助

如果遇到问题：

1. **查看文档**: `README.md`, `USAGE_GUIDE.md`, `ARCHITECTURE.md`
2. **检查日志**: 查看 `results/*/search_log_*.txt`
3. **查看Issue**: 检查是否有类似问题
4. **提交Issue**: 描述问题、错误信息、环境信息

---

**更新时间**: 2025-10-15  
**版本**: v1.0

