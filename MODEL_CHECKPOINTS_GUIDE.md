# 模型Checkpoint生成和使用指南

本文档说明如何生成、保存和使用替换层后的完整模型checkpoint。

## 📋 概述

GA搜索找到的是"最优层组合"（如 `[11, 13, 17, 21]`），但这只是配置信息。要实际使用这个模型，需要：

1. 加载Llamba基础模型
2. 用指定的Llama层替换相应位置
3. 保存为完整的模型checkpoint

这样生成的checkpoint可以直接加载使用，无需每次重新替换。

## 🎯 为什么需要Checkpoint？

**问题**：GA搜索结果只是层索引列表，如 `[11, 13, 17, 21]`

**解决方案**：生成包含完整模型权重的checkpoint

**优势**：
- ✅ 即开即用，无需每次替换层
- ✅ 可直接部署到生产环境
- ✅ 方便分享和复现结果
- ✅ 支持标准的模型加载接口

## 🚀 快速开始

### 方式1: 创建单个Checkpoint

```bash
cd model_preparation/

# 创建11, 13, 17, 21层替换的checkpoint
python create_replaced_model_checkpoint.py \
    --layers 11 13 17 21 \
    --output_dir ../model_checkpoints/llamba_replaced_11_13_17_21 \
    --description "GA搜索发现的4层最优组合" \
    --score 0.5700
```

**参数说明**：
- `--layers`: 要替换的层索引
- `--output_dir`: checkpoint输出目录
- `--description`: 描述信息（可选）
- `--score`: MMLU分数（可选）
- `--llama_layers_dir`: Llama层文件目录（默认`../extracted_llama_layers`）
- `--gpu`: 使用的GPU ID（默认0）

### 方式2: 批量创建所有最优Checkpoint

```bash
cd model_preparation/

# 批量创建所有GA发现的最优组合
./create_best_checkpoints.sh
```

这将自动创建以下checkpoint：
- `llamba_replaced_11_13_17_21` - 4层最优（0.5700）
- `llamba_replaced_13_16_17` - 3层最优（0.6542）
- `llamba_replaced_13_17` - 2层最优（0.5544）
- `llamba_replaced_17` - 1层最优（0.5144）
- 其他备选组合...

## 📂 Checkpoint结构

生成的每个checkpoint包含：

```
llamba_replaced_11_13_17_21/
├── model.pt                # 完整模型（~16GB）⭐
├── model_state_dict.pt     # 模型state dict（~16GB）
├── tokenizer/              # Tokenizer文件
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── ...
├── checkpoint_info.json    # 元数据信息
└── README.txt              # 使用说明
```

### 文件说明

| 文件 | 大小 | 用途 |
|------|------|------|
| `model.pt` | ~16GB | 完整模型，可直接加载 ⭐ |
| `model_state_dict.pt` | ~16GB | State dict，需要模型架构 |
| `tokenizer/` | ~2MB | Tokenizer文件 |
| `checkpoint_info.json` | ~1KB | 元数据（层配置、分数等） |
| `README.txt` | ~1KB | 使用说明 |

**推荐使用 `model.pt`**：最简单，直接 `torch.load()` 即可。

## 💻 使用Checkpoint

### 方法1: 直接加载完整模型（推荐）

```python
import torch

# 加载模型
model = torch.load('model_checkpoints/llamba_replaced_11_13_17_21/model.pt')
model.eval()
model = model.cuda()  # 移到GPU

# 加载tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('model_checkpoints/llamba_replaced_11_13_17_21/tokenizer')

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
    
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

### 方法2: 从State Dict加载

```python
import torch
from modelscope_utils import get_model_modelscope

# 加载模型架构
model, tokenizer, _, _ = get_model_modelscope('unaligned_llamba')

# 加载权重
state_dict = torch.load('model_checkpoints/llamba_replaced_11_13_17_21/model_state_dict.pt')
model.load_state_dict(state_dict)

model.eval()
model = model.cuda()
```

### 方法3: 查看Checkpoint信息

```python
import json

# 读取元数据
with open('model_checkpoints/llamba_replaced_11_13_17_21/checkpoint_info.json') as f:
    info = json.load(f)

print(f"Replaced layers: {info['replaced_layers']}")
print(f"MMLU score: {info.get('mmlu_score', 'N/A')}")
print(f"Creation time: {info['creation_time']}")
```

## 🧪 测试Checkpoint

### 快速测试（推理）

```bash
cd model_preparation/

# 测试基本推理功能
python test_checkpoint.py --checkpoint ../model_checkpoints/llamba_replaced_11_13_17_21
```

### 完整MMLU评估

```bash
# 快速评估（limit=100）
python test_checkpoint.py \
    --checkpoint ../model_checkpoints/llamba_replaced_11_13_17_21 \
    --full_eval \
    --limit 100

# 完整MMLU评估（无limit）
python test_checkpoint.py \
    --checkpoint ../model_checkpoints/llamba_replaced_11_13_17_21 \
    --full_eval
```

## 📊 示例：基于GA搜索结果创建Checkpoint

假设GA搜索发现最优组合是 `[11, 13, 17, 21]`，MMLU分数 `0.5700`：

```bash
cd model_preparation/

# 1. 创建checkpoint
python create_replaced_model_checkpoint.py \
    --layers 11 13 17 21 \
    --output_dir ../model_checkpoints/best_4layer \
    --description "GA搜索发现的4层最优组合" \
    --score 0.5700 \
    --gpu 0

# 2. 测试checkpoint
python test_checkpoint.py \
    --checkpoint ../model_checkpoints/best_4layer \
    --full_eval \
    --limit 50

# 3. 使用checkpoint
python -c "
import torch
model = torch.load('../model_checkpoints/best_4layer/model.pt')
print('✅ Model loaded successfully!')
print(f'Device: {next(model.parameters()).device}')
print(f'Total parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
"
```

## 💾 存储和管理

### 磁盘空间需求

- 单个checkpoint: ~16GB
- 5个最优组合: ~80GB
- 建议预留: 100GB+

### 目录组织

推荐的目录结构：

```
GA_Layer_Search/
├── extracted_llama_layers/      # Llama层文件（~40GB）
├── model_checkpoints/           # 生成的checkpoint
│   ├── llamba_replaced_11_13_17_21/  # ~16GB
│   ├── llamba_replaced_13_16_17/     # ~16GB
│   ├── llamba_replaced_13_17/        # ~16GB
│   └── ...
└── model_preparation/           # 工具脚本
    ├── create_replaced_model_checkpoint.py
    ├── create_best_checkpoints.sh
    └── test_checkpoint.py
```

### Git管理

**重要**：Checkpoint文件很大（~16GB），不应提交到Git。

`.gitignore` 已配置忽略：
```gitignore
model_checkpoints/
*.pt
*.pth
```

### 分享Checkpoint

如果需要分享checkpoint：

**选项1: 云存储**
```bash
# 压缩checkpoint
tar -czf llamba_replaced_11_13_17_21.tar.gz model_checkpoints/llamba_replaced_11_13_17_21/

# 上传到云存储（Google Drive, 百度云等）
# 接收方下载后解压即可使用
```

**选项2: 提供生成命令**
```bash
# 在README中提供命令，让用户自己生成
python create_replaced_model_checkpoint.py --layers 11 13 17 21 --output_dir ./checkpoints/best
```

## 🔧 高级用法

### 自定义层组合

```bash
# 创建自己的层组合
python create_replaced_model_checkpoint.py \
    --layers 5 10 15 20 25 30 \
    --output_dir ../model_checkpoints/custom_6layer \
    --description "自定义6层均匀分布" \
    --gpu 1
```

### 批量创建特定配置

```bash
# 创建2层、3层、4层的所有top组合
COMBINATIONS=(
    "13 17"
    "14 17"
    "13 16 17"
    "11 13 17 21"
    "10 14 17 30"
)

for layers in "${COMBINATIONS[@]}"; do
    layers_name=$(echo $layers | tr ' ' '_')
    python create_replaced_model_checkpoint.py \
        --layers $layers \
        --output_dir ../model_checkpoints/combo_${layers_name} \
        --gpu 0
done
```

### 在其他项目中使用

```python
# 在你的项目中加载checkpoint
import torch
import sys
sys.path.append('/path/to/GA_Layer_Search/model_preparation')

# 加载模型
MODEL_PATH = '/path/to/GA_Layer_Search/model_checkpoints/llamba_replaced_11_13_17_21/model.pt'
model = torch.load(MODEL_PATH)
model.eval()

# 正常使用
# ...
```

## 📝 最佳实践

### 1. 命名规范

使用清晰的命名：
- `llamba_replaced_<layers>` - 基本格式
- `llamba_replaced_11_13_17_21` - 4层组合
- `best_4layer` - 简短别名

### 2. 元数据记录

始终记录：
- 替换的层索引
- MMLU分数
- 创建时间
- 描述信息

### 3. 版本控制

为不同版本的checkpoint添加标识：
- `llamba_replaced_11_13_17_21_v1`
- `llamba_replaced_11_13_17_21_v2_finetuned`

### 4. 测试验证

创建后立即测试：
```bash
python test_checkpoint.py --checkpoint <path> --full_eval --limit 50
```

## ⚠️ 注意事项

1. **GPU内存**: 创建checkpoint需要~20GB GPU显存
2. **磁盘空间**: 每个checkpoint约16GB
3. **时间成本**: 创建单个checkpoint约5-10分钟
4. **依赖要求**: 需要已提取的Llama层文件
5. **一致性**: 确保使用相同版本的模型和层文件

## 🆘 故障排除

### 问题1: 找不到层文件

```
FileNotFoundError: Layer file not found: .../layer_11.pt
```

**解决**：
```bash
cd model_preparation/
python extract_layers.py --model_name llama --output_dir ../extracted_llama_layers
```

### 问题2: GPU内存不足

```
CUDA out of memory
```

**解决**：
- 使用显存更大的GPU
- 关闭其他占用GPU的进程
- 使用 `--gpu` 参数指定空闲GPU

### 问题3: 加载checkpoint失败

```
RuntimeError: Error loading model.pt
```

**解决**：
- 检查PyTorch版本兼容性
- 尝试加载 `model_state_dict.pt` 而非 `model.pt`
- 确认checkpoint完整性（文件大小正常）

## 📚 参考资料

- **创建工具**: `create_replaced_model_checkpoint.py`
- **测试工具**: `test_checkpoint.py`
- **批量脚本**: `create_best_checkpoints.sh`
- **模型准备**: `SETUP.md`
- **项目主文档**: `README.md`

---

**更新时间**: 2025-10-15  
**版本**: v1.0

