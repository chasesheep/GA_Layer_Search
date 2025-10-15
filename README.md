# GA Layer Search

> 遗传算法驱动的LLM层替换优化搜索系统

---

## 🎯 项目简介

从Llama模型中智能选择最优的几层，替换到Llamba模型中以提升MMLU性能。

**搜索流程**：
```
GA粗搜索（快速评估）→ 完整评估top候选 → 局部精炼 → 最优层组合
```

**搜索结果示例**（快速测试）：

| 排名 | 层组合 | MMLU分数 | 层数 |
|------|---------|----------|------|
| 1 | [11, 13, 17, 29] | 0.5877 | 4层 |
| 2 | [9, 13, 14, 17] | 0.5751 | 4层 |
| 3 | [8, 13, 14, 17] | 0.5737 | 4层 |
| 4 | [12, 14, 17, 25] | 0.5649 | 4层 |
| 5 | [13, 14, 17] | 0.5628 | 3层 |

---

## 🚀 快速开始

### 一键部署

```bash
git clone <repository> GA_Layer_Search
cd GA_Layer_Search
./DEPLOY_TEST.sh
```

**完成时间**：30-60分钟（首次需下载模型）

**脚本自动完成**：
- 创建conda环境 `ga_layer_search`
- 安装所有Python依赖
- 下载模型（Llamba + Llama）
- 提取Llama层文件（~40GB）
- 运行完整测试验证

---

## 📦 生成可部署的模型Checkpoint

### 为什么需要Checkpoint？

GA搜索结果是层索引（如`[11, 13, 17, 29]`）。Checkpoint将其转换为完整模型（~16GB），可直接部署使用。

### 生成Checkpoint

```bash
cd model_preparation/

# 单个checkpoint
python create_replaced_model_checkpoint.py \
    --layers 11 13 17 29 \
    --output_dir ../model_checkpoints/best_4layer

# 批量生成所有top组合
./create_best_checkpoints.sh
```

**时间**: 5-10分钟/个 | **大小**: ~16GB/个

### 使用Checkpoint

```python
import torch
import sys
sys.path.append('/path/to/GA_Layer_Search')
from model_preparation.modelscope_utils import get_model_modelscope
from transformers import AutoTokenizer

# 加载模型架构
model, _, _, _ = get_model_modelscope('unaligned_llamba')

# 加载checkpoint权重
state_dict = torch.load('model_checkpoints/best_4layer/model_state_dict.pt')
model.load_state_dict(state_dict)
model = model.cuda().eval()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('model_checkpoints/best_4layer/tokenizer')

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**注**：使用`model_state_dict.pt`（权重文件），不是`model.pt`。

### 测试Checkpoint

```bash
python test_checkpoint.py --checkpoint ../model_checkpoints/best_4layer --full_eval --limit 100
```

---

## 🔬 运行GA搜索（可选）

```bash
# 快速测试（~5小时）
cd scripts && ./quick_test_real.sh

# 完整搜索（~3-5天）
cd scripts && ./run_full_search_fast.sh

# 查看结果
tail -f results/*/search_log_*.txt
```

---

## 📂 项目结构

```
GA_Layer_Search/
├── genetic_algorithm/       # GA核心代码
├── model_preparation/       # 模型工具
│   ├── extract_layers.py   # 提取Llama层
│   ├── create_replaced_model_checkpoint.py  # 生成checkpoint
│   └── test_checkpoint.py  # 测试checkpoint
├── scripts/                 # 运行脚本
├── models/                  # Llamba模型代码（已在Git，~300KB）
├── DEPLOY_TEST.sh          # 一键部署脚本
└── requirements.txt        # Python依赖

自动生成：
├── modelscope_cache/       # 模型缓存（~30GB）
├── extracted_llama_layers/ # Llama层（~40GB）
└── model_checkpoints/      # 生成的checkpoint（~16GB/个）
```

---

## 💡 核心算法

**三阶段搜索**：
1. GA粗搜索 - 智能初始化 + 模式挖掘 + 快速评估
2. 完整评估 - 对top候选完整MMLU评估
3. 局部精炼 - 两阶段局部搜索

**效率**：节省80%+时间（vs全完整评估）

---

## 🛠️ 系统要求

- **GPU**: 18GB+ 显存（推荐RTX A6000 48GB）
- **硬盘**: 100GB+ 可用空间
- **系统**: Linux + Python 3.10 + CUDA

---

## 📚 详细文档

- **SETUP.md** - 安装配置指南
- **MODEL_CHECKPOINTS_GUIDE.md** - Checkpoint完整指南
- **ARCHITECTURE.md** - 技术架构

---

**作者**: Zhuangfei Hu | **版本**: v1.0 | **日期**: 2025-10-15

---

**核心命令**：

```bash
./DEPLOY_TEST.sh                     # 部署
cd model_preparation && \             # 生成checkpoint
  python create_replaced_model_checkpoint.py --layers 11 13 17 29 --output_dir ../model_checkpoints/best
```
