# GA Layer Search

> 遗传算法驱动的LLM层替换优化搜索系统

---

## 🎯 这是什么？

从Llama模型中智能选择最优的几层，替换到Llamba模型中以提升MMLU性能。

**核心思路**：
```
GA粗搜索（快速评估）→ 完整评估top候选 → 局部精炼 → 最优层组合
```

**搜索结果**（快速测试，limit=10/50）：

| 排名 | 层组合 | MMLU | 层数 |
|------|---------|------|------|
| 1 | [11, 13, 17, 29] | 0.5877 | 4层 |
| 2 | [9, 13, 14, 17] | 0.5751 | 4层 |
| 3 | [8, 13, 14, 17] | 0.5737 | 4层 |
| 4 | [12, 14, 17, 25] | 0.5649 | 4层 |
| 5 | [13, 14, 17] | 0.5628 | 3层 |

---

## 🚀 快速开始

### 一键部署（推荐）

```bash
git clone <repo> GA_Layer_Search
cd GA_Layer_Search

# 复制models目录（仅需一次）
cp -r /path/to/original/Gather-and-Aggregate/models ./models

# 一键部署（自动完成环境、模型、层提取、测试）
./DEPLOY_TEST.sh
```

**完成时间**：30-60分钟（主要是模型下载）

### 手动部署

```bash
# 1. 环境
conda create -n ga_layer_search python=3.10
conda activate ga_layer_search
pip install -r requirements.txt

# 2. 复制models
cp -r /path/to/original/models ./models

# 3. 提取Llama层
cd model_preparation
python extract_layers.py --model_name llama --output_dir ../extracted_llama_layers

# 4. 测试
python test_specific_combination.py --layers 17 --gpu_id 0 --limit 10
```

---

## 📦 生成可部署的模型Checkpoint

### 为什么需要？

GA搜索结果是**层索引**（如`[11, 13, 17, 29]`）。生成checkpoint将其转换为**完整模型**（~16GB），可直接加载使用。

### 生成Checkpoint

```bash
cd model_preparation/

# 单个checkpoint
python create_replaced_model_checkpoint.py \
    --layers 11 13 17 29 \
    --output_dir ../model_checkpoints/best_4layer \
    --gpu 0

# 或批量生成所有top组合
./create_best_checkpoints.sh
```

**时间**：5-10分钟/个  
**大小**：~16GB/个

### 使用Checkpoint

```python
import torch
from transformers import AutoTokenizer

# 加载
model = torch.load('model_checkpoints/best_4layer/model.pt')
tokenizer = AutoTokenizer.from_pretrained('model_checkpoints/best_4layer/tokenizer')

# 使用
model.eval().cuda()
inputs = tokenizer("Hello", return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 测试Checkpoint

```bash
cd model_preparation/
python test_checkpoint.py --checkpoint ../model_checkpoints/best_4layer --full_eval --limit 100
```

---

## 🔬 运行GA搜索（可选）

如果想自己搜索：

```bash
# 快速测试（~5小时）
cd scripts && ./quick_test_real.sh

# 完整搜索（~3-5天）
cd scripts && ./run_full_search_fast.sh
```

---

## 📂 项目结构

```
GA_Layer_Search/
├── genetic_algorithm/       # GA搜索代码
├── model_preparation/       # 模型工具（下载、层提取、checkpoint生成）
├── scripts/                 # 运行脚本
├── models/                  # Llamba模型代码（需复制）
├── config.sh               # 配置
├── DEPLOY_TEST.sh          # 一键部署
└── requirements.txt        # 依赖

运行时生成：
├── extracted_llama_layers/ # ~40GB
├── modelscope_cache/       # ~30GB
└── model_checkpoints/      # ~16GB/个
```

---

## 🛠️ 系统要求

- **GPU**: 18GB+ 显存（推荐RTX A6000）
- **硬盘**: 100GB+ 可用空间
- **系统**: Linux + Python 3.10 + CUDA

---

## 💡 核心算法

### 三阶段搜索

1. **阶段1：GA粗搜索** - 智能初始化 + 模式挖掘 + 快速评估
2. **阶段2：完整评估** - 对top-20候选进行完整MMLU评估
3. **阶段3：局部精炼** - 两阶段局部搜索（粗筛选+完整验证）

**效率**：节省80%+时间（vs全完整评估）

---

## 📚 详细文档

- **SETUP.md** - 详细安装指南
- **MODEL_CHECKPOINTS_GUIDE.md** - Checkpoint完整指南
- **CHECKPOINT_QUICKSTART.txt** - 快速命令参考
- **ARCHITECTURE.md** - 技术架构（开发者）

---

## 📧 联系

- **作者**: Zhuangfei Hu
- **版本**: v1.0
- **日期**: 2025-10-15

---

**快速命令总结**：

```bash
./DEPLOY_TEST.sh                           # 1. 部署
cd model_preparation && \                   # 2. 生成checkpoint
  python create_replaced_model_checkpoint.py --layers 11 13 17 29 --output_dir ../model_checkpoints/best
python -c "import torch; \                  # 3. 使用
  model=torch.load('model_checkpoints/best/model.pt')"
```

**就这么简单！** 🚀
