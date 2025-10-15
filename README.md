# 遗传算法层替换搜索系统

> 基于遗传算法的大语言模型层替换优化搜索，高效找到最优Llama层替换组合。

**核心成果**：通过GA搜索找到最优层组合，并生成可直接部署的模型checkpoint。

---

## 🎯 项目概述

### 问题
如何从Llama模型中选择最优的几层，替换到Llamba模型中以提升性能？

### 解决方案
本项目实现了一个**三阶段遗传算法搜索框架**：

```
阶段1: GA粗搜索 (快速评估)
   ↓
阶段2: 完整评估top候选
   ↓
阶段3: 局部精细优化
   ↓
最优层组合 + 可部署checkpoint
```

### 核心特性

- **智能搜索**：结合模式挖掘、引导变异、两阶段局部搜索
- **高效评估**：快速评估筛选 + 完整评估验证，节省80%+时间
- **即用checkpoint**：一键生成包含完整权重的模型checkpoint
- **自动化部署**：一键脚本完成环境、模型、测试全流程

### 搜索结果示例（快速测试）

| 排名 | 层组合 | MMLU分数 | 层数 |
|------|---------|----------|------|
| 1 | [11, 13, 17, 29] | 0.5877 | 4层 |
| 2 | [9, 13, 14, 17] | 0.5751 | 4层 |
| 3 | [8, 13, 14, 17] | 0.5737 | 4层 |
| 4 | [12, 14, 17, 25] | 0.5649 | 4层 |
| 5 | [13, 14, 17] | 0.5628 | 3层 |

*注：基于快速测试结果（limit=10/50），完整搜索结果可能更优*

---

## 🚀 快速开始

### 方式1：一键部署（推荐）

```bash
# 克隆项目
git clone <repository> GA_Layer_Search
cd GA_Layer_Search

# 运行自动化部署（包含环境、模型、测试）
./DEPLOY_TEST.sh
```

**完成！** 脚本自动完成：
- 创建conda环境 `ga_layer_search`
- 安装所有依赖
- 下载模型（Llamba + Llama）
- 提取Llama层文件
- 测试所有功能

**时间**：首次约30-60分钟（主要是模型下载）

### 方式2：手动部署

```bash
# 1. 创建环境
conda create -n ga_layer_search python=3.10
conda activate ga_layer_search
pip install -r requirements.txt

# 2. 准备models目录（复制或让脚本自动处理）
cp -r /path/to/Gather-and-Aggregate/models ./

# 3. 提取Llama层文件
cd model_preparation
python extract_layers.py --model_name llama --output_dir ../extracted_llama_layers

# 4. 验证
python test_specific_combination.py --layers 17 --gpu_id 0 --limit 10
```

---

## 📦 生成可部署的模型Checkpoint

### 什么是Checkpoint？

GA搜索的结果是**层索引**（如`[11, 13, 17, 21]`），但要实际使用模型，需要生成包含完整权重的checkpoint。

### 生成单个Checkpoint

```bash
cd model_preparation/

# 生成最优4层组合的checkpoint
python create_replaced_model_checkpoint.py \
    --layers 11 13 17 21 \
    --output_dir ../model_checkpoints/best_4layer \
    --description "GA搜索最优4层组合" \
    --score 0.5700 \
    --gpu 0
```

**输出**（~16GB）：
- `model.pt` - 完整模型（可直接`torch.load()`）
- `tokenizer/` - Tokenizer文件
- `checkpoint_info.json` - 元数据（层配置、分数等）

### 批量生成所有最优Checkpoint

```bash
cd model_preparation/
./create_best_checkpoints.sh
```

自动生成5个最优组合的checkpoint（1-4层）

### 使用Checkpoint

```python
import torch
from transformers import AutoTokenizer

# 加载模型
model = torch.load('model_checkpoints/best_4layer/model.pt')
model.eval().cuda()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained('model_checkpoints/best_4layer/tokenizer')

# 推理
inputs = tokenizer("Hello world", return_tensors="pt").cuda()
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 测试Checkpoint

```bash
cd model_preparation/

# 快速测试
python test_checkpoint.py --checkpoint ../model_checkpoints/best_4layer

# MMLU评估
python test_checkpoint.py \
    --checkpoint ../model_checkpoints/best_4layer \
    --full_eval \
    --limit 100
```

---

## 🔬 运行GA搜索（可选）

如果想自己运行搜索找到新的最优组合：

### 快速测试（~4-5小时）

```bash
cd scripts/
./quick_test_real.sh
```

配置：种群20，代数15，limit=10/50

### 完整搜索（~3-5天）

```bash
cd scripts/
./run_full_search_fast.sh
```

配置：种群40，代数20，limit=20/None

### 结果查看

```bash
# 查看日志
tail -f results/real_*/search_log_*.txt

# 查看最终结果
cat results/real_*/search_result_*.json | python -m json.tool
```

---

## 📂 项目结构

```
GA_Layer_Search/
├── genetic_algorithm/       # GA搜索核心代码
├── model_preparation/       # 模型准备和checkpoint工具
│   ├── extract_layers.py   # 提取Llama层
│   ├── create_replaced_model_checkpoint.py  # ⭐ 生成checkpoint
│   ├── test_checkpoint.py  # 测试checkpoint
│   └── create_best_checkpoints.sh  # 批量生成
├── scripts/                 # 运行脚本
├── config.sh               # 配置文件
├── DEPLOY_TEST.sh          # 一键部署脚本
└── requirements.txt        # Python依赖

运行时生成（不在Git中）：
├── extracted_llama_layers/  # Llama层文件（~40GB）
├── model_checkpoints/       # 生成的checkpoint（~16GB/个）
├── modelscope_cache/        # 模型缓存
└── results/                 # 搜索结果
```

---

## 💡 核心算法

### 三阶段搜索流程

1. **阶段1：GA粗搜索**
   - 智能初始化（精英种子 + 启发式 + 随机）
   - 模式挖掘和引导变异
   - 快速评估（limit=20-50）探索搜索空间
   - 输出：top-20候选

2. **阶段2：完整评估**
   - 对top-20候选进行完整MMLU评估
   - 去除噪音，找到真实top-10

3. **阶段3：局部精炼**
   - 两阶段局部搜索（粗评估筛选 + 完整评估验证）
   - 对top-3进行邻域搜索
   - 确认局部最优

### 效率优势

- 传统方法：~1400次完整评估，~580小时（24天）
- **本方法**：~1400次快速评估 + ~50次完整评估，~150小时（6天）
- **节省时间：80%+**

---

## 🛠️ 系统要求

### 硬件
- **GPU**: 18GB+ 显存（推荐RTX A6000 48GB）
- **CPU**: 多核CPU
- **内存**: 32GB+ RAM
- **硬盘**: 100GB+ 可用空间

### 软件
- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: 3.10
- **CUDA**: 11.x or 12.x
- **Conda**: Miniconda或Anaconda

---

## 📚 详细文档

- **SETUP.md** - 详细安装配置指南
- **MODEL_CHECKPOINTS_GUIDE.md** - Checkpoint生成和使用完整指南
- **CHECKPOINT_QUICKSTART.txt** - 快速命令参考
- **DEPLOYMENT_READY.md** - 部署检查清单
- **ARCHITECTURE.md** - 技术架构（开发者）

---

## ❓ 常见问题

### Q: 首次部署需要多久？
A: 30-60分钟（主要是下载模型和提取层文件）。使用`./DEPLOY_TEST.sh`一键完成。

### Q: 生成一个checkpoint需要多久？
A: 5-10分钟，生成约16GB的完整模型文件。

### Q: 必须运行GA搜索吗？
A: 不必须。可以直接使用我们提供的最优层组合生成checkpoint。

### Q: 如何分享checkpoint？
A: Checkpoint是标准的PyTorch模型文件，可以直接复制或打包分享。

### Q: 支持其他模型吗？
A: 目前针对Llamba/Llama，但框架可以迁移到其他模型。

---

## 🔍 故障排除

### 问题：找不到models目录
```bash
# 解决：从原项目复制或让DEPLOY_TEST.sh自动处理
cp -r /path/to/Gather-and-Aggregate/models ./
```

### 问题：CUDA OOM
```bash
# 解决：使用显存更大的GPU，或减小batch_size
python xxx.py --gpu 1  # 尝试其他GPU
```

### 问题：层文件不存在
```bash
# 解决：运行层提取脚本
cd model_preparation
python extract_layers.py --model_name llama --output_dir ../extracted_llama_layers
```

---

## 📄 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@software{ga_layer_search_2025,
  title={GA Layer Search: Genetic Algorithm for LLM Layer Replacement Optimization},
  author={Hu, Zhuangfei},
  year={2025},
  url={https://github.com/...}
}
```

---

## 📧 联系方式

- **作者**: Zhuangfei Hu
- **邮箱**: [您的邮箱]
- **项目**: GA_Layer_Search
- **版本**: v1.0
- **最后更新**: 2025-10-15

---

## 📝 许可证

MIT License

---

**🎯 核心流程总结**：

```bash
# 1. 一键部署
./DEPLOY_TEST.sh

# 2. 生成checkpoint
cd model_preparation
python create_replaced_model_checkpoint.py --layers 11 13 17 21 --output_dir ../model_checkpoints/best

# 3. 使用checkpoint
python -c "import torch; model = torch.load('model_checkpoints/best/model.pt'); print('✅ Ready!')"
```

**就这么简单！** 🚀
