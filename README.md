# 遗传算法层替换搜索系统

基于遗传算法的大语言模型层替换优化搜索系统，用于寻找Llamba模型的最优Llama层替换组合。

## 📚 项目概述

本项目实现了一个高效的遗传算法搜索框架，用于优化大语言模型的层替换策略。通过智能初始化、模式挖掘、引导变异和两阶段局部搜索，系统能够在大规模搜索空间中高效找到最优层组合。

### 核心特性

- **智能初始化**：结合已知优解、启发式构建和随机探索
- **动态模式挖掘**：自动发现并利用高质量层组合模式
- **模式引导变异**：使用发现的模式指导进化方向
- **三阶段搜索流程**：GA粗搜索 → 完整评估 → 局部精炼
- **两阶段局部搜索**：粗评估筛选 + 完整评估验证，节省90%+评估成本
- **完善的检查点机制**：自动保存进度，支持中断恢复
- **详细的日志记录**：完整记录搜索过程和发现的模式

## 📁 项目结构

```
GA_Layer_Search/
├── genetic_algorithm/          # 遗传算法核心模块
│   ├── config.py              # 配置参数
│   ├── individual.py          # 个体表示
│   ├── population.py          # 种群管理（智能初始化）
│   ├── operators.py           # 基础遗传算子
│   ├── operators_guided.py    # 模式引导算子
│   ├── pattern_miner.py       # 模式挖掘
│   ├── ga_core.py             # GA主流程
│   ├── local_search.py        # 单阶段局部搜索
│   ├── local_search_twostage.py  # 两阶段局部搜索
│   ├── fitness.py             # Mock适应度函数（测试用）
│   ├── real_fitness.py        # 真实MMLU评估函数
│   ├── utils.py               # 工具函数
│   ├── run_complete_search.py # 完整搜索流程
│   └── run_ga_search_real.py  # 命令行接口
│
├── model_preparation/          # 模型准备模块
│   ├── modelscope_utils.py    # ModelScope模型下载和评估工具
│   ├── extract_layers.py      # 层抽取脚本
│   └── test_specific_combination.py  # 测试特定层组合
│
├── scripts/                    # 运行和监控脚本
│   ├── quick_test_real.sh     # 快速测试脚本
│   ├── run_full_search.sh     # 完整搜索脚本
│   ├── run_full_search_fast.sh # 快速完整搜索
│   ├── check_tmux.sh          # 检查tmux会话
│   ├── monitor_progress.sh    # 监控进度
│   ├── check_results.sh       # 查看结果
│   └── single_layer_results.json  # 单层替换基准数据
│
├── 文档
│   ├── README.md              # 本文件（主文档）
│   ├── START_HERE.md          # 快速开始指南
│   ├── ARCHITECTURE.md        # 架构设计文档
│   ├── USAGE_GUIDE.md         # 详细使用指南
│   ├── SUMMARY.md             # 项目总结
│   └── FILES_INDEX.md         # 文件索引
│
├── requirements.txt            # Python依赖
└── .gitignore                 # Git忽略配置
```

## 🚀 快速开始

### 第0步：环境准备

```bash
# 创建conda环境
conda create -n ganda_new python=3.10
conda activate ganda_new

# 安装依赖
pip install -r requirements.txt
```

**主要依赖**：
- PyTorch (CUDA支持)
- transformers
- lm-eval (lm-evaluation-harness)
- modelscope
- numpy

### 第1步：准备模型文件

**重要**：在运行遗传算法搜索之前，需要先下载模型并提取层文件。

#### 1.1 下载模型

使用 `modelscope_utils.py` 中的工具从ModelScope下载模型：

```bash
cd model_preparation/

# 下载Llamba模型（未对齐版本）
python -c "from modelscope_utils import get_model_modelscope; get_model_modelscope('unaligned_llamba')"

# 下载Llama模型
python -c "from modelscope_utils import get_model_modelscope; get_model_modelscope('llama')"
```

模型将自动下载到缓存目录（通常是 `~/.cache/modelscope/`）。

#### 1.2 提取Llama层文件

为了避免重复加载完整模型，需要预先提取Llama模型的每一层：

```bash
# 提取Llama模型的所有层和rotary_emb
python extract_layers.py --model_name llama --output_dir ./extracted_llama_layers

# 验证提取的层文件
python extract_layers.py --verify --output_dir ./extracted_llama_layers
```

**提取后的文件结构**：
```
extracted_llama_layers/
├── metadata.json          # 元数据（层数、参数等）
├── rotary_emb.pt         # 旋转位置编码
├── layer_00.pt           # 第0层
├── layer_01.pt           # 第1层
├── ...
└── layer_31.pt           # 第31层（共32层）
```

**注意**：
- 层文件总大小约 **~40GB**，确保有足够磁盘空间
- 提取过程需要 **~18GB GPU显存**
- 这些文件已在 `.gitignore` 中排除，不会提交到Git

#### 1.3 测试层替换功能

确认模型准备正确：

```bash
# 测试特定层组合（例如：替换第13和17层）
python test_specific_combination.py --layers 13 17 --gpu_id 0

# 测试4层组合
python test_specific_combination.py --layers 10 14 17 30 --gpu_id 0
```

### 第2步：运行遗传算法搜索

#### 2.1 快速测试（Mock函数）

使用解析式Mock函数快速验证算法逻辑：

```bash
cd genetic_algorithm/
python run_complete_search.py
```

输出：
- JSON结果：`results/mock_results/search_result_*.json`
- 日志文件：`results/mock_results/search_log_*.txt`

#### 2.2 真实MMLU搜索

**基本用法**：

```bash
cd genetic_algorithm/

# 使用默认参数
python run_ga_search_real.py

# 指定详细参数
python run_ga_search_real.py \
  --gpu 3 \
  --fast-limit 20 \
  --full-limit None \
  --population 40 \
  --generations 20 \
  --no-improve 6 \
  --output-dir results/real_results \
  --verbose
```

**参数说明**：
- `--gpu`: GPU ID（默认3）
- `--fast-limit`: 阶段1和3的快速评估limit（默认50，设为20更快）
- `--full-limit`: 阶段2的完整评估limit（默认None=完整MMLU）
- `--population`: 种群大小（默认40）
- `--generations`: 最大代数（默认20）
- `--no-improve`: 无改进终止阈值（默认6代）
- `--output-dir`: 输出目录
- `--verbose`: 打印详细信息

**使用Shell脚本**（推荐）：

```bash
cd scripts/

# 快速测试（limit=10/50，小种群，GPU 3）
./quick_test_real.sh

# 完整搜索（limit=20/None，大种群，GPU 4）
./run_full_search_fast.sh
```

**后台运行**（长时间任务）：

```bash
# 使用tmux（推荐，支持断开连接）
tmux new-session -s ga_search "cd genetic_algorithm && python run_ga_search_real.py --gpu 3 --verbose"

# 或使用nohup
nohup python run_ga_search_real.py --gpu 3 --verbose > search.log 2>&1 &
```

### 第3步：监控和管理

#### 监控运行进度

```bash
cd scripts/

# 检查tmux会话状态
./check_tmux.sh

# 查看最新日志
tail -f ../genetic_algorithm/results/real_results/search_log_*.txt

# 监控检查点
./monitor_progress.sh ../genetic_algorithm/results/real_results
```

#### 查看结果

```bash
# 快速查看结果摘要
./check_results.sh

# 查看完整JSON结果
cat ../genetic_algorithm/results/real_results/search_result_*.json | python -m json.tool
```

## 📊 搜索流程详解

### 三阶段搜索策略

```
┌──────────────────────────────────────────────────────────────┐
│ 阶段1: GA粗搜索 (limit=20-50, 快速探索)                      │
│ ────────────────────────────────────────────────────────────│
│ • 智能初始化40个个体（精英种子+启发式+随机）                  │
│ • 演化20-30代                                                │
│ • 动态模式挖掘（每5代更新）                                   │
│ • 模式引导变异（概率从20%→60%）                              │
│ • 输出: top-20候选解                                         │
│                                                              │
│ 评估次数: ~400-500次 (limit=20-50)                          │
│ 预计时间: ~30-60小时                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 阶段2: 完整评估候选 (limit=None, 完整MMLU)                   │
│ ────────────────────────────────────────────────────────────│
│ • 对20个候选进行完整MMLU评估                                 │
│ • 去除噪音，找到真实top-10                                   │
│                                                              │
│ 评估次数: 20次 (limit=None)                                 │
│ 预计时间: ~10-12小时                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 阶段3: 局部精细优化 (两阶段局部搜索)                         │
│ ────────────────────────────────────────────────────────────│
│ • 对top-3进行邻域搜索（add/delete/replace操作）              │
│ • 粗评估筛选邻居 (limit=20-50)                               │
│ • 完整评估验证有希望的候选 (limit=None)                      │
│ • 确认局部最优解                                             │
│                                                              │
│ 评估次数:                                                    │
│   - 粗评估: ~600-900次 (limit=20-50)                        │
│   - 完整评估: ~20-30次 (limit=None)                         │
│ 预计时间: ~60-100小时                                        │
└──────────────────────────────────────────────────────────────┘

总计时间: ~100-170小时 (约4-7天，取决于缓存命中率)
总评估次数: ~1000-1400次快速 + ~40-50次完整
```

### 效率优势

传统方法（全部完整评估）：
- 评估次数：~1000-1400次完整评估
- 预计时间：~580-780小时（约24-32天）

**本系统（三阶段策略）**：
- 快速评估：~1000-1400次（低成本）
- 完整评估：~40-50次（高成本）
- **时间节省：80-85%** ✨

## 💾 检查点和恢复

### 自动检查点

系统每3代自动保存检查点，包含：
- 当前代数和评估计数
- 全局最优解和各层数最优解
- 种群Top-10个体
- 发现的模式（1/2/3层）
- 统计信息（层覆盖率、无改进次数等）

检查点位置：`results/*/checkpoints_YYYYMMDD_HHMMSS/checkpoint_genXXX.json`

### 查看检查点

```bash
cd genetic_algorithm/

# 列出所有检查点
python view_checkpoint.py results/real_results/checkpoints_20251014_083000/ --list

# 查看最新检查点
python view_checkpoint.py results/real_results/checkpoints_20251014_083000/

# 查看特定检查点
python view_checkpoint.py results/real_results/checkpoints_20251014_083000/checkpoint_gen006.json
```

## 📈 结果分析

### 输出文件

1. **JSON结果** (`search_result_*.json`)：
   - 最优层组合（Top-10）
   - 各层数最优解（2层、3层、4层）
   - 发现的模式（1/2/3层）
   - 统计信息（评估次数、时间等）

2. **日志文件** (`search_log_*.txt`)：
   - 完整搜索过程
   - 每代进化详情
   - 模式更新记录
   - 最终结果摘要

3. **检查点** (`checkpoints_*/checkpoint_gen*.json`)：
   - 中间状态快照
   - 用于恢复和分析

### 示例结果

```json
{
  "final_results": [
    {"layers": [13, 16, 17], "fitness": 0.6542, "num_layers": 3},
    {"layers": [13, 17], "fitness": 0.6501, "num_layers": 2},
    {"layers": [10, 14, 17, 30], "fitness": 0.6489, "num_layers": 4}
  ],
  
  "discovered_patterns": {
    "1_layer_patterns": [
      {"pattern": [17], "frequency": 25, "avg_fitness": 0.58}
    ],
    "2_layer_patterns": [
      {"pattern": [13, 17], "frequency": 12, "avg_fitness": 0.63}
    ]
  },
  
  "statistics": {
    "total_fast_evaluations": 487,
    "total_full_evaluations": 43,
    "total_time_hours": 156.3
  }
}
```

## 🔧 高级功能

### 1. 智能初始化

种群初始化策略：
- **10% 精英种子**：已知的1-2层最优解
- **40% 启发式构建**：基于单层分数加权采样
- **50% 随机探索**：纯随机生成

### 2. 模式挖掘

动态挖掘高质量层组合模式：
- 分层处理（1层、2层、3层模式）
- 质量评分：`avg_fitness × log(1 + frequency)`
- 频率门槛和Top-N过滤
- 每5代或发现新全局最优时更新

### 3. 模式引导

使用发现的模式指导进化：
- **引导添加**：优先添加高质量单层
- **引导删除**：优先保留高质量层
- **动态概率**：从20%逐渐增加到60%
- 平衡探索（随机）与利用（模式）

### 4. 两阶段局部搜索

高效的邻域搜索策略：
1. **粗评估阶段**：快速评估所有邻居（limit=20-50）
2. **精细评估阶段**：只对top-k邻居完整评估（limit=None）
3. **节省成本**：减少90%+的完整评估

## 📝 注意事项

### GPU和内存

- **GPU显存需求**：~18-20GB（模型加载）
- **推荐GPU**：RTX A6000 (48GB) 或更高
- **多GPU支持**：通过`--gpu`参数指定

### 评估时间

| Limit | 时间/次 | 用途 |
|-------|---------|------|
| 10 | ~200s | 超快速测试 |
| 20 | ~300s | 快速测试/粗评估 |
| 50 | ~500s | 标准快速评估 |
| None | ~2000s | 完整MMLU评估 |

### 总搜索时间估算

**快速配置** (limit=20/None, pop=40, gen=20):
- 阶段1: ~30-40小时
- 阶段2: ~10-12小时
- 阶段3: ~60-80小时
- **总计：约100-130小时（4-5天）**

**标准配置** (limit=50/None, pop=40, gen=20):
- 阶段1: ~56-70小时
- 阶段2: ~10-12小时
- 阶段3: ~80-100小时
- **总计：约150-180小时（6-7天）**

### 实用建议

1. **先小规模测试**：
   ```bash
   python run_ga_search_real.py --population 20 --generations 10 --fast-limit 10
   ```

2. **使用tmux**：避免SSH断开导致任务中断

3. **监控GPU**：定期检查GPU状态和温度
   ```bash
   watch -n 10 nvidia-smi
   ```

4. **检查点恢复**：如果任务中断，可以从最近的检查点继续（需实现恢复功能）

## 🧪 测试

### 单元测试

```bash
cd genetic_algorithm/

# 基础功能
python test_basic.py

# 遗传算子
python test_operators.py

# 模式挖掘
python test_pattern_mining.py

# 局部搜索
python test_local_search.py

# 完整GA
python test_ga_with_patterns.py
```

### 集成测试

```bash
# Mock函数测试
python run_complete_search.py

# 真实评估测试（小规模）
python run_ga_search_real.py --population 10 --generations 5 --fast-limit 10
```

## 📚 文档索引

- **START_HERE.md** - 新手快速开始指南
- **ARCHITECTURE.md** - 详细架构设计和代码结构
- **USAGE_GUIDE.md** - 完整使用指南和参数说明
- **SUMMARY.md** - 项目总结和关键决策
- **FILES_INDEX.md** - 所有文件的详细索引

## 🎯 常见问题

### Q: 如何选择GPU？

使用 `nvidia-smi` 查看空闲GPU，选择显存占用最少的GPU。避免使用GPU 0（通常是默认GPU，容易被其他任务占用）。

### Q: 搜索卡住/无进展怎么办？

1. 检查GPU状态和内存
2. 查看日志文件确认是否在正常评估
3. 检查是否有其他进程占用GPU
4. 考虑降低种群大小或使用更小的limit

### Q: 如何加速搜索？

1. 使用更小的 `--fast-limit`（如20而非50）
2. 减小种群大小（如 `--population 30`）
3. 降低代数（如 `--generations 15`）
4. 确保MMLU评估缓存正常工作

### Q: 结果不理想怎么办？

1. 检查是否陷入局部最优（查看模式多样性）
2. 尝试不同的随机种子（`--seed`）
3. 增加种群大小和代数
4. 调整模式引导概率（修改`config.py`）

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

---

**开发时间**: 2025年10月  
**版本**: v1.0  
**作者**: Zhuangfei Hu & Claude (Cursor AI Assistant)
