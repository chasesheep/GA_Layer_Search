# 文件索引

## 📚 快速查找指南

### 🚀 开始使用

| 文件 | 说明 | 用途 |
|------|------|------|
| **START_HERE.md** | 快速开始指南 | ⭐ 首次使用必读 |
| **quick_test_real.sh** | 快速测试脚本 | 运行4-5小时测试 |
| **run_full_search.sh** | 完整搜索脚本 | 运行7天完整搜索 |
| **run_ga_search_real.py** | 主运行程序 | 自定义参数运行 |

### 📖 文档

| 文件 | 内容 |
|------|------|
| **README.md** | 完整项目说明和使用指南 |
| **USAGE_GUIDE.md** | 详细的使用教程（启动、监控、故障处理） |
| **ARCHITECTURE.md** | 系统架构和代码结构 |
| **SUMMARY.md** | 项目完成总结 |
| **FILES_INDEX.md** | 本文件索引（你在这里👈） |

### 🛠️ 核心代码

| 文件 | 功能 |
|------|------|
| **config.py** | GA配置参数 |
| **individual.py** | 个体类（编码/解码/约束） |
| **population.py** | 种群管理（初始化/评估） |
| **operators.py** | 基础遗传算子（选择/交叉/变异） |
| **operators_guided.py** | 模式引导的遗传算子 |
| **pattern_miner.py** | 模式挖掘（动态发现优秀模式） |
| **ga_core.py** | GA主流程（演化/模式更新/检查点） |
| **local_search.py** | 单阶段局部搜索 |
| **local_search_twostage.py** | 两阶段局部搜索 |
| **run_complete_search.py** | 三阶段搜索流程编排 |
| **fitness.py** | Mock适应度函数（测试用） |
| **real_fitness.py** | 真实MMLU评估 |
| **utils.py** | 工具函数 |

### 🧪 测试代码

| 文件 | 测试内容 |
|------|----------|
| **test_basic.py** | Individual和Population |
| **test_initialization.py** | 智能初始化 |
| **test_operators.py** | 遗传算子 |
| **test_pattern_mining.py** | 模式挖掘 |
| **test_ga_complete.py** | 基础GA |
| **test_ga_with_patterns.py** | 模式引导GA |
| **test_local_search.py** | 局部搜索 |

### 🔧 工具脚本

| 文件 | 功能 |
|------|------|
| **view_checkpoint.py** | 查看检查点文件 |
| **monitor_progress.sh** | 监控搜索进度 |

### 📁 数据文件

| 文件/目录 | 内容 |
|-----------|------|
| **single_layer_results.json** | 32个单层替换结果 |
| **results/mock_results/** | Mock函数测试结果 |
| **results/real_test/** | 快速测试结果 |
| **results/real_results/** | 完整搜索结果 |
| **results/*/checkpoints_*/** | 检查点文件（每3代） |

### 🗑️ 备份文件（可忽略）

| 文件 | 说明 |
|------|------|
| **ga_core_basic.py** | 基础GA（无模式引导）备份 |
| **operators_basic.py** | 基础算子备份 |

---

## 🎯 常用操作快速索引

### 启动搜索
- 快速测试：`./quick_test_real.sh`
- 完整搜索：`./run_full_search.sh`
- 自定义参数：查看 **run_ga_search_real.py**

### 监控进度
- 查看检查点：`./monitor_progress.sh results/real_test`
- 实时日志：`tail -f results/*/search_log_*.txt`
- 查看GPU：`nvidia-smi`

### 查看结果
- 检查点详情：`python view_checkpoint.py results/*/checkpoints_*/`
- JSON结果：`cat results/*/search_result_*.json | jq`

### 学习代码
1. 先看 **ARCHITECTURE.md** 了解整体结构
2. 核心算法在 **ga_core.py**
3. 模式挖掘在 **pattern_miner.py**
4. 局部搜索在 **local_search_twostage.py**

### 测试验证
1. Mock测试：`python run_complete_search.py`
2. 真实MMLU测试：`python real_fitness.py --mode simple`
3. 单元测试：`python test_*.py`

---

## 📊 推荐阅读顺序

**新用户**：
1. START_HERE.md ← 开始这里
2. README.md
3. 运行 quick_test_real.sh
4. USAGE_GUIDE.md

**开发者**：
1. ARCHITECTURE.md
2. ga_core.py
3. pattern_miner.py
4. 测试文件

**结果分析**：
1. view_checkpoint.py 查看检查点
2. 分析 search_result_*.json
3. SUMMARY.md 参考总结

---

**提示**：所有文件都在 `/home/huzhuangfei/Code/GandA/genetic_layer_search/` 目录下

