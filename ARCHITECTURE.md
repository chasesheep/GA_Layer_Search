# GA层替换搜索系统架构

## 当前代码结构

```
genetic_layer_search/
├── 核心模块
│   ├── config.py                  # 配置参数
│   ├── individual.py              # 个体类（编码/解码/约束）
│   ├── population.py              # 种群管理（初始化/评估/统计）
│   ├── fitness.py                 # Mock适应度函数
│   ├── pattern_miner.py           # 模式挖掘（分层统计）
│   ├── operators.py               # 基础遗传算子（选择/交叉/变异）
│   ├── operators_guided.py        # 模式引导的遗传算子
│   ├── ga_core.py                 # GA主流程（集成所有功能）
│   ├── local_search.py            # 单阶段局部搜索
│   └── local_search_twostage.py   # 两阶段局部搜索
│
├── 备份版本
│   ├── ga_core_basic.py           # 基础GA（无模式引导）
│   └── operators_basic.py         # 基础遗传算子
│
├── 测试脚本
│   ├── test_basic.py              # 测试Individual和Population
│   ├── test_smart_init.py         # 测试智能初始化
│   ├── test_initialization.py     # 测试初始化详情
│   ├── test_operators.py          # 测试遗传算子
│   ├── test_ga_complete.py        # 测试基础GA
│   ├── test_pattern_mining.py     # 测试模式挖掘
│   ├── test_ga_with_patterns.py   # 测试模式引导GA
│   └── test_local_search.py       # 测试局部搜索
│
├── 数据文件
│   └── single_layer_results.json  # 单层替换结果
│
└── 结果目录
    └── results/
        ├── mock_results/
        └── real_results/
```

---

## 当前搜索逻辑（代码实现）

### **完整流程：run_complete_search.py (NEW)**

这是主入口，协调三个阶段：

```python
class CompleteSearchPipeline:
    阶段1: phase1_ga_search()      → 调用ga_core.py
    阶段2: phase2_full_evaluation() → 批量评估
    阶段3: phase3_local_refinement() → 调用local_search_twostage.py
```

---

### **阶段1：ga_core.py (GA粗搜索)**

#### 1. 初始化 (`GeneticAlgorithm.__init__`, 行70-110)
```python
def __init__(config, fitness_func, single_layer_scores, known_best_solutions):
    # 创建种群
    # 创建模式挖掘器（如果启用）
    # 初始化统计信息
```

#### 2. 种群初始化 (`initialize()`, 行130-148)
```python
def initialize():
    # 智能初始化种群（population.py, 行48-162）
    #   - 10% 精英种子（1-2层已知最优解）
    #   - 40% 启发式构建（基于单层分数加权采样）
    #   - 50% 随机探索
    # 评估初始种群
    # 更新全局最优
```

#### 3. 主循环 (`run()`, 行361-409)
```python
def run():
    initialize()
    
    while not should_terminate():
        evolve_one_generation()
        print_generation_summary()
    
    return results
```

#### 4. 单代演化 (`evolve_one_generation()`, 行199-280)
```python
def evolve_one_generation():
    # 1. 选择和繁殖
    for _ in range(population_size - elite_size):
        parent1 = tournament_selection()  # operators.py, 行13-40
        parent2 = tournament_selection()
        
        if pattern_mining_enabled:
            child1, child2 = reproduce_with_patterns()  # operators_guided.py, 行146-193
        else:
            child1, child2 = reproduce()  # operators.py, 行290-333
    
    # 2. 精英保留
    elites = population[:elite_size]
    
    # 3. 替换种群
    new_population = elites + offspring
    
    # 4. 评估新个体
    evaluate_population()
    
    # 5. 更新最优
    update_global_best()
    
    # 6. 更新模式（每5代或发现新最优时）
    if generation % 5 == 0 or new_best_found:
        update_patterns()  # 行172-197
```

#### 5. 模式更新 (`update_patterns()`, 行172-197)
```python
def update_patterns():
    # 1. 获取top-20个体（population.py, 行82-95）
    top_individuals = population.get_top_k(20)
    
    # 2. 清空旧模式
    pattern_miner.clear()
    
    # 3. 挖掘新模式（pattern_miner.py, 行117-161）
    pattern_miner.mine_patterns(top_individuals)
    
    # 4. 过滤和排序（pattern_miner.py, 行163-187）
    pattern_miner.filter_and_rank_patterns()
    
    # 5. 增加引导概率
    pattern_guided_prob += 0.05  (最高60%)
```

---

### **遗传算子：operators_guided.py**

#### 模式引导的变异 (`adaptive_mutation_with_patterns()`, 行96-144)
```python
def adaptive_mutation_with_patterns(individual, pattern_miner, guide_prob):
    num_layers = len(individual.layers)
    
    if num_layers == 2:
        if random() < 0.7:
            # 添加层（30%概率使用模式引导）
            if random() < guide_prob:
                layer = sample_layer_from_patterns()  # 行16-42
            else:
                layer = sample_from_single_scores()
    
    elif num_layers == 3:
        # 40%交换，30%添加，30%删除
        ...
    
    elif num_layers == 4:
        # 70%交换，30%删除
        ...
```

---

### **局部搜索：local_search.py**

#### 爬山算法 (`hill_climbing()`, 行69-154)
```python
def hill_climbing(initial_layers):
    current = initial_layers
    
    while iteration < max_iterations:
        # 1. 生成邻域（行28-66）
        neighbors = generate_neighbors(current)
        #   - 替换操作: n × (32-n)个
        #   - 添加操作: (32-n)个（如果<4层）
        #   - 删除操作: n个（如果>2层）
        
        # 2. 评估所有邻居
        best_neighbor = None
        for neighbor in neighbors:
            fitness = evaluate(neighbor)
            if fitness > current_fitness:
                best_neighbor = neighbor
        
        # 3. 移动到最优邻居
        if best_neighbor:
            current = best_neighbor
        else:
            break  # 达到局部最优
    
    return current
```

---

## 三阶段搜索流程总结

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段1: GA粗搜索                                              │
│ 文件: ga_core.py                                            │
│ 函数: GeneticAlgorithm.run()                                │
│ ─────────────────────────────────────────────────────────── │
│ 输入: fitness_func（快速评估，limit=50）                     │
│ 过程:                                                       │
│   - 智能初始化40个个体 (population.py:48-162)               │
│   - 演化20-30代                                             │
│     * 选择 (operators.py:13-40)                             │
│     * 交叉 (operators.py:88-126)                            │
│     * 变异 (operators_guided.py:96-144)                     │
│     * 模式更新 (ga_core.py:172-197, 每5代)                  │
│   - 返回GAResults对象                                       │
│ 输出: top-20候选解（按粗评估分数）                           │
│ 评估: ~500次 (limit=50)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段2: 完整评估候选                                          │
│ 文件: run_complete_search.py                                │
│ 函数: CompleteSearchPipeline.phase2_full_evaluation()       │
│ ─────────────────────────────────────────────────────────── │
│ 输入: 20个候选解（来自阶段1）                                │
│ 过程:                                                       │
│   - 对每个候选用完整评估函数评估                             │
│   - 按完整评估分数排序                                       │
│ 输出: 真实top-10解                                          │
│ 评估: 20次 (limit=None)                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段3: 局部精细优化                                          │
│ 文件: local_search_twostage.py                              │
│ 函数: TwoStageLocalSearch.two_stage_hill_climbing()         │
│ ─────────────────────────────────────────────────────────── │
│ 输入: 真实top-3解（来自阶段2）                               │
│ 过程: 对每个解                                              │
│   循环:                                                     │
│     1. 生成邻域（90-120个）                                 │
│     2. 粗评估所有邻居 (limit=50)                            │
│     3. 选择top-5有改进的邻居                                │
│     4. 完整评估这5个 (limit=None)                           │
│     5. 移动到最优邻居                                       │
│   直到无改进                                                │
│ 输出: 3个局部最优解                                         │
│ 评估:                                                       │
│   - 粗: ~900次 (3×100×3轮, limit=50)                       │
│   - 完整: ~30次 (3×5×2轮, limit=None)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 当前的完整搜索流程

### **流程图**

```
┌─────────────────────────────────────────────────────┐
│  阶段0: 准备                                         │
│  - 加载单层替换结果 (single_layer_results.json)      │
│  - 加载已知1-2层最优解                               │
│  - 创建适应度函数                                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  阶段1: GA粗搜索 (ga_core.py)                        │
│  使用: Mock函数 或 MMLU(limit=50)                    │
│  ────────────────────────────────────────────────    │
│  1. 智能初始化种群（40-50个）                         │
│  2. 演化20-30代                                      │
│     每代:                                            │
│       - 选择父代（锦标赛）                            │
│       - 交叉                                         │
│       - 变异（可能使用模式引导）                       │
│       - 精英保留                                     │
│     每5代或新最优:                                   │
│       - 更新模式库                                   │
│       - 增加引导概率                                 │
│  3. 输出: top-10候选解                               │
│  ────────────────────────────────────────────────    │
│  评估次数: ~500次（50个体×10代）                     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  阶段2: 局部优化 (local_search.py)                   │
│  使用: 与阶段1相同的评估函数                          │
│  ────────────────────────────────────────────────    │
│  对top-3候选解:                                      │
│    1. 生成邻域（90-120个）                           │
│    2. 评估所有邻居                                   │
│    3. 移动到最优邻居                                 │
│    4. 重复直到局部最优                               │
│  ────────────────────────────────────────────────    │
│  评估次数: ~300次（3个解×100邻域×1-2轮）              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  阶段3: 完整评估（如果需要）                          │
│  使用: MMLU(limit=None)                              │
│  ────────────────────────────────────────────────    │
│  对局部优化后的top-3解:                               │
│    - 完整MMLU评估                                    │
│    - 确定真实排名                                    │
│  ────────────────────────────────────────────────    │
│  评估次数: 3次（完整MMLU，昂贵）                      │
└─────────────────────────────────────────────────────┘
```

---

## 问题和混乱点

### 问题1: 局部搜索应该在什么评估级别？

**当前有两个局部搜索实现**：
1. `local_search.py` - 单阶段（假设用同一个fitness函数）
2. `local_search_twostage.py` - 两阶段（粗评估+完整评估）

**混乱**：
- 如果GA用limit=50，局部搜索也用limit=50 → 用`local_search.py`
- 如果要在局部搜索中引入完整评估 → 用`local_search_twostage.py`

### 问题2: 整体流程应该是几个阶段？

**选项A：两阶段（简单）**
```python
阶段1: GA (limit=50)
  → top-10候选
  
阶段2: 对top-10完整评估 (limit=None)
  → 找到真实top-3
```

**选项B：三阶段（当前混乱状态）**
```python
阶段1: GA (limit=50)
  → top-10
  
阶段2: 局部搜索 (limit=50???)
  → 优化top-3
  
阶段3: 完整评估 (limit=None)
  → ???
```

**选项C：四阶段（两阶段局部搜索）**
```python
阶段1: GA (limit=50)
  → top-20
  
阶段2: 局部搜索粗评估 (limit=50)
  → 对top-20做邻域搜索（粗评估筛选）
  → 找到10个局部最优
  
阶段3: 完整评估top-10 (limit=None)
  → 找到真实top-3
  
阶段4: 局部搜索精细评估 (limit=None)
  → 对真实top-3做邻域搜索（完整评估）
  → 确认最终最优
```

---

## 我的建议：**简化为三阶段**

```python
┌────────────────────────────────────────┐
│ 阶段1: GA粗搜索 (limit=50)              │
│ 代码: ga_core.py                       │
│ ──────────────────────────────────     │
│ - 智能初始化                            │
│ - 演化20-30代（带模式引导）              │
│ - 输出: top-20候选                      │
│ 评估: ~500次 (limit=50)                │
└────────────────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 阶段2: 完整评估候选解 (limit=None)      │
│ 代码: 简单的批量评估                    │
│ ──────────────────────────────────     │
│ - 对top-20候选完整评估                  │
│ - 找到真实top-5                        │
│ 评估: 20次 (limit=None)                │
└────────────────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 阶段3: 局部精细优化 (limit=None)        │
│ 代码: local_search_twostage.py         │
│ ──────────────────────────────────     │
│ 对真实top-3:                           │
│   - 邻域粗评估 (limit=50) 筛选         │
│   - 有希望的邻居完整评估 (limit=None)   │
│   - 迭代优化                           │
│ 评估:                                  │
│   - 粗: ~900次 (3×100×3轮, limit=50)  │
│   - 完整: ~30次 (3×3×3轮, limit=None) │
└────────────────────────────────────────┘
```

**评估成本**：
- limit=50: ~1400次（GA 500 + 局部搜索 900）
- limit=None: ~50次（候选评估 20 + 局部搜索 30）

**特点**：
- ✓ 清晰：三个独立阶段
- ✓ 高效：完整评估只有50次
- ✓ 充分：20个候选都被完整评估了

---

## 需要调整的代码

### 1. 明确各阶段的职责

```python
ga_core.py:
  - 只负责GA搜索（单一评估函数）
  - 不管是limit=50还是limit=None
  - 输出top-K结果

local_search_twostage.py:
  - 负责两阶段局部搜索
  - 接受fast和full两个fitness函数
  - 对单个解优化
```

### 2. 创建主运行脚本

```python
run_ga_search.py:
  阶段1: 用GA（limit=50） → top-20
  阶段2: 批量完整评估top-20 → 真实top-5
  阶段3: 对top-3两阶段局部搜索 → 最终最优
```

---

## 问题

**现在应该**：

1. **创建统一的运行脚本** `run_ga_search.py`
   - 整合三个阶段
   - 明确每个阶段的输入输出
   - 先用mock函数测试完整流程

2. **还是先讨论**流程设计是否合理？

**你觉得这个三阶段流程合理吗？还是有其他想法？**
