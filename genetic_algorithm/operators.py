"""
遗传算子：选择、交叉、变异
"""
from typing import List, Tuple, Optional
import numpy as np
from individual import Individual


# ==================== 选择算子 ====================

def tournament_selection(population: List[Individual], 
                        tournament_size: int = 3,
                        rng: Optional[np.random.Generator] = None) -> Individual:
    """
    锦标赛选择
    
    从种群中随机选择tournament_size个个体，返回其中适应度最高的
    
    Args:
        population: 种群个体列表
        tournament_size: 锦标赛大小
        rng: 随机数生成器
    
    Returns:
        选中的个体（副本）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 随机选择tournament_size个个体
    tournament_indices = rng.choice(len(population), size=tournament_size, replace=False)
    tournament = [population[i] for i in tournament_indices]
    
    # 返回适应度最高的个体（副本）
    winner = max(tournament, key=lambda ind: ind.fitness if ind.fitness else -float('inf'))
    return winner.copy()


# ==================== 交叉算子 ====================

def uniform_crossover(parent1: Individual, parent2: Individual,
                     crossover_rate: float = 0.5,
                     rng: Optional[np.random.Generator] = None) -> Tuple[Individual, Individual]:
    """
    均匀交叉
    
    每个基因位独立决定来自哪个父代
    
    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_rate: 交叉概率（每个基因位）
        rng: 随机数生成器
    
    Returns:
        两个子代
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_genes = len(parent1.genome)
    
    # 生成交叉掩码
    mask = rng.random(num_genes) < crossover_rate
    
    # 创建子代基因
    child1_genome = []
    child2_genome = []
    
    for i in range(num_genes):
        if mask[i]:
            # 交叉：交换基因
            child1_genome.append(parent2.genome[i])
            child2_genome.append(parent1.genome[i])
        else:
            # 不交叉：保持原样
            child1_genome.append(parent1.genome[i])
            child2_genome.append(parent2.genome[i])
    
    # 创建子代个体
    child1 = Individual(genome=child1_genome, num_layers=parent1.num_layers)
    child2 = Individual(genome=child2_genome, num_layers=parent2.num_layers)
    
    return child1, child2


def single_point_crossover(parent1: Individual, parent2: Individual,
                          rng: Optional[np.random.Generator] = None) -> Tuple[Individual, Individual]:
    """
    单点交叉
    
    随机选择一个交叉点，交换两个父代在该点之后的基因
    
    Args:
        parent1: 父代1
        parent2: 父代2
        rng: 随机数生成器
    
    Returns:
        两个子代
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_genes = len(parent1.genome)
    
    # 随机选择交叉点
    crossover_point = rng.integers(1, num_genes)
    
    # 创建子代基因
    child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
    child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
    
    # 创建子代个体
    child1 = Individual(genome=child1_genome, num_layers=parent1.num_layers)
    child2 = Individual(genome=child2_genome, num_layers=parent2.num_layers)
    
    return child1, child2


# ==================== 变异算子 ====================

def bit_flip_mutation(individual: Individual, 
                     mutation_rate: float = 0.1,
                     rng: Optional[np.random.Generator] = None) -> Individual:
    """
    比特翻转变异
    
    以mutation_rate的概率翻转每个基因位
    
    Args:
        individual: 个体
        mutation_rate: 变异率
        rng: 随机数生成器
    
    Returns:
        变异后的个体（新个体）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 复制个体
    mutated = individual.copy()
    
    # 对每个基因位进行变异
    for i in range(len(mutated.genome)):
        if rng.random() < mutation_rate:
            mutated.mutate_bit(i)
    
    return mutated


def swap_mutation(individual: Individual,
                 rng: Optional[np.random.Generator] = None) -> Individual:
    """
    交换变异（保持层数不变）
    
    随机选择一个1和一个0，将它们交换
    这样可以保持替换层数不变，不违反约束
    
    Args:
        individual: 个体
        rng: 随机数生成器
    
    Returns:
        变异后的个体（新个体）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 复制个体
    mutated = individual.copy()
    
    # 找到所有0和1的位置
    zeros = [i for i, gene in enumerate(mutated.genome) if gene == 0]
    ones = [i for i, gene in enumerate(mutated.genome) if gene == 1]
    
    # 如果有可交换的位置
    if zeros and ones:
        # 随机选择一个0和一个1
        zero_pos = rng.choice(zeros)
        one_pos = rng.choice(ones)
        
        # 交换
        mutated.genome[zero_pos] = 1
        mutated.genome[one_pos] = 0
        mutated._layers = None  # 清除缓存
        mutated.evaluated = False
    
    return mutated


def adaptive_mutation(individual: Individual,
                     base_mutation_rate: float = 0.1,
                     rng: Optional[np.random.Generator] = None) -> Individual:
    """
    自适应变异
    
    根据个体层数动态调整变异策略：
    - 2层个体：倾向添加层（翻转0→1）
    - 3层个体：均衡变异
    - 4层个体：倾向删除或替换（翻转1→0或swap）
    
    Args:
        individual: 个体
        base_mutation_rate: 基础变异率
        rng: 随机数生成器
    
    Returns:
        变异后的个体（新个体）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_layers = individual.num_replaced_layers
    
    # 策略选择
    if num_layers == 2:
        # 70%概率添加层，30%概率交换
        if rng.random() < 0.7:
            # 添加层：翻转一个0→1
            zeros = [i for i, gene in enumerate(individual.genome) if gene == 0]
            if zeros:
                mutated = individual.copy()
                pos = rng.choice(zeros)
                mutated.mutate_bit(pos)
                return mutated
        else:
            # 交换变异
            return swap_mutation(individual, rng)
    
    elif num_layers == 3:
        # 3层是平衡点：50%交换(保持), 25%添加, 25%删除
        rand = rng.random()
        if rand < 0.5:
            # 交换变异（保持3层）
            return swap_mutation(individual, rng)
        elif rand < 0.75:
            # 添加层（3→4）
            zeros = [i for i, gene in enumerate(individual.genome) if gene == 0]
            if zeros:
                mutated = individual.copy()
                pos = rng.choice(zeros)
                mutated.mutate_bit(pos)
                return mutated
            else:
                return swap_mutation(individual, rng)
        else:
            # 删除层（3→2）
            ones = [i for i, gene in enumerate(individual.genome) if gene == 1]
            if ones:
                mutated = individual.copy()
                pos = rng.choice(ones)
                mutated.mutate_bit(pos)
                return mutated
            else:
                return swap_mutation(individual, rng)
    
    else:  # num_layers == 4
        # 70%交换（保持4层），30%删除层
        if rng.random() < 0.7:
            return swap_mutation(individual, rng)
        else:
            # 删除层：翻转一个1→0
            ones = [i for i, gene in enumerate(individual.genome) if gene == 1]
            if ones:
                mutated = individual.copy()
                pos = rng.choice(ones)
                mutated.mutate_bit(pos)
                return mutated
    
    # 默认返回副本
    return individual.copy()


# ==================== 约束修复 ====================

def repair_individual(individual: Individual,
                     min_layers: int = 2,
                     max_layers: int = 4,
                     single_layer_scores: Optional[dict] = None,
                     rng: Optional[np.random.Generator] = None) -> Individual:
    """
    修复违反约束的个体
    
    Args:
        individual: 个体
        min_layers: 最少层数
        max_layers: 最多层数
        single_layer_scores: 单层分数（用于智能修复）
        rng: 随机数生成器
    
    Returns:
        修复后的个体（新个体）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    repaired = individual.copy()
    num_layers = repaired.num_replaced_layers
    
    # 情况1: 层数太多（>max_layers）
    if num_layers > max_layers:
        # 需要删除 (num_layers - max_layers) 层
        num_to_remove = num_layers - max_layers
        
        ones_positions = [i for i, gene in enumerate(repaired.genome) if gene == 1]
        
        if single_layer_scores:
            # 智能删除：删除分数最低的层
            layers_with_scores = [(pos, single_layer_scores.get(pos, 0.25)) 
                                 for pos in ones_positions]
            layers_with_scores.sort(key=lambda x: x[1])  # 升序，分数低的在前
            to_remove = [pos for pos, _ in layers_with_scores[:num_to_remove]]
        else:
            # 随机删除
            to_remove = rng.choice(ones_positions, size=num_to_remove, replace=False)
        
        # 删除选中的层
        for pos in to_remove:
            repaired.genome[pos] = 0
        
        repaired._layers = None
        repaired.evaluated = False
    
    # 情况2: 层数太少（<min_layers）
    elif num_layers < min_layers:
        # 需要添加 (min_layers - num_layers) 层
        num_to_add = min_layers - num_layers
        
        zeros_positions = [i for i, gene in enumerate(repaired.genome) if gene == 0]
        
        if single_layer_scores:
            # 智能添加：添加分数最高的层
            layers_with_scores = [(pos, single_layer_scores.get(pos, 0.25)) 
                                 for pos in zeros_positions]
            layers_with_scores.sort(key=lambda x: x[1], reverse=True)  # 降序
            to_add = [pos for pos, _ in layers_with_scores[:num_to_add]]
        else:
            # 随机添加
            to_add = rng.choice(zeros_positions, size=num_to_add, replace=False)
        
        # 添加选中的层
        for pos in to_add:
            repaired.genome[pos] = 1
        
        repaired._layers = None
        repaired.evaluated = False
    
    return repaired


def apply_crossover(parent1: Individual, parent2: Individual,
                   crossover_rate: float = 0.8,
                   crossover_type: str = 'uniform',
                   rng: Optional[np.random.Generator] = None) -> Tuple[Individual, Individual]:
    """
    应用交叉操作（带概率控制）
    
    Args:
        parent1: 父代1
        parent2: 父代2
        crossover_rate: 交叉发生的概率
        crossover_type: 交叉类型 ('uniform' 或 'single_point')
        rng: 随机数生成器
    
    Returns:
        两个子代
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 以crossover_rate概率进行交叉
    if rng.random() < crossover_rate:
        if crossover_type == 'uniform':
            return uniform_crossover(parent1, parent2, crossover_rate=0.5, rng=rng)
        elif crossover_type == 'single_point':
            return single_point_crossover(parent1, parent2, rng=rng)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")
    else:
        # 不交叉，直接返回父代副本
        return parent1.copy(), parent2.copy()


def apply_mutation(individual: Individual,
                  mutation_rate: float = 0.1,
                  mutation_type: str = 'adaptive',
                  single_layer_scores: Optional[dict] = None,
                  rng: Optional[np.random.Generator] = None) -> Individual:
    """
    应用变异操作
    
    Args:
        individual: 个体
        mutation_rate: 变异率
        mutation_type: 变异类型 ('bit_flip', 'swap', 'adaptive')
        single_layer_scores: 单层分数（用于智能变异）
        rng: 随机数生成器
    
    Returns:
        变异后的个体
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if mutation_type == 'bit_flip':
        return bit_flip_mutation(individual, mutation_rate, rng)
    elif mutation_type == 'swap':
        return swap_mutation(individual, rng)
    elif mutation_type == 'adaptive':
        return adaptive_mutation(individual, mutation_rate, rng)
    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")


# ==================== 完整的繁殖流程 ====================

def reproduce(parent1: Individual, parent2: Individual,
             config,
             single_layer_scores: Optional[dict] = None,
             rng: Optional[np.random.Generator] = None) -> Tuple[Individual, Individual]:
    """
    完整的繁殖流程：交叉 → 变异 → 修复
    
    Args:
        parent1: 父代1
        parent2: 父代2
        config: GA配置对象
        single_layer_scores: 单层分数
        rng: 随机数生成器
    
    Returns:
        两个子代（已修复约束）
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 1. 交叉
    child1, child2 = apply_crossover(
        parent1, parent2,
        crossover_rate=config.crossover_rate,
        crossover_type='uniform',
        rng=rng
    )
    
    # 2. 变异
    child1 = apply_mutation(
        child1,
        mutation_rate=config.mutation_rate,
        mutation_type='adaptive',
        single_layer_scores=single_layer_scores,
        rng=rng
    )
    
    child2 = apply_mutation(
        child2,
        mutation_rate=config.mutation_rate,
        mutation_type='adaptive',
        single_layer_scores=single_layer_scores,
        rng=rng
    )
    
    # 3. 修复约束
    if not child1.is_valid(config.min_layers, config.max_layers):
        child1 = repair_individual(
            child1,
            min_layers=config.min_layers,
            max_layers=config.max_layers,
            single_layer_scores=single_layer_scores,
            rng=rng
        )
    
    if not child2.is_valid(config.min_layers, config.max_layers):
        child2 = repair_individual(
            child2,
            min_layers=config.min_layers,
            max_layers=config.max_layers,
            single_layer_scores=single_layer_scores,
            rng=rng
        )
    
    return child1, child2

