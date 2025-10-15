"""
模式引导的遗传算子
"""
from typing import Optional, Dict
import numpy as np
from individual import Individual
from pattern_miner import PatternMiner
from operators import swap_mutation


def sample_layer_from_patterns(pattern_miner: PatternMiner, 
                               exclude_layers: set,
                               rng: np.random.Generator) -> Optional[int]:
    """
    从1层模式库中按质量采样一个层
    
    Args:
        pattern_miner: 模式挖掘器
        exclude_layers: 要排除的层
        rng: 随机数生成器
    
    Returns:
        采样的层ID，如果无可用层则返回None
    """
    # 获取1层模式
    patterns = pattern_miner.get_top_patterns(size=1)
    if not patterns:
        return None
    
    # 过滤已有的层
    candidates = [p for p in patterns if p.layers[0] not in exclude_layers]
    if not candidates:
        return None
    
    # 按质量加权采样
    qualities = np.array([p.avg_fitness * np.log(1 + p.frequency) for p in candidates])
    weights = qualities / qualities.sum()
    
    idx = rng.choice(len(candidates), p=weights)
    return candidates[idx].layers[0]


def find_weak_layer(individual: Individual,
                   pattern_miner: PatternMiner,
                   rng: np.random.Generator) -> int:
    """
    识别个体中的"弱层"（不在优秀模式中或质量低）
    
    Args:
        individual: 个体
        pattern_miner: 模式挖掘器
        rng: 随机数生成器
    
    Returns:
        要删除的层ID
    """
    # 获取1层模式质量
    patterns_1 = pattern_miner.get_top_patterns(size=1)
    
    if patterns_1:
        # 构建层→质量映射
        layer_quality = {p.layers[0]: p.avg_fitness * np.log(1 + p.frequency) 
                        for p in patterns_1}
        
        # 计算当前个体各层的质量
        current_layers = individual.layers
        layers_with_quality = [(layer, layer_quality.get(layer, 0)) 
                              for layer in current_layers]
        
        # 按质量排序，删除质量最低的
        layers_with_quality.sort(key=lambda x: x[1])
        return layers_with_quality[0][0]
    
    # 如果没有模式信息，随机选择
    return rng.choice(individual.layers)


def pattern_guided_add_layer(individual: Individual,
                            pattern_miner: PatternMiner,
                            single_layer_scores: Dict[int, float],
                            guide_prob: float,
                            rng: np.random.Generator) -> Individual:
    """
    模式引导的添加层操作
    
    Args:
        individual: 个体
        pattern_miner: 模式挖掘器
        single_layer_scores: 单层分数（兜底使用）
        guide_prob: 引导概率
        rng: 随机数生成器
    
    Returns:
        变异后的个体
    """
    current_layers = set(individual.layers)
    mutated = individual.copy()
    
    # 决定是否使用模式引导
    if rng.random() < guide_prob:
        # 使用模式引导
        new_layer = sample_layer_from_patterns(pattern_miner, current_layers, rng)
        
        if new_layer is None:
            # 如果模式库没有可用层，使用单层分数
            available = [l for l in range(32) if l not in current_layers]
            if available:
                scores = np.array([single_layer_scores.get(l, 0.25) for l in available])
                weights = (scores ** 2) / (scores ** 2).sum()
                new_layer = rng.choice(available, p=weights)
    else:
        # 基于单层分数采样
        available = [l for l in range(32) if l not in current_layers]
        if available:
            scores = np.array([single_layer_scores.get(l, 0.25) for l in available])
            weights = (scores ** 2) / (scores ** 2).sum()
            new_layer = rng.choice(available, p=weights)
        else:
            return mutated
    
    if new_layer is not None:
        mutated.genome[new_layer] = 1
        mutated._layers = None
        mutated.evaluated = False
    
    return mutated


def pattern_guided_remove_layer(individual: Individual,
                               pattern_miner: PatternMiner,
                               guide_prob: float,
                               rng: np.random.Generator) -> Individual:
    """
    模式引导的删除层操作
    
    优先删除"弱层"（不在优秀模式中的层）
    """
    mutated = individual.copy()
    
    if rng.random() < guide_prob:
        # 使用模式引导：删除弱层
        layer_to_remove = find_weak_layer(individual, pattern_miner, rng)
    else:
        # 随机删除
        layer_to_remove = rng.choice(individual.layers)
    
    mutated.genome[layer_to_remove] = 0
    mutated._layers = None
    mutated.evaluated = False
    
    return mutated


def adaptive_mutation_with_patterns(individual: Individual,
                                   base_mutation_rate: float,
                                   pattern_miner: PatternMiner,
                                   single_layer_scores: Dict[int, float],
                                   guide_prob: float,
                                   rng: Optional[np.random.Generator] = None) -> Individual:
    """
    自适应变异 + 模式引导
    
    Args:
        individual: 个体
        base_mutation_rate: 基础变异率（未使用，保留接口兼容）
        pattern_miner: 模式挖掘器
        single_layer_scores: 单层分数
        guide_prob: 模式引导概率
        rng: 随机数生成器
    
    Returns:
        变异后的个体
    """
    if rng is None:
        rng = np.random.default_rng()
    
    num_layers = individual.num_replaced_layers
    
    # 2层个体：倾向添加层
    if num_layers == 2:
        if rng.random() < 0.7:
            # 添加层（可能使用模式引导）
            return pattern_guided_add_layer(
                individual, pattern_miner, single_layer_scores, guide_prob, rng
            )
        else:
            # 交换变异
            return swap_mutation(individual, rng)
    
    # 3层个体：平衡
    elif num_layers == 3:
        rand = rng.random()
        if rand < 0.4:
            # 交换变异
            return swap_mutation(individual, rng)
        elif rand < 0.7:
            # 添加层（可能使用模式引导）
            return pattern_guided_add_layer(
                individual, pattern_miner, single_layer_scores, guide_prob, rng
            )
        else:
            # 删除层（可能使用模式引导）
            return pattern_guided_remove_layer(
                individual, pattern_miner, guide_prob, rng
            )
    
    # 4层个体：倾向保持或减少
    else:  # num_layers == 4
        if rng.random() < 0.7:
            # 交换变异
            return swap_mutation(individual, rng)
        else:
            # 删除层（可能使用模式引导）
            return pattern_guided_remove_layer(
                individual, pattern_miner, guide_prob, rng
            )
    
    # 默认返回副本
    return individual.copy()


def reproduce_with_patterns(parent1: Individual, parent2: Individual,
                           config,
                           pattern_miner: PatternMiner,
                           single_layer_scores: Dict[int, float],
                           guide_prob: float,
                           rng: Optional[np.random.Generator] = None) -> tuple:
    """
    完整的繁殖流程（带模式引导）：交叉 → 变异 → 修复
    
    Args:
        parent1: 父代1
        parent2: 父代2
        config: GA配置
        pattern_miner: 模式挖掘器
        single_layer_scores: 单层分数
        guide_prob: 模式引导概率
        rng: 随机数生成器
    
    Returns:
        两个子代
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # 导入标准交叉算子
    from operators import uniform_crossover, repair_individual
    
    # 1. 交叉（暂时不引导，使用标准交叉）
    if rng.random() < config.crossover_rate:
        child1, child2 = uniform_crossover(parent1, parent2, crossover_rate=0.5, rng=rng)
    else:
        child1, child2 = parent1.copy(), parent2.copy()
    
    # 2. 变异（使用模式引导）
    child1 = adaptive_mutation_with_patterns(
        child1, config.mutation_rate, pattern_miner, 
        single_layer_scores, guide_prob, rng
    )
    
    child2 = adaptive_mutation_with_patterns(
        child2, config.mutation_rate, pattern_miner,
        single_layer_scores, guide_prob, rng
    )
    
    # 3. 修复约束
    if not child1.is_valid(config.min_layers, config.max_layers):
        child1 = repair_individual(
            child1, config.min_layers, config.max_layers,
            single_layer_scores, rng
        )
    
    if not child2.is_valid(config.min_layers, config.max_layers):
        child2 = repair_individual(
            child2, config.min_layers, config.max_layers,
            single_layer_scores, rng
        )
    
    return child1, child2

