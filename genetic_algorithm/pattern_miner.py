"""
模式挖掘模块 - 从优秀个体中提取和学习模式
"""
from typing import List, Dict, Set, Tuple
from collections import Counter, defaultdict
import numpy as np
from individual import Individual


class Pattern:
    """模式类 - 表示一个层组合模式"""
    
    def __init__(self, layers: Tuple[int, ...]):
        """
        Args:
            layers: 层的元组（有序）
        """
        self.layers = tuple(sorted(layers))  # 保证有序
        self.frequency = 0  # 出现次数
        self.fitness_sum = 0.0  # 累计适应度
        self.fitness_count = 0  # 参与计算的个体数
    
    @property
    def avg_fitness(self) -> float:
        """平均适应度"""
        if self.fitness_count == 0:
            return 0.0
        return self.fitness_sum / self.fitness_count
    
    @property
    def size(self) -> int:
        """模式大小（包含的层数）"""
        return len(self.layers)
    
    def update(self, fitness: float):
        """更新模式统计"""
        self.frequency += 1
        self.fitness_sum += fitness
        self.fitness_count += 1
    
    def __repr__(self) -> str:
        return f"Pattern({list(self.layers)}, freq={self.frequency}, avg_fitness={self.avg_fitness:.4f})"
    
    def __hash__(self):
        return hash(self.layers)
    
    def __eq__(self, other):
        return self.layers == other.layers


class PatternMiner:
    """模式挖掘器 - 从种群中提取优秀模式（分层处理）"""
    
    def __init__(self, max_pattern_size: int = 3, min_frequency: int = 2,
                 top_n_per_size: Dict[int, int] = None):
        """
        Args:
            max_pattern_size: 最大模式大小（如3表示最多提取3层的模式）
            min_frequency: 最小频率（出现次数低于此值的模式被过滤）
            top_n_per_size: 每个大小保留的top-N {1: 10, 2: 15, 3: 10}
        """
        self.max_pattern_size = max_pattern_size
        self.min_frequency = min_frequency
        
        if top_n_per_size is None:
            self.top_n_per_size = {1: 10, 2: 15, 3: 10}
        else:
            self.top_n_per_size = top_n_per_size
        
        # 按大小分组的模式库（直接存储，不需要中间的patterns字典）
        # {size: {pattern_tuple: Pattern}}
        self.patterns_by_size: Dict[int, Dict[Tuple[int, ...], Pattern]] = {
            1: {},
            2: {},
            3: {},
        }
        
        # 排序后的模式列表（用于快速访问）
        self.ranked_patterns_by_size: Dict[int, List[Pattern]] = {
            1: [],
            2: [],
            3: [],
        }
    
    def extract_patterns_from_individual(self, individual: Individual) -> List[Tuple[int, ...]]:
        """
        从单个个体中提取所有子集模式
        
        Args:
            individual: 个体
        
        Returns:
            所有子集模式列表
        """
        layers = individual.layers
        patterns = []
        
        # 提取1层模式
        for layer in layers:
            patterns.append((layer,))
        
        # 提取2层模式
        if self.max_pattern_size >= 2:
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    patterns.append(tuple(sorted([layers[i], layers[j]])))
        
        # 提取3层模式
        if self.max_pattern_size >= 3:
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    for k in range(j + 1, len(layers)):
                        patterns.append(tuple(sorted([layers[i], layers[j], layers[k]])))
        
        return patterns
    
    def mine_patterns(self, individuals: List[Individual], top_k: int = None):
        """
        从一组个体中挖掘模式（分层处理）
        
        Args:
            individuals: 个体列表（应该是优秀个体）
            top_k: 如果指定，只从适应度top-k的个体中挖掘
        """
        # 如果指定top_k，只取前k个
        if top_k:
            sorted_individuals = sorted(individuals, key=lambda x: x.fitness or 0, reverse=True)
            individuals = sorted_individuals[:top_k]
        
        # 从每个个体提取模式（分层）
        for individual in individuals:
            if individual.fitness is None:
                continue
            
            layers = individual.layers
            
            # 1层模式
            for layer in layers:
                pattern_tuple = (layer,)
                if pattern_tuple not in self.patterns_by_size[1]:
                    self.patterns_by_size[1][pattern_tuple] = Pattern(pattern_tuple)
                self.patterns_by_size[1][pattern_tuple].update(individual.fitness)
            
            # 2层模式
            if self.max_pattern_size >= 2 and len(layers) >= 2:
                for i in range(len(layers)):
                    for j in range(i + 1, len(layers)):
                        pattern_tuple = tuple(sorted([layers[i], layers[j]]))
                        if pattern_tuple not in self.patterns_by_size[2]:
                            self.patterns_by_size[2][pattern_tuple] = Pattern(pattern_tuple)
                        self.patterns_by_size[2][pattern_tuple].update(individual.fitness)
            
            # 3层模式
            if self.max_pattern_size >= 3 and len(layers) >= 3:
                for i in range(len(layers)):
                    for j in range(i + 1, len(layers)):
                        for k in range(j + 1, len(layers)):
                            pattern_tuple = tuple(sorted([layers[i], layers[j], layers[k]]))
                            if pattern_tuple not in self.patterns_by_size[3]:
                                self.patterns_by_size[3][pattern_tuple] = Pattern(pattern_tuple)
                            self.patterns_by_size[3][pattern_tuple].update(individual.fitness)
    
    def filter_and_rank_patterns(self):
        """
        过滤并排序模式（分层处理）
        """
        # 对每个大小的模式库分别处理
        for size in [1, 2, 3]:
            if size > self.max_pattern_size:
                continue
            
            # 过滤低频模式
            patterns_dict = self.patterns_by_size[size]
            filtered = [
                pattern for pattern in patterns_dict.values()
                if pattern.frequency >= self.min_frequency
            ]
            
            # 按质量排序
            filtered.sort(
                key=lambda p: p.avg_fitness * np.log(1 + p.frequency),
                reverse=True
            )
            
            # 保留top-N
            top_n = self.top_n_per_size.get(size, 10)
            self.ranked_patterns_by_size[size] = filtered[:top_n]
    
    def get_top_patterns(self, size: int, top_k: int = None) -> List[Pattern]:
        """
        获取指定大小的top-k模式
        
        Args:
            size: 模式大小（1, 2, 或 3）
            top_k: 返回前k个，None则返回所有排序后的模式
        
        Returns:
            模式列表
        """
        if size not in self.ranked_patterns_by_size:
            return []
        
        patterns = self.ranked_patterns_by_size[size]
        if top_k is None:
            return patterns
        return patterns[:top_k]
    
    def get_pattern_weights(self, size: int) -> Dict[Tuple[int, ...], float]:
        """
        获取模式的采样权重（归一化概率）
        
        Args:
            size: 模式大小
        
        Returns:
            {模式: 权重} 字典，权重和为1
        """
        patterns = self.patterns_by_size.get(size, [])
        if not patterns:
            return {}
        
        # 计算权重：基于质量分数
        weights = {}
        total = 0.0
        
        for pattern in patterns:
            weight = pattern.avg_fitness * np.log(1 + pattern.frequency)
            weights[pattern.layers] = weight
            total += weight
        
        # 归一化
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def sample_pattern(self, size: int, rng: np.random.Generator) -> Tuple[int, ...]:
        """
        从模式库中采样一个模式
        
        Args:
            size: 模式大小
            rng: 随机数生成器
        
        Returns:
            采样的模式（层元组）
        """
        patterns = self.patterns_by_size.get(size, [])
        if not patterns:
            return tuple()
        
        # 获取权重
        pattern_tuples = [p.layers for p in patterns]
        weights = np.array([p.avg_fitness * np.log(1 + p.frequency) for p in patterns])
        weights = weights / weights.sum()
        
        # 采样
        idx = rng.choice(len(pattern_tuples), p=weights)
        return pattern_tuples[idx]
    
    def print_summary(self, top_k: int = 5):
        """打印模式摘要"""
        print(f"\n{'='*70}")
        print(f"模式挖掘结果")
        print(f"{'='*70}")
        
        for size in [1, 2, 3]:
            if size > self.max_pattern_size:
                continue
            
            patterns = self.get_top_patterns(size, top_k)
            if patterns:
                print(f"\n{size}层模式 (Top {min(top_k, len(patterns))}):")
                for i, pattern in enumerate(patterns, 1):
                    quality = pattern.avg_fitness * np.log(1 + pattern.frequency)
                    layers_str = str(list(pattern.layers))
                    print(f"  {i}. {layers_str:20s} "
                          f"freq={pattern.frequency:2d}, "
                          f"avg={pattern.avg_fitness:.4f}, "
                          f"quality={quality:.3f}")
        
        print(f"{'='*70}")
    
    def clear(self):
        """清空模式库"""
        for size in [1, 2, 3]:
            self.patterns_by_size[size].clear()
            self.ranked_patterns_by_size[size].clear()


def test_pattern_miner():
    """测试模式挖掘器"""
    print("\n" + "=" * 70)
    print("测试模式挖掘器")
    print("=" * 70)
    
    # 创建测试个体
    test_individuals = [
        ([13, 14, 17], 0.65),      # 优秀
        ([12, 14, 17], 0.63),      # 优秀
        ([13, 16, 17], 0.62),      # 优秀
        ([14, 17], 0.58),          # 良好
        ([13, 17], 0.58),          # 良好
        ([10, 13, 17], 0.55),      # 一般
        ([17, 18], 0.48),          # 较差
    ]
    
    individuals = []
    for layers, fitness in test_individuals:
        ind = Individual.from_layers(layers)
        ind.set_fitness(fitness)
        individuals.append(ind)
    
    # 创建模式挖掘器
    miner = PatternMiner(max_pattern_size=3, min_frequency=2)
    
    # 挖掘模式
    print("\n从7个个体中挖掘模式...")
    miner.mine_patterns(individuals)
    
    # 查看原始挖掘结果
    print(f"\n原始模式总数: {len(miner.patterns)}")
    
    # 过滤和排序模式
    miner.filter_and_rank_patterns()
    
    print(f"过滤后模式总数: {sum(len(patterns) for patterns in miner.patterns_by_size.values())}")
    
    # 打印结果
    miner.print_summary(top_k=5)
    
    # 验证
    print("\n验证:")
    
    # 检查高频1层模式
    patterns_1 = miner.get_top_patterns(size=1, top_k=3)
    print(f"\nTop 3 单层模式:")
    for p in patterns_1:
        print(f"  {p}")
    
    # 层17应该是最高频的
    layer_17_pattern = [p for p in patterns_1 if p.layers == (17,)][0]
    assert layer_17_pattern.frequency >= 6, "层17应该出现在至少6个个体中"
    print(f"✓ 层17出现{layer_17_pattern.frequency}次（符合预期）")
    
    # 检查高频2层模式
    patterns_2 = miner.get_top_patterns(size=2, top_k=3)
    print(f"\nTop 3 双层模式:")
    for p in patterns_2:
        print(f"  {p}")
    
    # (14,17)和(13,17)应该是高频的
    pair_1417 = [p for p in patterns_2 if p.layers == (14, 17)]
    assert len(pair_1417) > 0, "(14,17)应该被挖掘出来"
    print(f"✓ (14,17)模式出现{pair_1417[0].frequency}次")
    
    print("\n" + "=" * 70)
    print("模式挖掘器测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    test_pattern_miner()

