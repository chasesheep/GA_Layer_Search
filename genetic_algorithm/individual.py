"""
个体类 - 表示一个层替换方案
"""
from typing import List, Optional
import numpy as np


class Individual:
    """
    遗传算法个体
    
    编码方式：二进制向量，长度为num_layers
    例如：[0,0,0,1,0,1,0,...,1,0] 表示替换第3、5、...、30层
    """
    
    def __init__(self, genome: Optional[List[int]] = None, num_layers: int = 32):
        """
        Args:
            genome: 二进制基因序列，None则初始化为全0
            num_layers: 总层数
        """
        self.num_layers = num_layers
        
        if genome is None:
            self.genome = [0] * num_layers
        else:
            assert len(genome) == num_layers, \
                f"Genome length ({len(genome)}) must equal num_layers ({num_layers})"
            self.genome = list(genome)  # 复制一份
        
        self._layers = None  # 缓存解码结果
        self.fitness = None  # 适应度
        self.evaluated = False  # 是否已评估
    
    @property
    def layers(self) -> List[int]:
        """解码基因得到层索引列表"""
        if self._layers is None:
            self._layers = [i for i, gene in enumerate(self.genome) if gene == 1]
        return self._layers
    
    @property
    def num_replaced_layers(self) -> int:
        """替换的层数"""
        return len(self.layers)
    
    def is_valid(self, min_layers: int = 2, max_layers: int = 4) -> bool:
        """
        检查个体是否满足约束
        
        Args:
            min_layers: 最少替换层数
            max_layers: 最多替换层数
        
        Returns:
            是否有效
        """
        num_layers = self.num_replaced_layers
        return min_layers <= num_layers <= max_layers
    
    def set_fitness(self, fitness: float):
        """设置适应度"""
        self.fitness = fitness
        self.evaluated = True
    
    def copy(self) -> 'Individual':
        """复制个体"""
        new_individual = Individual(genome=self.genome.copy(), num_layers=self.num_layers)
        new_individual.fitness = self.fitness
        new_individual.evaluated = self.evaluated
        return new_individual
    
    def mutate_bit(self, position: int):
        """翻转指定位置的基因"""
        assert 0 <= position < self.num_layers, f"Invalid position: {position}"
        self.genome[position] = 1 - self.genome[position]
        self._layers = None  # 清除缓存
        self.evaluated = False  # 需要重新评估
    
    def __lt__(self, other: 'Individual') -> bool:
        """比较运算符（用于排序，适应度越大越好）"""
        if self.fitness is None:
            return True
        if other.fitness is None:
            return False
        return self.fitness < other.fitness
    
    def __eq__(self, other: 'Individual') -> bool:
        """相等判断（基于基因序列）"""
        return self.genome == other.genome
    
    def __hash__(self):
        """哈希值（用于集合和字典）"""
        return hash(tuple(self.genome))
    
    def __repr__(self) -> str:
        """字符串表示"""
        layers_str = str(self.layers) if len(self.layers) <= 5 else \
                     f"{self.layers[:3]}...{self.layers[-2:]}"
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"Individual(layers={layers_str}, fitness={fitness_str})"
    
    @classmethod
    def random(cls, num_layers: int, min_layers: int, max_layers: int, 
               rng: Optional[np.random.Generator] = None) -> 'Individual':
        """
        创建随机个体（保证满足约束）
        
        Args:
            num_layers: 总层数
            min_layers: 最少替换层数
            max_layers: 最多替换层数
            rng: 随机数生成器
        
        Returns:
            随机个体
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # 随机选择要替换的层数
        k = rng.integers(min_layers, max_layers + 1)
        
        # 随机选择k个层
        selected_layers = rng.choice(num_layers, size=k, replace=False)
        
        # 构建基因序列
        genome = [0] * num_layers
        for layer_idx in selected_layers:
            genome[layer_idx] = 1
        
        return cls(genome=genome, num_layers=num_layers)
    
    @classmethod
    def from_layers(cls, layers: List[int], num_layers: int = 32) -> 'Individual':
        """
        从层索引列表创建个体
        
        Args:
            layers: 要替换的层索引列表，例如 [13, 14, 16, 17]
            num_layers: 总层数
        
        Returns:
            个体
        """
        genome = [0] * num_layers
        for layer_idx in layers:
            assert 0 <= layer_idx < num_layers, f"Invalid layer index: {layer_idx}"
            genome[layer_idx] = 1
        
        return cls(genome=genome, num_layers=num_layers)

