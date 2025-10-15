"""
遗传算法配置参数
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GAConfig:
    """遗传算法配置类"""
    
    # ========== 问题相关参数 ==========
    num_layers: int = 32  # 总层数（Llamba模型有32层）
    min_layers: int = 2   # 最少替换层数（attention budget下限）
    max_layers: int = 4   # 最多替换层数（attention budget上限）
    
    # ========== 种群参数 ==========
    population_size: int = 40  # 种群大小
    elite_size: int = 3        # 精英个体数量（直接保留到下一代）
    
    # ========== 遗传算子参数 ==========
    crossover_rate: float = 0.8      # 交叉概率
    mutation_rate: float = 0.1       # 变异概率
    tournament_size: int = 3         # 锦标赛选择大小
    
    # ========== 终止条件 ==========
    max_generations: int = 50              # 最大代数
    no_improvement_threshold: int = 10     # 连续无改进代数（提前终止）
    target_fitness: Optional[float] = None # 目标适应度（达到即终止）
    
    # ========== 模式挖掘参数 ==========
    pattern_mining_enabled: bool = True       # 是否启用模式挖掘
    pattern_update_interval: int = 5          # 模式更新间隔（代数）
    pattern_mining_top_k: int = 20            # 从top-k个体中挖掘
    pattern_max_size: int = 3                 # 最大模式大小
    pattern_min_frequency: int = 2            # 最小频率阈值
    
    # 模式库大小（每个size保留的top-N）
    pattern_top_n_size1: int = 10             # 1层模式保留10个
    pattern_top_n_size2: int = 15             # 2层模式保留15个
    pattern_top_n_size3: int = 10             # 3层模式保留10个
    
    # 模式引导强度（动态调整）
    pattern_guided_prob_initial: float = 0.2  # 初始引导概率
    pattern_guided_prob_increment: float = 0.05  # 每次更新增加
    pattern_guided_prob_max: float = 0.6      # 最大引导概率
    
    # ========== 其他 ==========
    random_seed: Optional[int] = 42  # 随机种子（None表示不固定）
    verbose: bool = True              # 是否打印详细信息
    
    def __post_init__(self):
        """参数验证"""
        assert 1 <= self.min_layers <= self.max_layers <= self.num_layers, \
            f"Invalid layer constraints: {self.min_layers} <= layers <= {self.max_layers} (total: {self.num_layers})"
        assert 0 < self.crossover_rate <= 1, f"Invalid crossover_rate: {self.crossover_rate}"
        assert 0 < self.mutation_rate <= 1, f"Invalid mutation_rate: {self.mutation_rate}"
        assert self.elite_size < self.population_size, \
            f"Elite size ({self.elite_size}) must be less than population size ({self.population_size})"
        assert self.tournament_size >= 2, f"Tournament size must be >= 2, got {self.tournament_size}"


# 默认配置
DEFAULT_CONFIG = GAConfig()

