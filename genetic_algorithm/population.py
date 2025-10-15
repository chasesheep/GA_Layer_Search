"""
种群类 - 管理个体集合
"""
from typing import List, Callable, Dict, Optional
import numpy as np
from individual import Individual


class Population:
    """遗传算法种群"""
    
    def __init__(self, size: int, num_layers: int = 32):
        """
        Args:
            size: 种群大小
            num_layers: 总层数
        """
        self.size = size
        self.num_layers = num_layers
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Individual] = []  # 每代最优个体历史
    
    def initialize_random(self, min_layers: int, max_layers: int, 
                          rng: Optional[np.random.Generator] = None):
        """
        随机初始化种群
        
        Args:
            min_layers: 最少替换层数
            max_layers: 最多替换层数
            rng: 随机数生成器
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.individuals = []
        for _ in range(self.size):
            individual = Individual.random(
                num_layers=self.num_layers,
                min_layers=min_layers,
                max_layers=max_layers,
                rng=rng
            )
            self.individuals.append(individual)
    
    def initialize_smart(self, min_layers: int, max_layers: int,
                        single_layer_scores: Dict[int, float],
                        known_best_solutions: List[List[int]] = None,
                        rng: Optional[np.random.Generator] = None):
        """
        智能初始化种群，利用已知结果
        
        Args:
            min_layers: 最少替换层数
            max_layers: 最多替换层数
            single_layer_scores: 单层分数字典 {layer_id: score}
            known_best_solutions: 已知最优解列表
            rng: 随机数生成器
        """
        if rng is None:
            rng = np.random.default_rng()
        
        self.individuals = []
        
        # 获取top单层
        top_layers = sorted(single_layer_scores.items(), key=lambda x: x[1], reverse=True)
        top_layer_ids = [layer_id for layer_id, _ in top_layers[:10]]
        
        # 1. 精英种子 (10%)
        elite_count = max(1, int(self.size * 0.1))
        elite_added = 0
        
        if known_best_solutions:
            for solution in known_best_solutions[:elite_count]:
                if min_layers <= len(solution) <= max_layers:
                    ind = Individual.from_layers(solution, num_layers=self.num_layers)
                    self.individuals.append(ind)
                    elite_added += 1
                    if elite_added >= elite_count:
                        break
        
        # 如果精英种子不够，添加一些基础的好解
        if elite_added < elite_count:
            # 添加单层17
            if min_layers <= 1 <= max_layers:
                self.individuals.append(Individual.from_layers([17], num_layers=self.num_layers))
                elite_added += 1
            
            # 添加尾部探索
            remaining = elite_count - elite_added
            tail_solutions = [[17, 30], [17, 31], [14, 17, 30], [13, 17, 31]]
            for solution in tail_solutions[:remaining]:
                if min_layers <= len(solution) <= max_layers:
                    ind = Individual.from_layers(solution, num_layers=self.num_layers)
                    self.individuals.append(ind)
                    elite_added += 1
        
        # 2. 启发式构建 (40%) - 基于单层分数加权采样
        heuristic_count = int(self.size * 0.4)
        
        # 计算采样权重：基于单层分数
        layer_ids = list(single_layer_scores.keys())
        layer_scores = np.array([single_layer_scores[lid] for lid in layer_ids])
        
        # 归一化为概率分布（分数越高，被选中概率越大）
        # 使用平方来增强高分层的权重
        weights = layer_scores ** 2
        weights = weights / weights.sum()
        
        for _ in range(heuristic_count):
            # 随机选择层数（倾向更多层）
            # 使用权重：2层(20%), 3层(40%), 4层(40%)
            layer_count_probs = [0.0, 0.0, 0.2, 0.4, 0.4]  # index 0-4对应0-4层
            num_layers_to_select = rng.choice(range(5), p=layer_count_probs)
            num_layers_to_select = max(min_layers, min(num_layers_to_select, max_layers))
            
            # 按权重采样（无放回）
            selected_indices = rng.choice(
                len(layer_ids), 
                size=num_layers_to_select, 
                replace=False,
                p=weights
            )
            selected_layers = [layer_ids[i] for i in selected_indices]
            
            # 创建个体
            if min_layers <= len(selected_layers) <= max_layers:
                ind = Individual.from_layers(selected_layers, num_layers=self.num_layers)
                self.individuals.append(ind)
        
        # 3. 随机探索 (50%)
        random_count = self.size - len(self.individuals)
        
        for _ in range(random_count):
            individual = Individual.random(
                num_layers=self.num_layers,
                min_layers=min_layers,
                max_layers=max_layers,
                rng=rng
            )
            self.individuals.append(individual)
        
        # 确保种群大小正确
        assert len(self.individuals) == self.size, \
            f"Population size mismatch: {len(self.individuals)} != {self.size}"
    
    def add_individual(self, individual: Individual):
        """添加个体到种群"""
        self.individuals.append(individual)
    
    def evaluate_all(self, fitness_func: Callable[[List[int]], float], 
                     skip_evaluated: bool = True):
        """
        评估所有个体的适应度
        
        Args:
            fitness_func: 适应度函数，输入层列表，输出适应度分数
            skip_evaluated: 是否跳过已评估的个体
        """
        for individual in self.individuals:
            if skip_evaluated and individual.evaluated:
                continue
            
            fitness = fitness_func(individual.layers)
            individual.set_fitness(fitness)
    
    def sort_by_fitness(self):
        """按适应度降序排序"""
        self.individuals.sort(reverse=True)  # 使用Individual的__lt__方法
    
    def update_best(self):
        """更新最优个体"""
        self.sort_by_fitness()
        current_best = self.individuals[0]
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        # 记录历史
        self.history.append(current_best.copy())
    
    def get_top_k(self, k: int) -> List[Individual]:
        """
        获取适应度最高的k个个体
        
        Args:
            k: 个体数量
        
        Returns:
            top-k个体列表（副本）
        """
        self.sort_by_fitness()
        return [ind.copy() for ind in self.individuals[:k]]
    
    def get_by_layer_count(self) -> Dict[int, List[Individual]]:
        """
        按层数分组个体
        
        Returns:
            字典，key为层数，value为该层数的个体列表
        """
        grouped = {}
        for individual in self.individuals:
            num_layers = individual.num_replaced_layers
            if num_layers not in grouped:
                grouped[num_layers] = []
            grouped[num_layers].append(individual)
        
        # 每组内按适应度排序
        for layer_count in grouped:
            grouped[layer_count].sort(reverse=True)
        
        return grouped
    
    def get_statistics(self) -> Dict:
        """
        获取种群统计信息
        
        Returns:
            统计字典
        """
        if not self.individuals:
            return {
                'generation': self.generation,
                'size': 0,
                'best_fitness': None,
                'avg_fitness': None,
                'worst_fitness': None,
            }
        
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        
        # 按层数统计
        by_layer_count = self.get_by_layer_count()
        layer_stats = {}
        for num_layers, individuals in by_layer_count.items():
            best_ind = max(individuals, key=lambda x: x.fitness if x.fitness else -float('inf'))
            layer_stats[num_layers] = {
                'count': len(individuals),
                'best_fitness': best_ind.fitness,
                'best_layers': best_ind.layers
            }
        
        return {
            'generation': self.generation,
            'size': len(self.individuals),
            'best_fitness': max(fitnesses) if fitnesses else None,
            'avg_fitness': np.mean(fitnesses) if fitnesses else None,
            'worst_fitness': min(fitnesses) if fitnesses else None,
            'by_layer_count': layer_stats,
        }
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Population(gen={stats['generation']}, size={stats['size']}, "
                f"best={stats['best_fitness']:.4f if stats['best_fitness'] else 'N/A'})")

