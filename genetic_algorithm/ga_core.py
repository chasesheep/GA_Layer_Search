"""
遗传算法核心类
"""
import time
from typing import List, Dict, Callable, Optional
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from config import GAConfig
from individual import Individual
from population import Population
from operators import tournament_selection, reproduce
from operators_guided import reproduce_with_patterns
from pattern_miner import PatternMiner


@dataclass
class GAResults:
    """GA运行结果"""
    best_individual: Individual
    best_fitness: float
    best_layers: List[int]
    
    # 每代统计
    generation_history: List[Dict] = field(default_factory=list)
    
    # 按层数统计的最优解
    best_by_layer_count: Dict[int, Individual] = field(default_factory=dict)
    
    # 运行信息
    total_generations: int = 0
    total_evaluations: int = 0
    total_time: float = 0.0
    termination_reason: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        return {
            'best_individual': {
                'layers': self.best_layers,
                'fitness': self.best_fitness,
            },
            'best_by_layer_count': {
                k: {'layers': v.layers, 'fitness': v.fitness}
                for k, v in self.best_by_layer_count.items()
            },
            'total_generations': self.total_generations,
            'total_evaluations': self.total_evaluations,
            'total_time': self.total_time,
            'termination_reason': self.termination_reason,
            'generation_history': self.generation_history,
        }


class GeneticAlgorithm:
    """遗传算法主类"""
    
    def __init__(self, 
                 config: GAConfig,
                 fitness_func: Callable[[List[int]], float],
                 single_layer_scores: Dict[int, float],
                 known_best_solutions: Optional[List[List[int]]] = None):
        """
        Args:
            config: GA配置
            fitness_func: 适应度函数
            single_layer_scores: 单层分数（用于智能初始化和修复）
            known_best_solutions: 已知最优解（用于初始化）
        """
        self.config = config
        self.fitness_func = fitness_func
        self.single_layer_scores = single_layer_scores
        self.known_best_solutions = known_best_solutions
        
        # 初始化随机数生成器
        if config.random_seed is not None:
            self.rng = np.random.default_rng(config.random_seed)
        else:
            self.rng = np.random.default_rng()
        
        # 创建种群
        self.population = Population(size=config.population_size, 
                                    num_layers=config.num_layers)
        
        # 统计信息
        self.generation = 0
        self.total_evaluations = 0
        self.no_improvement_count = 0
        self.best_fitness_history = []
        
        # 全局最优（按层数分组）
        self.global_best_by_layer_count = {}  # {层数: Individual}
        self.global_best = None
        
        # 搜索空间覆盖监控
        self.layer_coverage = set()  # 记录所有被尝试过的层
        self.combination_cache = set()  # 记录所有评估过的层组合（用于去重）
        
        # 检查点设置
        self.checkpoint_dir = None
        self.save_checkpoint_interval = 3  # 每3代保存一次检查点
        
        # 模式挖掘
        if config.pattern_mining_enabled:
            self.pattern_miner = PatternMiner(
                max_pattern_size=config.pattern_max_size,
                min_frequency=config.pattern_min_frequency,
                top_n_per_size={
                    1: config.pattern_top_n_size1,
                    2: config.pattern_top_n_size2,
                    3: config.pattern_top_n_size3,
                }
            )
            self.pattern_guided_prob = config.pattern_guided_prob_initial
        else:
            self.pattern_miner = None
            self.pattern_guided_prob = 0.0
    
    def initialize(self):
        """初始化种群"""
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"初始化种群")
            print(f"{'='*70}")
            print(f"种群大小: {self.config.population_size}")
            print(f"层数约束: {self.config.min_layers}-{self.config.max_layers}")
        
        # 智能初始化
        self.population.initialize_smart(
            min_layers=self.config.min_layers,
            max_layers=self.config.max_layers,
            single_layer_scores=self.single_layer_scores,
            known_best_solutions=self.known_best_solutions,
            rng=self.rng
        )
        
        # 评估初始种群
        self.evaluate_population()
        self.population.update_best()
        self.update_global_best()
        
        if self.config.verbose:
            stats = self.population.get_statistics()
            print(f"\n初始种群统计:")
            print(f"  最佳适应度: {stats['best_fitness']:.4f}")
            print(f"  平均适应度: {stats['avg_fitness']:.4f}")
            print(f"  最优个体: {self.population.best_individual.layers}")
            print(f"  层数分布: ", end="")
            for k, v in sorted(stats['by_layer_count'].items()):
                print(f"{k}层({v['count']}个) ", end="")
            print()
    
    def evaluate_population(self):
        """评估种群中所有未评估的个体"""
        for individual in self.population.individuals:
            if not individual.evaluated:
                # 记录层覆盖
                for layer in individual.layers:
                    self.layer_coverage.add(layer)
                
                # 记录组合
                combo_tuple = tuple(sorted(individual.layers))
                self.combination_cache.add(combo_tuple)
                
                # 评估
                fitness = self.fitness_func(individual.layers)
                individual.set_fitness(fitness)
                self.total_evaluations += 1
    
    def update_global_best(self):
        """更新全局最优解（总体 + 按层数分组）"""
        # 更新总体最优
        current_best = self.population.best_individual
        if self.global_best is None or current_best.fitness > self.global_best.fitness:
            self.global_best = current_best.copy()
        
        # 更新按层数分组的最优
        by_layer_count = self.population.get_by_layer_count()
        for num_layers, individuals in by_layer_count.items():
            best_in_group = max(individuals, key=lambda x: x.fitness)
            
            if num_layers not in self.global_best_by_layer_count:
                self.global_best_by_layer_count[num_layers] = best_in_group.copy()
            else:
                if best_in_group.fitness > self.global_best_by_layer_count[num_layers].fitness:
                    self.global_best_by_layer_count[num_layers] = best_in_group.copy()
    
    def update_patterns(self):
        """更新模式库"""
        if not self.config.pattern_mining_enabled:
            return
        
        # 从top-k个体中挖掘模式
        top_individuals = self.population.get_top_k(self.config.pattern_mining_top_k)
        
        # 清空旧模式
        self.pattern_miner.clear()
        
        # 挖掘新模式
        self.pattern_miner.mine_patterns(top_individuals)
        self.pattern_miner.filter_and_rank_patterns()
        
        # 增加模式引导概率
        if self.pattern_guided_prob < self.config.pattern_guided_prob_max:
            self.pattern_guided_prob = min(
                self.pattern_guided_prob + self.config.pattern_guided_prob_increment,
                self.config.pattern_guided_prob_max
            )
        
        # 打印模式摘要（如果verbose）
        if self.config.verbose:
            print(f"\n  模式更新 (引导概率: {self.pattern_guided_prob:.1%})")
            self.pattern_miner.print_summary(top_k=3)
    
    def evolve_one_generation(self):
        """演化一代"""
        # 1. 选择和繁殖产生新个体
        offspring = []
        
        # 生成 (population_size - elite_size) 个新个体
        num_offspring_needed = self.config.population_size - self.config.elite_size
        
        while len(offspring) < num_offspring_needed:
            # 选择两个父代
            parent1 = tournament_selection(
                self.population.individuals,
                tournament_size=self.config.tournament_size,
                rng=self.rng
            )
            parent2 = tournament_selection(
                self.population.individuals,
                tournament_size=self.config.tournament_size,
                rng=self.rng
            )
            
            # 繁殖（使用模式引导版本）
            if self.config.pattern_mining_enabled and self.pattern_miner:
                child1, child2 = reproduce_with_patterns(
                    parent1, parent2,
                    config=self.config,
                    pattern_miner=self.pattern_miner,
                    single_layer_scores=self.single_layer_scores,
                    guide_prob=self.pattern_guided_prob,
                    rng=self.rng
                )
            else:
                # 使用标准繁殖
                child1, child2 = reproduce(
                    parent1, parent2,
                    config=self.config,
                    single_layer_scores=self.single_layer_scores,
                    rng=self.rng
                )
            
            offspring.append(child1)
            if len(offspring) < num_offspring_needed:
                offspring.append(child2)
        
        # 2. 精英保留：保留当前种群的最优个体
        self.population.sort_by_fitness()
        elites = [ind.copy() for ind in self.population.individuals[:self.config.elite_size]]
        
        # 3. 组成新种群：精英 + 子代
        new_individuals = elites + offspring[:num_offspring_needed]
        
        # 4. 替换种群
        self.population.individuals = new_individuals
        self.population.generation += 1
        self.generation += 1
        
        # 5. 评估新个体
        self.evaluate_population()
        
        # 6. 更新最优
        self.population.update_best()
        prev_global_best_fitness = self.global_best.fitness if self.global_best else 0
        self.update_global_best()
        
        # 7. 检查是否有改进
        new_best_found = self.global_best.fitness > prev_global_best_fitness
        if new_best_found:
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 8. 更新模式（定期或发现新最优解时）
        should_update_patterns = (
            self.config.pattern_mining_enabled and 
            (self.generation % self.config.pattern_update_interval == 0 or new_best_found)
        )
        
        if should_update_patterns:
            self.update_patterns()
        
        # 9. 记录历史
        self.best_fitness_history.append(self.global_best.fitness)
    
    def should_terminate(self) -> tuple[bool, str]:
        """
        检查是否应该终止
        
        Returns:
            (是否终止, 终止原因)
        """
        # 条件1: 达到最大代数
        if self.generation >= self.config.max_generations:
            return True, f"达到最大代数 ({self.config.max_generations})"
        
        # 条件2: 连续N代无改进
        if self.no_improvement_count >= self.config.no_improvement_threshold:
            return True, f"连续{self.config.no_improvement_threshold}代无改进"
        
        # 条件3: 达到目标适应度（如果设置）
        if self.config.target_fitness is not None:
            if self.global_best and self.global_best.fitness >= self.config.target_fitness:
                return True, f"达到目标适应度 ({self.config.target_fitness:.4f})"
        
        return False, ""
    
    def save_checkpoint(self):
        """保存检查点"""
        if self.checkpoint_dir is None:
            return
        
        import json
        from datetime import datetime
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_gen{self.generation:03d}.json"
        
        # 收集检查点数据
        checkpoint_data = {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'total_evaluations': self.total_evaluations,
            'global_best': {
                'layers': self.global_best.layers,
                'fitness': float(self.global_best.fitness)
            },
            'best_by_layer_count': {
                str(k): {'layers': v.layers, 'fitness': float(v.fitness)}
                for k, v in self.global_best_by_layer_count.items()
            },
            'population_top10': [
                {'layers': ind.layers, 'fitness': float(ind.fitness)}
                for ind in self.population.individuals[:10]
            ],
            'statistics': {
                'no_improvement_count': self.no_improvement_count,
                'layer_coverage': list(self.layer_coverage),
                'unique_combinations': len(self.combination_cache),
            }
        }
        
        # 添加模式信息
        if self.config.pattern_mining_enabled and self.pattern_miner:
            patterns_data = {}
            for size in [1, 2, 3]:
                patterns = self.pattern_miner.get_top_patterns(size, top_k=5)
                if patterns:
                    patterns_data[f'{size}_layer'] = [
                        {
                            'layers': list(p.layers),
                            'frequency': p.frequency,
                            'avg_fitness': float(p.avg_fitness)
                        }
                        for p in patterns
                    ]
            checkpoint_data['patterns'] = patterns_data
        
        # 保存
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"  💾 检查点已保存: {checkpoint_file.name}")
    
    def print_generation_summary(self):
        """打印当前代的摘要信息"""
        if not self.config.verbose:
            return
        
        stats = self.population.get_statistics()
        
        print(f"\n代 {self.generation:3d} | "
              f"最优: {self.global_best.fitness:.4f} {self.global_best.layers} | "
              f"当代最优: {stats['best_fitness']:.4f} | "
              f"平均: {stats['avg_fitness']:.4f} | "
              f"无改进: {self.no_improvement_count}")
        
        # 定期保存检查点
        if (self.checkpoint_dir and 
            self.generation > 0 and 
            self.generation % self.save_checkpoint_interval == 0):
            self.save_checkpoint()
        
        # 每10代显示详细的层数统计
        if self.generation % 10 == 0:
            print(f"  各层数最优:")
            for num_layers in sorted(self.global_best_by_layer_count.keys()):
                best = self.global_best_by_layer_count[num_layers]
                print(f"    {num_layers}层: {best.fitness:.4f} {best.layers}")
    
    def run(self) -> GAResults:
        """
        运行完整的遗传算法
        
        Returns:
            GA运行结果
        """
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"🧬 开始遗传算法搜索")
            print(f"{'='*70}")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"配置: population={self.config.population_size}, "
                  f"max_gen={self.config.max_generations}, "
                  f"no_improve={self.config.no_improvement_threshold}")
        
        # 初始化
        self.initialize()
        
        # 主循环
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"开始演化")
            print(f"{'='*70}")
        
        while True:
            # 检查终止条件
            should_stop, reason = self.should_terminate()
            if should_stop:
                if self.config.verbose:
                    print(f"\n终止: {reason}")
                break
            
            # 演化一代
            self.evolve_one_generation()
            
            # 打印摘要
            self.print_generation_summary()
            
            # 记录历史
            stats = self.population.get_statistics()
            self.population.history.append(self.population.best_individual.copy())
        
        # 收集结果
        end_time = time.time()
        total_time = end_time - start_time
        
        # 创建结果对象
        results = GAResults(
            best_individual=self.global_best,
            best_fitness=self.global_best.fitness,
            best_layers=self.global_best.layers,
            best_by_layer_count=self.global_best_by_layer_count.copy(),
            total_generations=self.generation,
            total_evaluations=self.total_evaluations,
            total_time=total_time,
            termination_reason=reason,
            generation_history=[
                {
                    'generation': i,
                    'best_fitness': self.best_fitness_history[i],
                }
                for i in range(len(self.best_fitness_history))
            ]
        )
        
        # 打印最终结果
        if self.config.verbose:
            self.print_final_results(results)
        
        return results
    
    def print_final_results(self, results: GAResults):
        """打印最终结果"""
        print(f"\n{'='*70}")
        print(f"🎯 搜索完成")
        print(f"{'='*70}")
        print(f"总代数: {results.total_generations}")
        print(f"总评估次数: {results.total_evaluations}")
        print(f"评估的不同组合数: {len(self.combination_cache)}")
        print(f"总耗时: {results.total_time:.2f}秒")
        print(f"终止原因: {results.termination_reason}")
        
        print(f"\n全局最优解:")
        print(f"  层组合: {results.best_layers}")
        print(f"  适应度: {results.best_fitness:.4f}")
        
        print(f"\n各层数最优解:")
        for num_layers in sorted(results.best_by_layer_count.keys()):
            best = results.best_by_layer_count[num_layers]
            print(f"  {num_layers}层: {best.fitness:.4f} {best.layers}")
        
        print(f"\n搜索空间覆盖:")
        print(f"  覆盖的层数: {len(self.layer_coverage)} / 32")
        print(f"  覆盖的层: {sorted(self.layer_coverage)}")
        
        # 未覆盖的层
        uncovered = set(range(32)) - self.layer_coverage
        if uncovered:
            print(f"  ⚠️  未覆盖的层 ({len(uncovered)}个): {sorted(uncovered)}")
        else:
            print(f"  ✓ 所有32层都被探索过")
        
        print(f"\n收敛曲线（每10代）:")
        for i in range(0, len(results.generation_history), 10):
            gen_info = results.generation_history[i]
            print(f"  Gen {gen_info['generation']:3d}: {gen_info['best_fitness']:.4f}")
        
        # 最后一代
        if results.total_generations % 10 != 0:
            last_gen = results.generation_history[-1]
            print(f"  Gen {last_gen['generation']:3d}: {last_gen['best_fitness']:.4f}")
        
        print(f"{'='*70}\n")

