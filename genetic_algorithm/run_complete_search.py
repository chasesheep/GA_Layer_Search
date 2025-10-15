"""
完整的三阶段搜索流程
"""
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Dict
import logging
import numpy as np

from config import GAConfig
from ga_core import GeneticAlgorithm
from local_search_twostage import TwoStageLocalSearch
from fitness import create_analytical_mock_fitness
from utils import load_single_layer_results, load_known_best_solutions


class DualOutput:
    """同时输出到控制台和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class CompleteSearchPipeline:
    """完整的搜索流程管道"""
    
    def __init__(self,
                 fast_fitness_func: Callable,  # limit=50
                 full_fitness_func: Callable,  # limit=None  
                 single_layer_scores: Dict[int, float],
                 known_best_solutions: List[List[int]],
                 config: GAConfig):
        """
        Args:
            fast_fitness_func: 快速评估函数（用于GA和局部搜索筛选）
            full_fitness_func: 完整评估函数（用于最终验证）
            single_layer_scores: 单层分数
            known_best_solutions: 已知1-2层最优解
            config: GA配置
        """
        self.fast_fitness = fast_fitness_func
        self.full_fitness = full_fitness_func
        self.single_layer_scores = single_layer_scores
        self.known_best_solutions = known_best_solutions
        self.config = config
        
        # 统计
        self.stats = {
            'phase1_evaluations': 0,
            'phase2_evaluations': 0,
            'phase3_fast_evaluations': 0,
            'phase3_full_evaluations': 0,
        }
        
        # 存储GA结果用于后续分析
        self.ga_results = None
        self.ga_instance = None
    
    def phase1_ga_search(self, top_k: int = 20, checkpoint_dir: Path = None) -> List[Tuple[List[int], float]]:
        """
        阶段1: GA粗搜索
        
        Args:
            top_k: 返回前k个候选解
            checkpoint_dir: 检查点保存目录
        
        Returns:
            [(层组合, 粗评估分数), ...]
        """
        print(f"\n{'='*70}")
        print(f"阶段1: GA粗搜索 (使用快速评估)")
        print(f"{'='*70}")
        
        # 创建GA
        ga = GeneticAlgorithm(
            config=self.config,
            fitness_func=self.fast_fitness,  # 使用快速评估
            single_layer_scores=self.single_layer_scores,
            known_best_solutions=self.known_best_solutions
        )
        
        # 设置检查点目录
        if checkpoint_dir:
            ga.checkpoint_dir = checkpoint_dir
            ga.save_checkpoint_interval = 3  # 每3代保存一次
        
        # 运行GA
        ga_results = ga.run()
        
        # 保存GA实例和结果用于后续分析
        self.ga_instance = ga
        self.ga_results = ga_results
        
        # 记录统计
        self.stats['phase1_evaluations'] = ga_results.total_evaluations
        
        # 收集top-k候选
        candidates = []
        
        # 全局最优
        candidates.append((ga_results.best_layers, ga_results.best_fitness))
        
        # 各层数最优
        for num_layers, individual in ga_results.best_by_layer_count.items():
            combo = (individual.layers, individual.fitness)
            if combo not in candidates:
                candidates.append(combo)
        
        # 从GA的最终种群中获取更多候选
        ga.population.sort_by_fitness()
        for ind in ga.population.individuals:
            combo = (ind.layers, ind.fitness)
            if combo not in candidates and len(candidates) < top_k:
                candidates.append(combo)
        
        # 按粗评估分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]
        
        print(f"\n阶段1完成:")
        print(f"  找到{len(candidates)}个候选解")
        print(f"  评估次数: {self.stats['phase1_evaluations']}次")
        print(f"  适应度范围: {candidates[0][1]:.4f} - {candidates[-1][1]:.4f}")
        
        return candidates
    
    def phase2_full_evaluation(self, candidates: List[Tuple[List[int], float]], 
                              top_k: int = 10) -> List[Tuple[List[int], float]]:
        """
        阶段2: 完整评估候选解
        
        Args:
            candidates: [(层组合, 粗评估分数), ...]
            top_k: 保留前k个
        
        Returns:
            [(层组合, 完整评估分数), ...] 按完整评估排序
        """
        print(f"\n{'='*70}")
        print(f"阶段2: 完整评估Top-{len(candidates)}候选解")
        print(f"{'='*70}")
        
        results = []
        
        for i, (layers, fast_score) in enumerate(candidates, 1):
            # 完整评估
            full_score = self.full_fitness(layers)
            self.stats['phase2_evaluations'] += 1
            
            results.append((layers, full_score))
            
            gap = full_score - fast_score
            print(f"  {i:2d}. {str(layers):30s} "
                  f"粗={fast_score:.4f}, 完整={full_score:.4f} (Δ={gap:+.4f})")
        
        # 按完整评估分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n阶段2完成:")
        print(f"  完整评估次数: {self.stats['phase2_evaluations']}次")
        print(f"  真实Top-{min(top_k, len(results))}:")
        for i, (layers, score) in enumerate(results[:top_k], 1):
            print(f"    {i}. {layers}: {score:.4f}")
        
        return results[:top_k]
    
    def phase3_local_refinement(self, candidates: List[Tuple[List[int], float]], 
                                top_k: int = 3) -> List[Tuple[List[int], float]]:
        """
        阶段3: 局部精细优化
        
        对真实top-k解进行两阶段局部搜索
        
        Args:
            candidates: [(层组合, 完整评估分数), ...]
            top_k: 对前k个进行局部搜索
        
        Returns:
            [(优化后层组合, 完整评估分数), ...]
        """
        print(f"\n{'='*70}")
        print(f"阶段3: 局部精细优化 (两阶段搜索)")
        print(f"{'='*70}")
        print(f"对Top-{min(top_k, len(candidates))}解进行邻域优化")
        
        optimized_results = []
        
        for i, (initial_layers, initial_full_score) in enumerate(candidates[:top_k], 1):
            print(f"\n--- 优化候选解{i}: {initial_layers} (fitness={initial_full_score:.4f}) ---")
            
            # 创建两阶段局部搜索器
            local_searcher = TwoStageLocalSearch(
                fast_fitness_func=self.fast_fitness,
                full_fitness_func=self.full_fitness,
                verbose=True
            )
            
            # 执行两阶段局部搜索
            optimized_layers, optimized_full_score = local_searcher.two_stage_hill_climbing(
                initial_layers=initial_layers,
                initial_fitness_fast=None,  # 会自动计算
                max_iterations=10,
                top_k_to_verify=5  # 每轮选5个邻居完整评估
            )
            
            optimized_results.append((optimized_layers, optimized_full_score))
            
            # 记录统计
            self.stats['phase3_fast_evaluations'] += local_searcher.fast_evaluations
            self.stats['phase3_full_evaluations'] += local_searcher.full_evaluations
            
            improvement = optimized_full_score - initial_full_score
            if improvement > 0:
                print(f"  ✓ 找到改进: {improvement:.4f}")
            else:
                print(f"  ✓ 已是局部最优")
        
        # 去重并排序
        unique_results = {}
        for layers, score in optimized_results:
            key = tuple(sorted(layers))
            if key not in unique_results or score > unique_results[key]:
                unique_results[key] = (layers, score)
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n阶段3完成:")
        print(f"  粗评估次数: {self.stats['phase3_fast_evaluations']}次")
        print(f"  完整评估次数: {self.stats['phase3_full_evaluations']}次")
        print(f"  去重后解数: {len(final_results)}")
        
        return final_results
    
    def run(self, output_dir: Path = None) -> Dict:
        """
        运行完整的三阶段搜索
        
        Args:
            output_dir: 输出目录
        
        Returns:
            完整结果字典
        """
        start_time = time.time()
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 设置输出目录
        if output_dir is None:
            output_dir = Path("results/mock_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件
        log_file = output_dir / f"search_log_{timestamp_str}.txt"
        
        # 重定向输出到文件和控制台
        dual_output = DualOutput(log_file)
        old_stdout = sys.stdout
        sys.stdout = dual_output
        
        try:
            print(f"\n{'='*70}")
            print(f"🧬 完整搜索流程")
            print(f"{'='*70}")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"日志文件: {log_file}")
            
            # 创建检查点目录
            checkpoint_dir = output_dir / f"checkpoints_{timestamp_str}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"检查点目录: {checkpoint_dir}")
            
            # 阶段1: GA粗搜索
            phase1_candidates = self.phase1_ga_search(top_k=20, checkpoint_dir=checkpoint_dir)
            
            # 阶段2: 完整评估
            phase2_candidates = self.phase2_full_evaluation(phase1_candidates, top_k=10)
            
            # 阶段3: 局部优化
            final_results = self.phase3_local_refinement(phase2_candidates, top_k=3)
            
            # 总结
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n{'='*70}")
            print(f"🎯 完整搜索完成")
            print(f"{'='*70}")
            
            print(f"\n最终Top-5解:")
            for i, (layers, score) in enumerate(final_results[:5], 1):
                layer_str = str(layers).ljust(25)
                print(f"  {i}. {layer_str} fitness={score:.4f} ({len(layers)}层)")
            
            # 输出发现的模式
            if self.config.pattern_mining_enabled and self.ga_instance and self.ga_instance.pattern_miner:
                print(f"\n发现的优秀模式:")
                pattern_miner = self.ga_instance.pattern_miner
                
                for size in [1, 2, 3]:
                    patterns = pattern_miner.get_top_patterns(size, top_k=5)
                    if patterns:
                        print(f"\n  {size}层模式 (Top-5):")
                        for j, pattern in enumerate(patterns, 1):
                            layers_str = str(list(pattern.layers)).ljust(20)
                            print(f"    {j}. {layers_str} "
                                  f"freq={pattern.frequency:2d}, "
                                  f"avg_fitness={pattern.avg_fitness:.4f}, "
                                  f"quality={pattern.avg_fitness * np.log(1 + pattern.frequency):.3f}")
            
            # 输出各层数最优解
            if self.ga_results:
                print(f"\n各层数最优解:")
                for num_layers in sorted(self.ga_results.best_by_layer_count.keys()):
                    ind = self.ga_results.best_by_layer_count[num_layers]
                    layers_str = str(ind.layers).ljust(25)
                    print(f"  {num_layers}层: {layers_str} fitness={ind.fitness:.4f}")
            
            print(f"\n评估统计:")
            print(f"  阶段1 (GA粗搜索):        {self.stats['phase1_evaluations']:4d}次 (快速评估)")
            print(f"  阶段2 (完整评估候选):     {self.stats['phase2_evaluations']:4d}次 (完整评估)")
            print(f"  阶段3 (局部优化-粗):      {self.stats['phase3_fast_evaluations']:4d}次 (快速评估)")
            print(f"  阶段3 (局部优化-完整):    {self.stats['phase3_full_evaluations']:4d}次 (完整评估)")
            print(f"  ────────────────────────────────")
            total_fast = self.stats['phase1_evaluations'] + self.stats['phase3_fast_evaluations']
            total_full = self.stats['phase2_evaluations'] + self.stats['phase3_full_evaluations']
            print(f"  总计 - 快速评估:         {total_fast:4d}次")
            print(f"  总计 - 完整评估:         {total_full:4d}次")
            print(f"  总耗时: {total_time:.2f}秒")
            
            print(f"{'='*70}\n")
            
            # 提取模式信息
            patterns_info = {}
            if self.config.pattern_mining_enabled and self.ga_instance and self.ga_instance.pattern_miner:
                pattern_miner = self.ga_instance.pattern_miner
                for size in [1, 2, 3]:
                    patterns = pattern_miner.get_top_patterns(size, top_k=10)
                    if patterns:
                        patterns_info[f'{size}_layer_patterns'] = [
                            {
                                'layers': list(p.layers),
                                'frequency': p.frequency,
                                'avg_fitness': float(p.avg_fitness),
                                'quality_score': float(p.avg_fitness * np.log(1 + p.frequency))
                            }
                            for p in patterns
                        ]
            
            # 提取各层数最优解
            best_by_layer_count = {}
            if self.ga_results:
                for num_layers, ind in self.ga_results.best_by_layer_count.items():
                    best_by_layer_count[str(num_layers)] = {
                        'layers': ind.layers,
                        'fitness': float(ind.fitness)
                    }
            
            # 构建详细结果
            result_dict = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'timestamp_str': timestamp_str,
                    'total_time': total_time,
                    'config': {
                        'population_size': self.config.population_size,
                        'max_generations': self.config.max_generations,
                        'no_improvement_threshold': self.config.no_improvement_threshold,
                        'elite_size': self.config.elite_size,
                        'pattern_mining_enabled': self.config.pattern_mining_enabled,
                        'random_seed': self.config.random_seed,
                    }
                },
                'final_results': [
                    {'layers': layers, 'fitness': float(score), 'num_layers': len(layers)}
                    for layers, score in final_results
                ],
                'best_by_layer_count': best_by_layer_count,
                'discovered_patterns': patterns_info,
                'phase1_top20_candidates': [
                    {'layers': layers, 'fast_fitness': float(score)}
                    for layers, score in phase1_candidates
                ],
                'phase2_top10_verified': [
                    {'layers': layers, 'full_fitness': float(score)}
                    for layers, score in phase2_candidates
                ],
                'statistics': {
                    'phase1_evaluations': self.stats['phase1_evaluations'],
                    'phase2_evaluations': self.stats['phase2_evaluations'],
                    'phase3_fast_evaluations': self.stats['phase3_fast_evaluations'],
                    'phase3_full_evaluations': self.stats['phase3_full_evaluations'],
                    'total_fast_evaluations': total_fast,
                    'total_full_evaluations': total_full,
                },
                'files': {
                    'log_file': str(log_file),
                    'result_file': str(output_dir / f"search_result_{timestamp_str}.json"),
                }
            }
            
            # 保存JSON结果
            result_file = output_dir / f"search_result_{timestamp_str}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            print(f"结果已保存:")
            print(f"  - JSON: {result_file}")
            print(f"  - 日志: {log_file}")
            print()
            
            return result_dict
        
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            dual_output.close()


def test_complete_pipeline_mock():
    """用Mock函数测试完整流程"""
    print("\n")
    print("🧬 测试完整搜索流程（Mock函数）")
    print("=" * 70)
    
    # 加载数据
    single_layer_data = load_single_layer_results()
    single_layer_scores = {lid: data['score'] for lid, data in single_layer_data.items()}
    known_solutions = load_known_best_solutions()
    known_best_layers = [layers for layers, score in known_solutions]
    
    # 创建fitness函数（暂时用同一个mock函数，实际使用时会不同）
    fast_fitness = create_analytical_mock_fitness()
    full_fitness = create_analytical_mock_fitness()  # 暂时相同
    
    # 配置
    config = GAConfig(
        population_size=40,
        max_generations=20,
        no_improvement_threshold=6,
        elite_size=3,
        verbose=True,
        pattern_mining_enabled=True,
        pattern_update_interval=5,
        random_seed=42,
    )
    
    # 创建搜索管道
    pipeline = CompleteSearchPipeline(
        fast_fitness_func=fast_fitness,
        full_fitness_func=full_fitness,
        single_layer_scores=single_layer_scores,
        known_best_solutions=known_best_layers,
        config=config
    )
    
    # 运行完整搜索
    results = pipeline.run()
    
    # 保存结果
    output_dir = Path("results/mock_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "complete_search_result.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存到: {output_file}")
    
    return results


def main():
    """主函数"""
    try:
        results = test_complete_pipeline_mock()
        
        print("\n" * 2)
        print("🎉 " + "=" * 66 + " 🎉")
        print("   完整搜索流程测试通过！")
        print("🎉 " + "=" * 66 + " 🎉")
        print("\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

