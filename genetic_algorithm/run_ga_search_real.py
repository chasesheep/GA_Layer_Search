#!/usr/bin/env python3
"""
使用真实MMLU评估运行GA搜索
"""
import sys
import argparse
from pathlib import Path

from config import GAConfig
from run_complete_search import CompleteSearchPipeline
from real_fitness import create_mmlu_fitness, MMLUFitnessFunction
from utils import load_single_layer_results, load_known_best_solutions
from path_config import get_llama_layers_dir


def main():
    parser = argparse.ArgumentParser(description='GA搜索 - 真实MMLU评估')
    
    # GPU设置
    parser.add_argument('--gpu', type=int, default=3, help='GPU ID (default: 3)')
    
    # 搜索参数
    parser.add_argument('--fast-limit', type=int, default=50, 
                       help='阶段1和3的快速评估limit (default: 50)')
    parser.add_argument('--full-limit', type=int, default=None,
                       help='阶段2完整评估limit (default: None, 即完整MMLU)')
    
    # GA参数
    parser.add_argument('--population', type=int, default=40, help='种群大小 (default: 40)')
    parser.add_argument('--generations', type=int, default=20, help='最大代数 (default: 20)')
    parser.add_argument('--no-improve', type=int, default=6, help='无改进终止阈值 (default: 6)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (default: 42)')
    
    # 输出设置
    parser.add_argument('--output-dir', type=str, default='results/real_results',
                       help='输出目录 (default: results/real_results)')
    parser.add_argument('--verbose', action='store_true', help='打印详细信息')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧬 GA搜索 - 真实MMLU评估")
    print("="*70)
    print(f"\nGPU: {args.gpu}")
    print(f"快速评估limit: {args.fast_limit}")
    print(f"完整评估limit: {'None (完整MMLU)' if args.full_limit is None else args.full_limit}")
    print(f"种群大小: {args.population}")
    print(f"最大代数: {args.generations}")
    print(f"输出目录: {args.output_dir}")
    
    # 设置路径（从path_config获取）
    llama_layers_dir = get_llama_layers_dir()
    print(f"  - Llama层文件目录: {llama_layers_dir}")
    
    # 加载数据
    print(f"\n加载数据...")
    single_layer_data = load_single_layer_results()
    single_layer_scores = {lid: data['score'] for lid, data in single_layer_data.items()}
    known_solutions = load_known_best_solutions()
    known_best_layers = [layers for layers, score in known_solutions]
    
    print(f"  - 单层结果: {len(single_layer_scores)}个")
    print(f"  - 已知最优解: {len(known_best_layers)}个")
    
    # 创建fitness函数
    print(f"\n创建MMLU评估器...")
    
    # 快速评估函数 (limit=50)
    fast_fitness = create_mmlu_fitness(
        llama_layers_dir=llama_layers_dir,
        limit=args.fast_limit,
        gpu_id=args.gpu,
        verbose=args.verbose
    )
    
    # 完整评估函数 (limit=None或指定值)
    full_fitness = MMLUFitnessFunction.create_fitness_function(
        limit=args.full_limit,
        verbose=args.verbose
    )
    
    # 创建GA配置
    config = GAConfig(
        population_size=args.population,
        max_generations=args.generations,
        no_improvement_threshold=args.no_improve,
        elite_size=3,
        verbose=True,
        pattern_mining_enabled=True,
        pattern_update_interval=5,
        random_seed=args.seed,
    )
    
    print(f"\n配置:")
    print(f"  种群大小: {config.population_size}")
    print(f"  最大代数: {config.max_generations}")
    print(f"  无改进阈值: {config.no_improvement_threshold}")
    print(f"  模式挖掘: {'启用' if config.pattern_mining_enabled else '禁用'}")
    
    # 创建搜索管道
    print(f"\n创建搜索管道...")
    pipeline = CompleteSearchPipeline(
        fast_fitness_func=fast_fitness,
        full_fitness_func=full_fitness,
        single_layer_scores=single_layer_scores,
        known_best_solutions=known_best_layers,
        config=config
    )
    
    # 运行搜索
    print(f"\n{'='*70}")
    print("开始搜索...")
    print("="*70)
    
    output_dir = Path(args.output_dir)
    results = pipeline.run(output_dir=output_dir)
    
    # 清理
    print(f"\n清理资源...")
    MMLUFitnessFunction.cleanup()
    
    print(f"\n{'='*70}")
    print("🎉 搜索完成！")
    print("="*70)
    print(f"\n结果文件:")
    print(f"  - JSON: {results['files']['result_file']}")
    print(f"  - 日志: {results['files']['log_file']}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

