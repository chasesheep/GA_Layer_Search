"""
局部搜索 - 对GA找到的最优解进行邻域优化
"""
from typing import List, Callable, Tuple, Set
import numpy as np
from individual import Individual


class LocalSearch:
    """局部搜索器"""
    
    def __init__(self, 
                 fitness_func: Callable[[List[int]], float],
                 min_layers: int = 2,
                 max_layers: int = 4,
                 num_layers: int = 32,
                 verbose: bool = True):
        """
        Args:
            fitness_func: 适应度函数
            min_layers: 最少层数
            max_layers: 最多层数
            num_layers: 总层数
            verbose: 是否打印详细信息
        """
        self.fitness_func = fitness_func
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.num_layers = num_layers
        self.verbose = verbose
        
        # 统计信息
        self.total_evaluations = 0
        self.evaluated_combinations = set()
    
    def generate_neighbors(self, layers: List[int]) -> List[Tuple[str, List[int]]]:
        """
        生成邻域解
        
        对于层组合 [a, b, c]，邻域包括：
        1. 替换操作：[x, b, c], [a, x, c], [a, b, x] (删a加x, 删b加x, ...)
        2. 添加操作：[a, b, c, x] (如果未达到max_layers)
        3. 删除操作：[b, c], [a, c], [a, b] (如果未低于min_layers)
        
        Args:
            layers: 当前层列表
        
        Returns:
            [(操作描述, 新层列表), ...]
        """
        neighbors = []
        current_set = set(layers)
        num_current = len(layers)
        
        # 1. 替换操作（对每一层，尝试替换为其他层）
        for i, layer_to_remove in enumerate(layers):
            for layer_to_add in range(self.num_layers):
                if layer_to_add not in current_set:
                    new_layers = layers[:i] + [layer_to_add] + layers[i+1:]
                    new_layers = sorted(new_layers)
                    operation = f"替换{layer_to_remove}→{layer_to_add}"
                    neighbors.append((operation, new_layers))
        
        # 2. 添加操作（如果未达到max_layers）
        if num_current < self.max_layers:
            for layer_to_add in range(self.num_layers):
                if layer_to_add not in current_set:
                    new_layers = sorted(layers + [layer_to_add])
                    operation = f"添加{layer_to_add}"
                    neighbors.append((operation, new_layers))
        
        # 3. 删除操作（如果未低于min_layers）
        if num_current > self.min_layers:
            for i, layer_to_remove in enumerate(layers):
                new_layers = layers[:i] + layers[i+1:]
                operation = f"删除{layer_to_remove}"
                neighbors.append((operation, new_layers))
        
        return neighbors
    
    def hill_climbing(self, initial_layers: List[int], max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        爬山算法：贪心的局部搜索
        
        Args:
            initial_layers: 初始解
            max_iterations: 最大迭代次数
        
        Returns:
            (最优层组合, 最优适应度)
        """
        current_layers = initial_layers
        current_fitness = self.fitness_func(current_layers)
        self.total_evaluations += 1
        self.evaluated_combinations.add(tuple(sorted(current_layers)))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"局部搜索（爬山算法）")
            print(f"{'='*70}")
            print(f"初始解: {current_layers}, fitness={current_fitness:.4f}")
        
        iteration = 0
        improvement_history = []
        
        while iteration < max_iterations:
            # 生成所有邻域解
            neighbors = self.generate_neighbors(current_layers)
            
            if self.verbose and iteration == 0:
                print(f"\n邻域大小: {len(neighbors)}")
            
            # 评估所有邻域解
            best_neighbor = None
            best_neighbor_fitness = current_fitness
            
            for operation, neighbor_layers in neighbors:
                # 跳过已评估的组合
                combo_tuple = tuple(sorted(neighbor_layers))
                if combo_tuple in self.evaluated_combinations:
                    continue
                
                # 评估
                fitness = self.fitness_func(neighbor_layers)
                self.total_evaluations += 1
                self.evaluated_combinations.add(combo_tuple)
                
                # 更新最优邻居
                if fitness > best_neighbor_fitness:
                    best_neighbor = neighbor_layers
                    best_neighbor_fitness = fitness
                    best_operation = operation
            
            # 如果找到更好的邻居，移动到该邻居
            if best_neighbor is not None:
                improvement = best_neighbor_fitness - current_fitness
                current_layers = best_neighbor
                current_fitness = best_neighbor_fitness
                iteration += 1
                improvement_history.append((iteration, current_layers, current_fitness, best_operation))
                
                if self.verbose:
                    print(f"  迭代{iteration}: {best_operation:15s} → {current_layers}, "
                          f"fitness={current_fitness:.4f} (+{improvement:.4f})")
            else:
                # 没有更好的邻居，达到局部最优
                if self.verbose:
                    print(f"\n达到局部最优（无更好的邻居）")
                break
        
        if self.verbose:
            if iteration >= max_iterations:
                print(f"\n达到最大迭代次数 ({max_iterations})")
            
            print(f"\n局部搜索完成:")
            print(f"  最终解: {current_layers}")
            print(f"  适应度: {current_fitness:.4f}")
            print(f"  改进: {current_fitness - self.fitness_func(initial_layers):.4f}")
            print(f"  迭代次数: {len(improvement_history)}")
            print(f"  评估次数: {self.total_evaluations}")
            print(f"{'='*70}")
        
        return current_layers, current_fitness
    
    def first_improvement_search(self, initial_layers: List[int], max_iterations: int = 50) -> Tuple[List[int], float]:
        """
        首次改进搜索：找到第一个改进的邻居就接受
        
        比爬山算法更快，但可能陷入更差的局部最优
        
        Args:
            initial_layers: 初始解
            max_iterations: 最大迭代次数
        
        Returns:
            (最优层组合, 最优适应度)
        """
        current_layers = initial_layers
        current_fitness = self.fitness_func(current_layers)
        self.total_evaluations += 1
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"局部搜索（首次改进）")
            print(f"{'='*70}")
            print(f"初始解: {current_layers}, fitness={current_fitness:.4f}")
        
        iteration = 0
        
        while iteration < max_iterations:
            # 生成邻域
            neighbors = self.generate_neighbors(current_layers)
            
            # 随机打乱邻域顺序
            rng = np.random.default_rng(iteration)
            rng.shuffle(neighbors)
            
            # 寻找第一个改进的邻居
            found_improvement = False
            
            for operation, neighbor_layers in neighbors:
                # 跳过已评估的
                combo_tuple = tuple(sorted(neighbor_layers))
                if combo_tuple in self.evaluated_combinations:
                    continue
                
                # 评估
                fitness = self.fitness_func(neighbor_layers)
                self.total_evaluations += 1
                self.evaluated_combinations.add(combo_tuple)
                
                # 找到改进
                if fitness > current_fitness:
                    improvement = fitness - current_fitness
                    current_layers = neighbor_layers
                    current_fitness = fitness
                    iteration += 1
                    found_improvement = True
                    
                    if self.verbose:
                        print(f"  迭代{iteration}: {operation:15s} → {current_layers}, "
                              f"fitness={current_fitness:.4f} (+{improvement:.4f})")
                    break
            
            if not found_improvement:
                if self.verbose:
                    print(f"\n达到局部最优")
                break
        
        if self.verbose:
            print(f"\n局部搜索完成:")
            print(f"  最终解: {current_layers}")
            print(f"  适应度: {current_fitness:.4f}")
            print(f"  迭代次数: {iteration}")
            print(f"  评估次数: {self.total_evaluations}")
            print(f"{'='*70}")
        
        return current_layers, current_fitness


def perform_local_search_on_top_solutions(ga_results,
                                          fitness_func: Callable,
                                          top_k: int = 3,
                                          method: str = 'hill_climbing',
                                          verbose: bool = True) -> List[Tuple[List[int], float]]:
    """
    对GA找到的top-k解进行局部搜索
    
    Args:
        ga_results: GA运行结果
        fitness_func: 适应度函数
        top_k: 对前k个解进行局部搜索
        method: 搜索方法 ('hill_climbing' 或 'first_improvement')
        verbose: 是否打印详细信息
    
    Returns:
        [(优化后的层组合, 适应度), ...]
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"对Top-{top_k}解进行局部优化")
        print(f"{'='*70}")
    
    # 收集候选解
    candidates = []
    
    # 添加全局最优
    candidates.append((ga_results.best_layers, ga_results.best_fitness, "全局最优"))
    
    # 添加各层数最优
    for num_layers, individual in ga_results.best_by_layer_count.items():
        if individual.layers not in [c[0] for c in candidates]:
            candidates.append((individual.layers, individual.fitness, f"{num_layers}层最优"))
    
    # 按适应度排序，取top-k
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:top_k]
    
    if verbose:
        print(f"\n待优化的候选解:")
        for i, (layers, fitness, desc) in enumerate(candidates, 1):
            print(f"  {i}. {desc}: {layers}, fitness={fitness:.4f}")
    
    # 对每个候选解进行局部搜索
    optimized_results = []
    
    for i, (initial_layers, initial_fitness, desc) in enumerate(candidates, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"优化候选解{i}: {desc}")
            print(f"{'='*70}")
        
        # 创建局部搜索器
        local_searcher = LocalSearch(
            fitness_func=fitness_func,
            min_layers=2,
            max_layers=4,
            num_layers=32,
            verbose=verbose
        )
        
        # 执行局部搜索
        if method == 'hill_climbing':
            optimized_layers, optimized_fitness = local_searcher.hill_climbing(
                initial_layers, max_iterations=20
            )
        else:
            optimized_layers, optimized_fitness = local_searcher.first_improvement_search(
                initial_layers, max_iterations=20
            )
        
        optimized_results.append((optimized_layers, optimized_fitness))
        
        if verbose:
            improvement = optimized_fitness - initial_fitness
            if improvement > 0:
                print(f"  ✓ 找到改进: {improvement:.4f}")
            else:
                print(f"  ✓ 已是局部最优")
    
    return optimized_results

