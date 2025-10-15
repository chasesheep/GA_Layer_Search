"""
工具函数
"""
import json
from typing import Dict, List, Tuple
from pathlib import Path


def load_single_layer_results(filepath: str = None) -> Dict[int, Dict]:
    """
    加载单层替换结果
    
    Args:
        filepath: 结果文件路径，如果为None则自动查找
    
    Returns:
        字典，key为层ID，value为结果信息
    """
    if filepath is None:
        # 自动查找文件
        try:
            from path_config import get_single_layer_results_file
            filepath = get_single_layer_results_file()
        except:
            # 回退到相对路径
            filepath = Path(__file__).parent.parent / 'scripts' / 'single_layer_results.json'
            if not filepath.exists():
                filepath = "single_layer_results.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 转换key为int
    results = {}
    for key, value in data['results'].items():
        results[int(key)] = value
    
    return results


def load_known_best_solutions(filepath: str = None) -> List[Tuple[List[int], float]]:
    """
    加载已知的1-2层最优解（仅用于公平的初始化，不包含3-4层解）
    
    Args:
        filepath: 结果文件路径，如果为None则返回硬编码的已知最优解
    
    Returns:
        列表，每个元素是 (层列表, 分数)
    """
    if filepath is None:
        # 仅包含1-2层的已知解（公平起见，不泄露3-4层答案）
        return [
            ([17], 0.5144),              # 1层最优
            ([13, 17], 0.5544),          # 2层最优
            ([14, 17], 0.5404),          # 2层次优
            ([12, 17], 0.5382),          # 2层
            ([10, 17], 0.5323),          # 2层
            ([9, 17], 0.5312),           # 2层
            ([16, 17], 0.5278),          # 2层
            # 不包含3-4层解，让GA自己发现
        ]
    else:
        # 从文件加载
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        solutions = []
        for key, value in data.items():
            if value.get('success', False):
                layers = value['replaced_layers']
                score = value['score']
                solutions.append((layers, score))
        
        # 按分数排序
        solutions.sort(key=lambda x: x[1], reverse=True)
        return solutions


def get_top_layers_by_score(single_layer_results: Dict[int, Dict], 
                            top_k: int = 10) -> List[int]:
    """
    获取单层效果最好的top-k层
    
    Args:
        single_layer_results: 单层结果字典
        top_k: 返回前k个
    
    Returns:
        层ID列表，按分数降序
    """
    sorted_layers = sorted(
        single_layer_results.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )
    
    return [layer_id for layer_id, _ in sorted_layers[:top_k]]


def analyze_layer_distribution(single_layer_results: Dict[int, Dict]):
    """
    分析层的分数分布
    
    Args:
        single_layer_results: 单层结果字典
    """
    print("\n" + "=" * 60)
    print("单层分数分布分析")
    print("=" * 60)
    
    # 按区域分组
    regions = {
        '前部 (0-7)': range(0, 8),
        '中前 (8-15)': range(8, 16),
        '中后 (16-23)': range(16, 24),
        '尾部 (24-31)': range(24, 32),
    }
    
    for region_name, layer_range in regions.items():
        scores = [single_layer_results[i]['score'] for i in layer_range if i in single_layer_results]
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            max_layer = [i for i in layer_range if i in single_layer_results 
                        and single_layer_results[i]['score'] == max_score][0]
            print(f"{region_name:15s}: avg={avg_score:.4f}, max={max_score:.4f} (layer {max_layer})")
    
    print()
    
    # Top 10
    top_layers = get_top_layers_by_score(single_layer_results, top_k=10)
    print("Top 10 单层:")
    for i, layer_id in enumerate(top_layers, 1):
        score = single_layer_results[layer_id]['score']
        print(f"  {i:2d}. Layer {layer_id:2d}: {score:.4f}")
    
    print("=" * 60)

