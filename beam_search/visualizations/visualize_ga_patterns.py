#!/usr/bin/env python3
"""
可视化GA每次迭代挖掘出的模式
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.patches import Rectangle

def parse_ga_patterns(log_file):
    """解析GA日志中的模式信息"""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 分割每次模式挖掘
    pattern_sections = re.split(r'模式挖掘结果\n={70}', content)
    
    all_patterns = []
    
    for i, section in enumerate(pattern_sections[1:], 1):  # 跳过第一个空段
        # 提取1层模式
        one_layer_match = re.search(r'1层模式 \(Top 3\):(.*?)(?=2层模式|3层模式|$)', section, re.DOTALL)
        # 提取2层模式
        two_layer_match = re.search(r'2层模式 \(Top 3\):(.*?)(?=3层模式|$)', section, re.DOTALL)
        # 提取3层模式
        three_layer_match = re.search(r'3层模式 \(Top 3\):(.*?)(?=--|\Z)', section, re.DOTALL)
        
        patterns_1 = []
        patterns_2 = []
        patterns_3 = []
        
        # 解析1层模式
        if one_layer_match:
            one_layer_text = one_layer_match.group(1)
            for match in re.finditer(r'\d+\.\s+\[([^\]]+)\]\s+freq=\s*(\d+),\s+avg=([\d.]+),\s+quality=([\d.]+)', one_layer_text):
                layers = [int(x.strip()) for x in match.group(1).split(',')]
                freq = int(match.group(2))
                avg_score = float(match.group(3))
                quality = float(match.group(4))
                patterns_1.append({
                    'layers': layers,
                    'freq': freq,
                    'avg_score': avg_score,
                    'quality': quality
                })
        
        # 解析2层模式
        if two_layer_match:
            two_layer_text = two_layer_match.group(1)
            for match in re.finditer(r'\d+\.\s+\[([^\]]+)\]\s+freq=\s*(\d+),\s+avg=([\d.]+),\s+quality=([\d.]+)', two_layer_text):
                layers = [int(x.strip()) for x in match.group(1).split(',')]
                freq = int(match.group(2))
                avg_score = float(match.group(3))
                quality = float(match.group(4))
                patterns_2.append({
                    'layers': layers,
                    'freq': freq,
                    'avg_score': avg_score,
                    'quality': quality
                })
        
        # 解析3层模式
        if three_layer_match:
            three_layer_text = three_layer_match.group(1)
            for match in re.finditer(r'\d+\.\s+\[([^\]]+)\]\s+freq=\s*(\d+),\s+avg=([\d.]+),\s+quality=([\d.]+)', three_layer_text):
                layers = [int(x.strip()) for x in match.group(1).split(',')]
                freq = int(match.group(2))
                avg_score = float(match.group(3))
                quality = float(match.group(4))
                patterns_3.append({
                    'layers': layers,
                    'freq': freq,
                    'avg_score': avg_score,
                    'quality': quality
                })
        
        all_patterns.append({
            'iteration': i,
            '1-layer': patterns_1,
            '2-layer': patterns_2,
            '3-layer': patterns_3
        })
    
    return all_patterns

def create_pattern_visualization(all_patterns):
    """创建模式可视化表格"""
    
    fig = plt.figure(figsize=(20, 14))
    
    n_iterations = len(all_patterns)
    
    # 创建表格数据
    print('\n' + '='*100)
    print('📊 GA Pattern Mining Evolution')
    print('='*100)
    
    for i, patterns in enumerate(all_patterns, 1):
        print(f'\n【Iteration {i}】')
        
        if patterns['1-layer']:
            print('  1-Layer Patterns:')
            for j, p in enumerate(patterns['1-layer'][:3], 1):
                print(f'    {j}. {p["layers"]} - freq={p["freq"]}, avg={p["avg_score"]:.4f}, quality={p["quality"]:.3f}')
        
        if patterns['2-layer']:
            print('  2-Layer Patterns:')
            for j, p in enumerate(patterns['2-layer'][:3], 1):
                print(f'    {j}. {p["layers"]} - freq={p["freq"]}, avg={p["avg_score"]:.4f}, quality={p["quality"]:.3f}')
        
        if patterns['3-layer']:
            print('  3-Layer Patterns:')
            for j, p in enumerate(patterns['3-layer'][:3], 1):
                print(f'    {j}. {p["layers"]} - freq={p["freq"]}, avg={p["avg_score"]:.4f}, quality={p["quality"]:.3f}')
    
    # 创建主图：Top-1模式演化
    ax1 = plt.subplot(2, 1, 1)
    
    # 提取Top-1的1层模式
    iterations = []
    top1_1layer = []
    top1_1layer_freq = []
    top1_1layer_quality = []
    
    for patterns in all_patterns:
        if patterns['1-layer']:
            iterations.append(patterns['iteration'])
            p = patterns['1-layer'][0]
            top1_1layer.append(str(p['layers']))
            top1_1layer_freq.append(p['freq'])
            top1_1layer_quality.append(p['quality'])
    
    # 绘制频率和质量的演化
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar([x-0.2 for x in iterations], top1_1layer_freq, width=0.4, 
                    alpha=0.7, color='steelblue', label='Frequency')
    line1 = ax1_twin.plot(iterations, top1_1layer_quality, 'ro-', linewidth=2, 
                          markersize=8, label='Quality', alpha=0.8)
    
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='steelblue')
    ax1_twin.set_ylabel('Quality', fontsize=12, fontweight='bold', color='red')
    ax1.set_title('Top-1 Single-Layer Pattern Evolution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.set_xticks(iterations)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 标注模式
    for i, (x, layer) in enumerate(zip(iterations, top1_1layer)):
        ax1.text(x, top1_1layer_freq[i] + 0.5, layer, ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='darkblue')
    
    # 创建子图2：Top-1的2层模式演化
    ax2 = plt.subplot(2, 1, 2)
    
    iterations_2 = []
    top1_2layer = []
    top1_2layer_freq = []
    top1_2layer_quality = []
    
    for patterns in all_patterns:
        if patterns['2-layer']:
            iterations_2.append(patterns['iteration'])
            p = patterns['2-layer'][0]
            top1_2layer.append(str(p['layers']))
            top1_2layer_freq.append(p['freq'])
            top1_2layer_quality.append(p['quality'])
    
    ax2_twin = ax2.twinx()
    
    bars2 = ax2.bar([x-0.2 for x in iterations_2], top1_2layer_freq, width=0.4, 
                    alpha=0.7, color='forestgreen', label='Frequency')
    line2 = ax2_twin.plot(iterations_2, top1_2layer_quality, 'mo-', linewidth=2, 
                          markersize=8, label='Quality', alpha=0.8)
    
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='forestgreen')
    ax2_twin.set_ylabel('Quality', fontsize=12, fontweight='bold', color='magenta')
    ax2.set_title('Top-1 Two-Layer Pattern Evolution', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='forestgreen')
    ax2_twin.tick_params(axis='y', labelcolor='magenta')
    ax2.set_xticks(iterations_2)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 标注模式
    for i, (x, layer) in enumerate(zip(iterations_2, top1_2layer)):
        ax2.text(x, top1_2layer_freq[i] + 0.3, layer, ha='center', va='bottom', 
                fontsize=9, fontweight='bold', color='darkgreen')
    
    plt.suptitle('GA Pattern Mining Evolution Across Iterations', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def create_pattern_table(all_patterns):
    """创建详细的模式表格"""
    
    fig, axes = plt.subplots(len(all_patterns), 1, figsize=(18, 3*len(all_patterns)))
    
    if len(all_patterns) == 1:
        axes = [axes]
    
    for idx, (ax, patterns) in enumerate(zip(axes, all_patterns)):
        ax.axis('off')
        
        # 准备表格数据
        table_data = [[f'Iteration {patterns["iteration"]}', 'Pattern', 'Freq', 'Avg Score', 'Quality']]
        
        # 添加1层模式
        for i, p in enumerate(patterns['1-layer'][:3], 1):
            row_label = f'1-Layer #{i}' if i == 1 else ''
            table_data.append([
                row_label,
                str(p['layers']),
                str(p['freq']),
                f"{p['avg_score']:.4f}",
                f"{p['quality']:.3f}"
            ])
        
        # 添加分隔行
        if patterns['1-layer']:
            table_data.append(['', '', '', '', ''])
        
        # 添加2层模式
        for i, p in enumerate(patterns['2-layer'][:3], 1):
            row_label = f'2-Layer #{i}' if i == 1 else ''
            table_data.append([
                row_label,
                str(p['layers']),
                str(p['freq']),
                f"{p['avg_score']:.4f}",
                f"{p['quality']:.3f}"
            ])
        
        # 添加分隔行
        if patterns['2-layer']:
            table_data.append(['', '', '', '', ''])
        
        # 添加3层模式
        for i, p in enumerate(patterns['3-layer'][:3], 1):
            row_label = f'3-Layer #{i}' if i == 1 else ''
            table_data.append([
                row_label,
                str(p['layers']),
                str(p['freq']),
                f"{p['avg_score']:.4f}",
                f"{p['quality']:.3f}"
            ])
        
        # 创建表格
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.18, 0.30, 0.12, 0.20, 0.20])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # 样式化表头
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 高亮Top-1行
        for i in [1, 5, 9]:  # 每个类别的第一行
            if i < len(table_data):
                for j in range(5):
                    table[(i, j)].set_facecolor('#FFE082')
        
        ax.set_title(f'Iteration {patterns["iteration"]} - Pattern Mining Results', 
                    fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('GA Pattern Mining Details', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig

def main():
    log_file = '../GandA/genetic_layer_search/results/real_test/search_log_20251014_101452.txt'
    
    print('🔍 Parsing GA pattern mining results...')
    all_patterns = parse_ga_patterns(log_file)
    print(f'  Found {len(all_patterns)} iterations with pattern mining')
    
    # 生成演化图
    print('\n📊 Generating pattern evolution plot...')
    fig1 = create_pattern_visualization(all_patterns)
    fig1.savefig('ga_pattern_evolution.png', dpi=300, bbox_inches='tight')
    print('✅ Saved: ga_pattern_evolution.png')
    
    # 生成详细表格
    print('\n📋 Generating detailed pattern table...')
    fig2 = create_pattern_table(all_patterns)
    fig2.savefig('ga_pattern_details.png', dpi=300, bbox_inches='tight')
    print('✅ Saved: ga_pattern_details.png')
    
    print('\n' + '='*100)
    print('✅ All visualizations generated successfully!')
    print('='*100)

if __name__ == '__main__':
    main()


