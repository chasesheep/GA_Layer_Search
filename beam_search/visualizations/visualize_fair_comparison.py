#!/usr/bin/env python3
"""
最公平对比：Beam Quick vs GA
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import re
from pathlib import Path

def parse_beam_log(log_file):
    """解析beam search log"""
    history = []
    evaluation_count = 0
    best_so_far = 0.0
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if "✅ MMLU completed" in line:
            score_str = line.split(": ")[-1].strip()
            score = float(score_str)
            evaluation_count += 1
            
            if score > best_so_far:
                best_so_far = score
            
            history.append({
                'evaluation': evaluation_count,
                'score': score,
                'best_so_far': best_so_far
            })
    
    return history

def parse_ga_history():
    """解析GA历史，带阶段标注"""
    log_file = "../GandA/genetic_layer_search/results/real_test/search_log_20251014_101452.txt"
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    pattern = r'MMLU evaluation with layers (\[[\d, ]+\]) \(limit=(\d+)\)\.\.\.[\s\S]*?✅ MMLU completed in [\d.]+s: ([\d.]+)'
    matches = re.findall(pattern, content)
    
    history = []
    best_so_far = 0.0
    
    for i, (layers_str, limit, score_str) in enumerate(matches):
        score = float(score_str)
        # 公平起见，GA评估次数+32（假设前32次用于单层评估，虽未在log中显示）
        eval_num = i + 1 + 32
        
        if score > best_so_far:
            best_so_far = score
        
        # 判断阶段 (加上前置的32次单层评估)
        # 阶段0: 单层评估 (1-32, 未在此log中)
        # 阶段1: GA粗搜索 (33-307)
        # 阶段2: Top-17完整评估 (308-324)
        # 阶段3: 局部精细优化 (325-835)
        if eval_num <= 32:
            phase = 'single_layer'
        elif eval_num <= 307:  # 275 + 32
            phase = 'ga_coarse'
        elif eval_num <= 324:  # 292 + 32
            phase = 'top17_eval'
        else:
            phase = 'local_search'
        
        history.append({
            'evaluation': eval_num,
            'score': score,
            'best_so_far': best_so_far,
            'phase': phase
        })
    
    phase_boundaries = {
        32: 'GA Coarse\nSearch',
        307: 'Top-17\nEval',
        324: 'Local\nSearch\n(non-GA)'
    }
    
    return history, phase_boundaries

def create_fair_comparison_plot(beam_quick, ga_history, ga_phase_boundaries):
    """创建公平对比图"""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35, 
                         height_ratios=[1.3, 1, 1.1],
                         top=0.94, bottom=0.05)
    
    beam_evals = [h['evaluation'] for h in beam_quick]
    beam_scores = [h['score'] for h in beam_quick]
    beam_best = [h['best_so_far'] for h in beam_quick]
    
    ga_evals = [h['evaluation'] for h in ga_history]
    ga_scores = [h['score'] for h in ga_history]
    ga_best = [h['best_so_far'] for h in ga_history]
    
    # 图1: 主收敛曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(beam_evals, beam_best, 'b-', linewidth=3, label='Beam Search (width=5)', alpha=0.8)
    ax1.plot(ga_evals, ga_best, 'r-', linewidth=3, label='GA + Local Search (pop=20)', alpha=0.8)
    
    # 标注GA阶段 - 靠近对应位置但错开避免重叠
    colors_phase = ['green', 'orange', 'purple']
    boundaries_list = list(ga_phase_boundaries.items())
    
    # 阶段标注：靠近虚线但y位置错开
    phase_annotations = [
        (60, 0.47, 'GA Coarse\nSearch', 'green'),       # 接近32虚线
        (275, 0.47, 'Top-17\nEval', 'orange'),         # 接近307虚线  
        (370, 0.47, 'Local Search\n(non-GA)', 'purple') # 接近324虚线
    ]
    
    for i, (boundary, label) in enumerate(boundaries_list):
        ax1.axvline(x=boundary, color=colors_phase[i], linestyle='--', alpha=0.7, linewidth=2)
    
    # 添加阶段标注
    for x_pos, y_pos, text, color in phase_annotations:
        ax1.text(x_pos, y_pos, text, 
                rotation=0, ha='center', va='center', fontsize=8, 
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.75, 
                         edgecolor=color, linewidth=1.5))
    
    # 标注Final结果 - 使用适中的相对偏移，不会超出图表
    ax1.annotate(f'Beam Final\n{beam_best[-1]:.4f}\n({beam_evals[-1]} evals)', 
                xy=(beam_evals[-1], beam_best[-1]),
                xytext=(-80, -40), textcoords='offset points',
                fontsize=9, color='blue', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9, edgecolor='blue', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax1.annotate(f'GA Final\n{ga_best[-1]:.4f}\n({ga_evals[-1]} evals)', 
                xy=(ga_evals[-1], ga_best[-1]),
                xytext=(-50, -40), textcoords='offset points',
                fontsize=9, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9, edgecolor='red', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax1.set_xlabel('Number of Evaluations', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Best Score Found', fontsize=13, fontweight='bold')
    ax1.set_title('Fair Comparison: Beam Search vs GA (Both using MMLU)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 早期阶段 (前100次)
    ax2 = fig.add_subplot(gs[1, 0])
    beam_early = [(e, b) for e, b in zip(beam_evals, beam_best) if e <= 100]
    ga_early = [(e, b) for e, b in zip(ga_evals, ga_best) if e <= 100]
    
    if beam_early:
        e, b = zip(*beam_early)
        ax2.plot(e, b, 'b-', linewidth=2.5, label='Beam Search (width=5)', marker='o', markersize=4, markevery=10)
    if ga_early:
        e, b = zip(*ga_early)
        ax2.plot(e, b, 'r-', linewidth=2.5, label='GA (pop=20)', marker='s', markersize=4, markevery=10)
    
    # 标注GA阶段
    for i, (boundary, label) in enumerate(ga_phase_boundaries.items()):
        if boundary <= 100:
            ax2.axvline(x=boundary, color=colors_phase[i], linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax2.set_xlabel('Number of Evaluations', fontsize=11)
    ax2.set_ylabel('Best Score Found', fontsize=11)
    ax2.set_title('Early Stage Convergence (First 100 Evaluations)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # 图3: 仅GA阶段对比 (1-307次，包含32次单层评估)
    ax3 = fig.add_subplot(gs[1, 1])
    beam_ga_phase = [(e, b) for e, b in zip(beam_evals, beam_best) if e <= 307]
    ga_ga_phase = [(e, b) for e, b in zip(ga_evals, ga_best) if e <= 307]
    
    if beam_ga_phase:
        e, b = zip(*beam_ga_phase)
        ax3.plot(e, b, 'b-', linewidth=2.5, label='Beam Search (width=5)', marker='o', markersize=4, markevery=20)
    if ga_ga_phase:
        e, b = zip(*ga_ga_phase)
        ax3.plot(e, b, 'r-', linewidth=2.5, label='GA (pop=20)', marker='s', markersize=4, markevery=15)
    
    # 标注单层评估结束点
    ax3.axvline(x=32, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
    ax3.text(32, 0.56, 'Single-layer\nphase ends', 
            rotation=0, ha='center', va='center', fontsize=8, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7, edgecolor='green', linewidth=1))
    
    ax3.set_xlabel('Number of Evaluations', fontsize=11)
    ax3.set_ylabel('Best Score Found', fontsize=11)
    ax3.set_title('Pure GA Phase (1-307 evals, before local search)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 320)
    
    # 图4: 散点图
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(beam_evals, beam_scores, alpha=0.3, s=15, c='blue', label='Beam (width=5) evals')
    ax4.scatter(ga_evals, ga_scores, alpha=0.3, s=15, c='red', label='GA (pop=20) evals')
    ax4.plot(beam_evals, beam_best, 'b-', linewidth=2.5, alpha=0.8, label='Beam best')
    ax4.plot(ga_evals, ga_best, 'r-', linewidth=2.5, alpha=0.8, label='GA best')
    
    ax4.set_xlabel('Number of Evaluations', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('All Evaluated Scores', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 图5: 效率对比表格
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # 计算关键指标
    milestones = [0.52, 0.54, 0.56, 0.58, 0.59]
    table_data = [['Milestone', 'Beam Search', 'GA', 'GA Advantage']]
    table_data.append(['', '', '', ''])
    
    # 最终分数
    table_data.append(['Final Score', f'{beam_best[-1]:.4f}', f'{ga_best[-1]:.4f}', 
                      f'{ga_best[-1] - beam_best[-1]:+.4f}'])
    table_data.append(['Total Evals', str(beam_evals[-1]), str(ga_evals[-1]), 
                      f'{ga_evals[-1] / beam_evals[-1]:.1f}x more'])
    table_data.append(['', '', '', ''])
    
    # GA阶段结束时对比 (307次 = 32单层 + 275 GA粗搜索)
    beam_at_307_idx = min(306, len(beam_best) - 1)
    beam_at_307 = beam_best[beam_at_307_idx]
    ga_at_307_idx = next((i for i, h in enumerate(ga_history) if h['evaluation'] >= 307), len(ga_history)-1)
    ga_at_307 = ga_best[ga_at_307_idx] if ga_at_307_idx < len(ga_best) else ga_best[-1]
    table_data.append(['Score @ 307 evals', f'{beam_at_307:.4f}', f'{ga_at_307:.4f}',
                      'GA phase end'])
    
    table_data.append(['', '', '', ''])
    
    # 达到里程碑的评估次数
    for milestone in milestones:
        beam_idx = next((i for i, b in enumerate(beam_best) if b >= milestone), None)
        ga_idx = next((i for i, b in enumerate(ga_best) if b >= milestone), None)
        
        beam_evals_milestone = str(beam_evals[beam_idx]) if beam_idx is not None else 'N/A'
        ga_evals_milestone = str(ga_evals[ga_idx]) if ga_idx is not None else 'N/A'
        
        if beam_idx is not None and ga_idx is not None:
            advantage = f'{beam_evals[beam_idx] / ga_evals[ga_idx]:.1f}x'
        else:
            advantage = 'N/A'
        
        table_data.append([f'Reach {milestone}', beam_evals_milestone, ga_evals_milestone, advantage])
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='upper center',
                     colWidths=[0.28, 0.24, 0.24, 0.24],
                     bbox=[0, -0.05, 1, 0.95])  # [x, y, width, height] 调整表格位置
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # 样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for row in [1, 4, 6]:
        for col in range(4):
            table[(row, col)].set_facecolor('#E0E0E0')
    
    # 高亮最佳值
    table[(2, 2)].set_facecolor('#FFE082') if ga_best[-1] > beam_best[-1] else table[(2, 1)].set_facecolor('#FFE082')
    
    ax5.set_title('Efficiency Comparison Summary', fontsize=12, fontweight='bold', pad=25, y=1.0)
    
    plt.suptitle('Fair Comparison: Beam Search vs GA (MMLU)', 
                 fontsize=16, fontweight='bold', y=0.985)
    
    return fig

def print_detailed_analysis(beam_quick, ga_history):
    """打印详细分析"""
    print("\n" + "="*80)
    print("📊 Fair Comparison Analysis (Both using MMLU)")
    print("="*80)
    
    beam_best = [h['best_so_far'] for h in beam_quick]
    ga_best = [h['best_so_far'] for h in ga_history]
    
    print(f"\n【Final Results】")
    print(f"  Beam Search: {beam_best[-1]:.4f} ({len(beam_quick)} evals)")
    print(f"  GA Total:    {ga_best[-1]:.4f} ({len(ga_history)} evals)")
    print(f"  Difference:  {ga_best[-1] - beam_best[-1]:+.4f}")
    
    print(f"\n【GA Phase Only (1-307 evals, including 32 single-layer)】")
    # 找GA在307次评估时的分数（GA粗搜索阶段结束）
    ga_at_307_idx = next((i for i, h in enumerate(ga_history) if h['evaluation'] >= 307), len(ga_history)-1)
    ga_at_307 = ga_best[ga_at_307_idx] if ga_at_307_idx < len(ga_best) else ga_best[-1]
    
    beam_at_307_idx = min(306, len(beam_best) - 1)
    beam_at_307 = beam_best[beam_at_307_idx]
    print(f"  GA @ 307:   {ga_at_307:.4f}")
    print(f"  Beam @ 307: {beam_at_307:.4f}")
    print(f"  GA Advantage: {ga_at_307 - beam_at_307:+.4f}")
    
    print(f"\n【Efficiency to reach 0.58】")
    beam_58_idx = next((i for i, b in enumerate(beam_best) if b >= 0.58), None)
    ga_58_idx = next((i for i, h in enumerate(ga_history) if h['best_so_far'] >= 0.58), None)
    if beam_58_idx is not None and ga_58_idx is not None:
        beam_58_evals = beam_58_idx + 1
        ga_58_evals = ga_history[ga_58_idx]['evaluation']  # 使用实际的evaluation值（已+32）
        print(f"  Beam: {beam_58_evals} evals")
        print(f"  GA:   {ga_58_evals} evals")
        print(f"  GA is {beam_58_evals / ga_58_evals:.1f}x faster")
    
    print("\n" + "="*80)

def main():
    print("🔍 Parsing experiment data...")
    
    beam_quick = parse_beam_log("beam_quick_log_20251102_201647_clean.txt")
    print(f"  Beam Quick: {len(beam_quick)} evaluations, best={beam_quick[-1]['best_so_far']:.4f}")
    
    ga_history, ga_phase_boundaries = parse_ga_history()
    print(f"  GA:         {len(ga_history)} evaluations, best={ga_history[-1]['best_so_far']:.4f}")
    
    print("\n📊 Generating fair comparison plot...")
    fig = create_fair_comparison_plot(beam_quick, ga_history, ga_phase_boundaries)
    
    output_file = "fair_comparison_beam_vs_ga.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_file}")
    
    # 保存数据
    data = {
        'beam_quick': beam_quick,
        'ga': ga_history,
        'ga_phase_boundaries': {str(k): v for k, v in ga_phase_boundaries.items()}
    }
    with open('fair_comparison_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Data saved: fair_comparison_data.json")
    
    # 打印分析
    print_detailed_analysis(beam_quick, ga_history)

if __name__ == "__main__":
    main()

