#!/usr/bin/env python3
"""
从beam search log实时解析进度
"""

import re
import sys

def parse_beam_log(log_file):
    """解析beam search log"""
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ 日志文件未找到: {log_file}")
        return None
    
    # 提取评估记录
    pattern = r'\[\s*(\d+)/(\d+)\] Testing (\[[\d, ]+\])[\s\S]*?✅ MMLU completed in ([\d.]+)s: ([\d.]+)'
    
    matches = re.findall(pattern, content)
    
    history = []
    best_so_far = 0.0
    
    for idx_str, total_str, layers_str, time_str, score_str in matches:
        score = float(score_str)
        
        if score > best_so_far:
            best_so_far = score
        
        history.append({
            'index': int(idx_str),
            'total': int(total_str),
            'layers': eval(layers_str),
            'score': score,
            'time': float(time_str),
            'best_so_far': best_so_far
        })
    
    return history

def print_progress(history, log_name):
    """打印进度摘要"""
    
    if not history:
        print(f"❌ {log_name}: 无数据")
        return
    
    last = history[-1]
    total_evals = len(history)
    best_score = max(h['best_so_far'] for h in history)
    
    # 判断当前阶段
    if last['total'] == 32:
        stage = "Depth 1 (单层)"
    elif last['total'] > 100:
        stage = f"Depth {len(last['layers'])}"
    else:
        stage = "未知"
    
    print(f"\n{'='*70}")
    print(f"📊 {log_name}")
    print(f"{'='*70}")
    print(f"  总评估: {total_evals} 次")
    print(f"  当前阶段: {stage}")
    print(f"  最新: [{last['index']}/{last['total']}] {last['layers']} → {last['score']:.4f}")
    print(f"  最优分数: {best_score:.4f}")
    print(f"  最优组合: {[h['layers'] for h in history if h['best_so_far'] == best_score][0]}")
    
    # 显示top 5单层（如果在Depth 1）
    if last['total'] == 32 and len(history) >= 5:
        print(f"\n  🏆 Top 5单层:")
        sorted_history = sorted(history, key=lambda x: x['score'], reverse=True)
        for i, h in enumerate(sorted_history[:5]):
            print(f"    {i+1}. Layer {h['layers'][0]}: {h['score']:.4f}")

def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="解析beam search进度")
    parser.add_argument("--log", type=str, nargs='+',
                       help="指定log文件（支持多个）")
    
    args = parser.parse_args()
    
    print("🔍 Beam Search实验进度解析")
    print("="*70)
    
    # 如果指定了log文件
    if args.log:
        for log_file in args.log:
            history = parse_beam_log(log_file)
            if history:
                print_progress(history, log_file)
    else:
        # 自动查找所有log文件
        log_files = list(Path(".").glob("beam_*_log_*.txt"))
        
        if not log_files:
            print("❌ 未找到log文件")
            return 1
        
        for log_file in sorted(log_files):
            history = parse_beam_log(str(log_file))
            if history:
                name = str(log_file.stem).replace('_log_', ' ').replace('_', ' ')
                print_progress(history, name)
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()


