#!/usr/bin/env python3
"""
Vanilla Beam Search - 标准Beam Search实现
- 无先验知识
- 所有32层等价对待
- 用于与GA方法的公平对比
"""

import os
import sys
import torch
import time
import json
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Mute all warnings and logging (参考GandA实现)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set all loggers to CRITICAL level
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("datasets").setLevel(logging.CRITICAL)
logging.getLogger("lm_eval").setLevel(logging.CRITICAL)
logging.getLogger("modelscope").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)

# Mute specific warning categories
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*huggingface-hub.*")
warnings.filterwarnings("ignore", message=".*pretrained.*")
warnings.filterwarnings("ignore", message=".*loading script.*")
warnings.filterwarnings("ignore", message=".*Parquet.*")

# Set environment variables
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

# 强制使用本地缓存（避免网络下载）
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add parent directory to path for imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
# Use GandA's modelscope_utils which has downloaded models
sys.path.insert(0, str(parent_dir / "GandA" / "Gather-and-Aggregate"))
# Add GA_Layer_Search for cartesia_pytorch
sys.path.insert(0, str(parent_dir / "GA_Layer_Search"))

def get_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        }
    return None

def force_memory_cleanup():
    """强制清理GPU内存"""
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    torch.cuda.synchronize()

def print_memory_status(stage=""):
    """打印内存状态"""
    mem_info = get_memory_info()
    if mem_info:
        print(f"    📊 Memory {stage}: {mem_info['allocated_gb']:.2f}GB allocated, {mem_info['free_gb']:.2f}GB free")

def load_llamba_model():
    """加载Llamba模型"""
    print("\n🧪 Loading Llamba model...")
    
    try:
        from modelscope_utils import get_model_modelscope
        
        print("  - Loading Llamba-8B-unaligned model...")
        llamba_model, llamba_tokenizer, _, _ = get_model_modelscope(
            'unaligned_llamba', is_minimal=False
        )
        print("    ✅ Llamba model loaded successfully")
        
        mem_info = get_memory_info()
        if mem_info:
            print(f"    📊 Memory: {mem_info['allocated_gb']:.2f}GB allocated")
        
        return llamba_model, llamba_tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_extracted_layer(layers_dir, layer_idx, device='cuda'):
    """加载预提取的层文件"""
    layers_path = Path(layers_dir)
    layer_path = layers_path / f"layer_{layer_idx:02d}.pt"
    
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer {layer_idx} not found at {layer_path}")
    
    return torch.load(layer_path, map_location=device)

def load_extracted_rotary_emb(layers_dir, device='cuda'):
    """加载预提取的rotary_emb"""
    layers_path = Path(layers_dir)
    rotary_emb_path = layers_path / "rotary_emb.pt"
    
    if not rotary_emb_path.exists():
        raise FileNotFoundError(f"Rotary embeddings not found at {rotary_emb_path}")
    
    return torch.load(rotary_emb_path, map_location=device)

def eval_mmlu_with_replacement(model, tokenizer, replaced_layers, llama_layers_dir, 
                              limit=None, batch_size=16, use_cache=True):
    """
    使用指定层替换进行MMLU评估（参考GandA的实现）
    """
    limit_str = "full" if limit is None else str(limit)
    print(f"    ⏳ MMLU evaluation with layers {replaced_layers} (limit={limit_str})...")
    
    try:
        from modelscope_utils import run_eval
        
        # 保存原始状态
        original_layers = {}
        original_rotary_emb = None
        if hasattr(model.backbone, 'rotary_emb'):
            original_rotary_emb = model.backbone.rotary_emb
        
        # 加载并替换指定层
        llama_rotary_emb = load_extracted_rotary_emb(llama_layers_dir, device=model.device)
        model.backbone.rotary_emb = llama_rotary_emb
        
        # 存储加载的Llama层，用于后续清理
        loaded_llama_layers = []
        for layer_idx in replaced_layers:
            original_layers[layer_idx] = model.backbone.layers[layer_idx]
            llama_layer = load_extracted_layer(llama_layers_dir, layer_idx, device=model.device)
            model.backbone.layers[layer_idx] = llama_layer
            loaded_llama_layers.append(llama_layer)
        
        # 立即清理加载的Llama层，避免累积显存占用
        del llama_rotary_emb
        for llama_layer in loaded_llama_layers:
            del llama_layer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        start_time = time.time()
        
        # 运行评估（重定向stderr抑制警告）
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            results = run_eval(
                model=model, 
                tokenizer=tokenizer, 
                tasks=["mmlu"],
                limit=limit,
                batch_size=batch_size,
                cache_requests=use_cache
            )
        finally:
            sys.stderr = old_stderr
        
        # 评估完成后立即清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        eval_time = time.time() - start_time
        
        # 恢复原始状态
        for layer_idx in replaced_layers:
            model.backbone.layers[layer_idx] = original_layers[layer_idx]
        if original_rotary_emb is not None:
            model.backbone.rotary_emb = original_rotary_emb
        
        if 'results' in results and 'mmlu' in results['results']:
            mmlu_score = results['results']['mmlu']['acc,none']
            print(f"    ✅ MMLU completed in {eval_time:.2f}s: {mmlu_score:.3f}")
            
            return {
                'replaced_layers': replaced_layers,
                'score': mmlu_score,
                'time': eval_time,
                'success': True
            }
        else:
            print(f"    ❌ No MMLU results found")
            return {
                'replaced_layers': replaced_layers,
                'score': 0.0,
                'time': 0.0,
                'success': False,
                'error': "No MMLU results found"
            }
        
    except Exception as e:
        print(f"    ❌ MMLU evaluation failed: {e}")
        return {
            'replaced_layers': replaced_layers,
            'score': 0.0,
            'time': 0.0,
            'success': False,
            'error': str(e)
        }

def vanilla_beam_search(model, tokenizer, llama_layers_dir, 
                       limit=100, beam_width=5, 
                       min_layers=2, max_layers=4):
    """
    标准Beam Search - 无先验知识，所有层等价
    
    Args:
        model: Llamba模型
        tokenizer: tokenizer
        llama_layers_dir: Llama层目录
        limit: MMLU评估limit
        beam_width: beam宽度（对比GA的population_size）
        min_layers: 最少替换层数
        max_layers: 最多替换层数
    
    Returns:
        all_results: 所有评估结果
        best_result: 最佳结果
        evaluation_count: 评估次数
        search_history: 搜索历史（用于可视化）
    """
    print(f"\n🔍 Vanilla Beam Search (No Prior)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  - MMLU limit: {limit}")
    print(f"  - Beam width: {beam_width}")
    print(f"  - Layer range: {min_layers}-{max_layers} layers")
    print(f"  - Candidate layers: ALL 32 layers (0-31)")
    print(f"  - No prior knowledge: All layers treated equally")
    
    # 所有32层都是候选
    all_layers = list(range(32))
    
    # 初始化
    current_beam = [[]]  # 从空开始
    all_results = {}  # 存储所有结果
    evaluation_count = 0
    search_history = []  # 记录搜索历史（用于可视化）
    best_score_so_far = 0.0  # 记录迄今为止的最优分数
    
    # 逐层搜索
    for depth in range(1, max_layers + 1):
        print(f"\n{'='*70}")
        print(f"🔧 Depth {depth}: Testing {depth}-layer combinations")
        print(f"{'='*70}")
        
        # 生成候选
        candidates = []
        for base_layers in current_beam:
            for layer in all_layers:
                if layer not in base_layers:
                    new_combo = sorted(base_layers + [layer])
                    if new_combo not in candidates:
                        candidates.append(new_combo)
        
        print(f"📋 Generated {len(candidates)} candidates from {len(current_beam)} base combinations")
        
        # 评估所有候选
        depth_results = []
        for i, combo in enumerate(candidates):
            print(f"\n  [{i+1:3d}/{len(candidates)}] Testing {combo}")
            
            result = eval_mmlu_with_replacement(
                model, tokenizer, combo, llama_layers_dir,
                limit=limit, batch_size=16, use_cache=True
            )
            
            evaluation_count += 1
            depth_results.append(result)
            all_results[tuple(combo)] = result
            
            # 记录搜索历史
            if result['success'] and result['score'] > best_score_so_far:
                best_score_so_far = result['score']
            
            search_history.append({
                'evaluation': evaluation_count,
                'combination': combo,
                'score': result['score'] if result['success'] else None,
                'best_so_far': best_score_so_far,
                'depth': depth
            })
            
            force_memory_cleanup()
        
        # 选择top beam_width个结果
        successful_results = [r for r in depth_results if r['success']]
        
        if not successful_results:
            print(f"\n❌ No successful results at depth {depth}. Stopping.")
            break
        
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        current_beam = [r['replaced_layers'] for r in successful_results[:beam_width]]
        
        print(f"\n📈 Top {min(beam_width, len(successful_results))} at depth {depth}:")
        for i, result in enumerate(successful_results[:beam_width]):
            print(f"  {i+1}. {result['replaced_layers']}: {result['score']:.4f}")
        
        # 保存中间checkpoint（每个depth完成后）
        checkpoint_file = f"checkpoint_depth{depth}.json"
        save_checkpoint(all_results, evaluation_count, search_history, depth, checkpoint_file)
    
    # 找出最佳结果（满足层数约束）
    valid_results = [
        r for r in all_results.values() 
        if r['success'] and min_layers <= len(r['replaced_layers']) <= max_layers
    ]
    
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['score'])
    else:
        best_result = None
    
    print(f"\n{'='*70}")
    print(f"📊 Search Summary")
    print(f"{'='*70}")
    print(f"  - Total evaluations: {evaluation_count}")
    print(f"  - Successful: {sum(1 for r in all_results.values() if r['success'])}")
    print(f"  - Failed: {sum(1 for r in all_results.values() if not r['success'])}")
    
    if best_result:
        print(f"\n🏆 Best Result:")
        print(f"  - Layers: {best_result['replaced_layers']}")
        print(f"  - MMLU Score: {best_result['score']:.4f}")
        print(f"  - Evaluation time: {best_result['time']:.1f}s")
    
    return all_results, best_result, evaluation_count, search_history

def save_checkpoint(all_results, evaluation_count, search_history, depth, filename):
    """保存中间checkpoint"""
    # 找当前最优
    valid_results = [r for r in all_results.values() if r['success']]
    current_best = max(valid_results, key=lambda x: x['score']) if valid_results else None
    
    checkpoint = {
        'depth': depth,
        'evaluations': evaluation_count,
        'best_score': current_best['score'] if current_best else 0.0,
        'best_layers': current_best['replaced_layers'] if current_best else [],
        'search_history': search_history,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        print(f"    💾 Checkpoint saved: {filename}")
    except Exception as e:
        print(f"    ⚠️ Checkpoint save failed: {e}")

def save_results(all_results, best_result, evaluation_count, search_history, args, filename=None):
    """保存结果"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vanilla_beam_search_results_{timestamp}.json"
    
    # 转换为可序列化格式
    results_serializable = {}
    for key, value in all_results.items():
        key_str = str(list(key) if isinstance(key, tuple) else key)
        results_serializable[key_str] = value
    
    output = {
        'method': 'vanilla_beam_search',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'limit': args.limit,
            'beam_width': args.beam_width,
            'min_layers': args.min_layers,
            'max_layers': args.max_layers,
        },
        'statistics': {
            'total_evaluations': evaluation_count,
            'successful': sum(1 for r in all_results.values() if r['success']),
            'failed': sum(1 for r in all_results.values() if not r['success']),
        },
        'best_result': best_result,
        'all_results': results_serializable,
        'search_history': search_history  # 用于可视化
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Vanilla Beam Search - No Prior Knowledge"
    )
    parser.add_argument("--llama_layers_dir", type=str, 
                       default="../GA_Layer_Search/extracted_llama_layers",
                       help="Directory containing extracted Llama layers")
    parser.add_argument("--limit", type=int, default=100,
                       help="MMLU evaluation limit")
    parser.add_argument("--beam_width", type=int, default=5,
                       help="Beam width (number of candidates to keep)")
    parser.add_argument("--min_layers", type=int, default=2,
                       help="Minimum number of layers to replace")
    parser.add_argument("--max_layers", type=int, default=4,
                       help="Maximum number of layers to replace")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON filename")
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"🎮 Using GPU {args.gpu_id}")
    
    print("🚀 Vanilla Beam Search - Fair Comparison Baseline")
    print("=" * 70)
    print("No prior knowledge - All 32 layers treated equally")
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return 1
    
    device = torch.cuda.current_device()
    print(f"✅ GPU: {torch.cuda.get_device_name(device)}")
    print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # 检查层文件
    llama_layers_path = Path(args.llama_layers_dir)
    if not llama_layers_path.exists():
        print(f"❌ Llama layers not found: {args.llama_layers_dir}")
        return 1
    
    # 清理缓存
    torch.cuda.empty_cache()
    
    # 加载模型
    model, tokenizer = load_llamba_model()
    if model is None:
        print("❌ Model loading failed")
        return 1
    
    # 运行Beam Search
    start_time = time.time()
    all_results, best_result, evaluation_count, search_history = vanilla_beam_search(
        model, tokenizer, args.llama_layers_dir,
        limit=args.limit,
        beam_width=args.beam_width,
        min_layers=args.min_layers,
        max_layers=args.max_layers
    )
    total_time = time.time() - start_time
    
    # 保存结果
    save_results(all_results, best_result, evaluation_count, search_history, args, args.output)
    
    # 最终总结
    print(f"\n{'='*70}")
    print(f"🎉 VANILLA BEAM SEARCH COMPLETED")
    print(f"{'='*70}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Evaluations: {evaluation_count}")
    
    if best_result:
        print(f"🏆 Best: {best_result['replaced_layers']} → {best_result['score']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

