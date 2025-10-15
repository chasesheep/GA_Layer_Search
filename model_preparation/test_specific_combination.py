#!/usr/bin/env python3
"""
测试特定层组合的MMLU性能
专门用于验证指定的层替换组合
"""

import os
import sys
import json
import torch
import time
from datetime import datetime
from pathlib import Path
import warnings
import logging
import gc
import io
from contextlib import redirect_stderr

# 警告和日志抑制
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("datasets").setLevel(logging.CRITICAL)
logging.getLogger("lm_eval").setLevel(logging.CRITICAL)
logging.getLogger("modelscope").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("torch.cuda").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")
warnings.filterwarnings("ignore", message=".*huggingface-hub.*")
warnings.filterwarnings("ignore", message=".*pretrained.*")
warnings.filterwarnings("ignore", message=".*already-initialized.*")
warnings.filterwarnings("ignore", message=".*loading script.*")
warnings.filterwarnings("ignore", message=".*Parquet.*")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return {
            'allocated_gb': torch.cuda.memory_allocated(device) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(device) / 1e9,
            'free_gb': (torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)) / 1e9
        }
    return None

def force_memory_cleanup():
    """强制清理GPU内存"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def print_memory_status(stage=""):
    """打印内存状态"""
    mem_info = get_memory_info()
    if mem_info:
        print(f"  📊 Memory {stage}: {mem_info['allocated_gb']:.2f}GB allocated, {mem_info['free_gb']:.2f}GB free")

def load_llamba_model():
    """加载Llamba模型"""
    print("\n🧪 Loading Llamba model...")
    
    try:
        from modelscope_utils import get_model_modelscope
        
        print("  - Loading Llamba-8B-unaligned model...")
        llamba_model, llamba_tokenizer, _, _ = get_model_modelscope('unaligned_llamba', is_minimal=False)
        print("    ✅ Llamba model loaded successfully")
        
        mem_info = get_memory_info()
        if mem_info:
            print(f"    📊 Memory after loading: {mem_info['allocated_gb']:.2f}GB allocated")
        
        return llamba_model, llamba_tokenizer
        
    except Exception as e:
        print(f"❌ Llamba model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_extracted_layer(layers_dir, layer_idx, device='cuda'):
    """加载预提取的层文件"""
    layers_path = Path(layers_dir)
    layer_path = layers_path / f"layer_{layer_idx:02d}.pt"
    
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer {layer_idx} not found at {layer_path}")
    
    layer = torch.load(layer_path, map_location=device)
    return layer

def load_extracted_rotary_emb(layers_dir, device='cuda'):
    """加载预提取的rotary_emb"""
    layers_path = Path(layers_dir)
    rotary_emb_path = layers_path / "rotary_emb.pt"
    
    if not rotary_emb_path.exists():
        raise FileNotFoundError(f"Rotary embeddings not found at {rotary_emb_path}")
    
    rotary_emb = torch.load(rotary_emb_path, map_location=device)
    return rotary_emb

def eval_mmlu_with_replacement(model, tokenizer, replaced_layers, llama_layers_dir, 
                              limit=None, batch_size=16, use_cache=True):
    """
    使用指定层替换进行MMLU评估
    """
    limit_str = "full" if limit is None else str(limit)
    print(f"\n  ⏳ MMLU evaluation with layers {replaced_layers} (limit={limit_str})...")
    
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
        gc.collect()
        
        # 监控内存状态
        print_memory_status("(after layer replacement)")
        
        start_time = time.time()
        
        # 运行评估，重定向stderr以抑制warning
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
            # 恢复原始stderr
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
            print(f"  ✅ MMLU completed in {eval_time:.2f}s: {mmlu_score:.3f}")
            
            return {
                'replaced_layers': replaced_layers,
                'score': mmlu_score,
                'time': eval_time,
                'success': True
            }
        else:
            print(f"  ❌ No MMLU results found")
            return {
                'replaced_layers': replaced_layers,
                'score': 0.0,
                'time': 0.0,
                'success': False,
                'error': "No MMLU results found"
            }
        
    except Exception as e:
        print(f"  ❌ MMLU evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'replaced_layers': replaced_layers,
            'score': 0.0,
            'time': 0.0,
            'success': False,
            'error': str(e)
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test specific layer combination')
    parser.add_argument('--layers', type=int, nargs='+', required=True,
                       help='Layers to replace (e.g., --layers 10 14 17 30)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit samples per task (default: None for full MMLU)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # 提取层路径
    llama_layers_dir = "/home/huzhuangfei/Code/GandA/Gather-and-Aggregate/extracted_llama_layers"
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"Layers to replace: {args.layers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Limit: {args.limit if args.limit else 'None (full MMLU)'}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Llama layers path: {llama_layers_dir}")
    print(f"{'='*80}\n")
    
    # 加载Llamba模型
    model, tokenizer = load_llamba_model()
    if model is None or tokenizer is None:
        print("\n❌ Failed to load Llamba model")
        sys.exit(1)
    
    print_memory_status("after model loading")
    
    # 运行评估
    print(f"\n{'='*80}")
    print(f"TESTING COMBINATION: {sorted(args.layers)}")
    print(f"{'='*80}")
    
    result = eval_mmlu_with_replacement(
        model=model,
        tokenizer=tokenizer,
        replaced_layers=sorted(args.layers),
        llama_layers_dir=llama_layers_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        use_cache=True
    )
    
    if result['success']:
        # 保存结果
        output_file = f"combination_test_{'_'.join(map(str, sorted(args.layers)))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result['batch_size'] = args.batch_size
        result['limit'] = args.limit
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULT")
        print(f"{'='*80}")
        print(f"Layers: {result['replaced_layers']}")
        print(f"MMLU Accuracy: {result['score']:.4f}")
        print(f"Evaluation Time: {result['time']:.2f}s")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
    else:
        print(f"\n❌ Evaluation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # 清理
    del model
    del tokenizer
    force_memory_cleanup()

if __name__ == "__main__":
    main()
