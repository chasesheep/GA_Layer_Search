"""
真实MMLU评估适应度函数 (基于adaptive beam search的成功经验)
"""
import os
import sys
import torch
import time
import warnings
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
import io

# 添加Gather-and-Aggregate到路径
GA_DIR = str(Path(__file__).parent.parent / "Gather-and-Aggregate")
if GA_DIR not in sys.path:
    sys.path.insert(0, GA_DIR)

# Mute all warnings and logging (和beam search一样)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"

# Set all loggers to CRITICAL
logging.getLogger().setLevel(logging.CRITICAL)
for logger_name in ["transformers", "datasets", "lm_eval", "modelscope", 
                     "huggingface_hub", "urllib3", "requests", "torch", "torch.cuda"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_extracted_layer(layers_dir, layer_idx, device='cuda'):
    """加载预提取的层文件（和beam search一样）"""
    layers_path = Path(layers_dir)
    layer_path = layers_path / f"layer_{layer_idx:02d}.pt"
    
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer {layer_idx} not found at {layer_path}")
    
    layer = torch.load(layer_path, map_location=device, weights_only=False)
    return layer


def load_extracted_rotary_emb(layers_dir, device='cuda'):
    """加载预提取的rotary_emb（和beam search一样）"""
    layers_path = Path(layers_dir)
    rotary_emb_path = layers_path / "rotary_emb.pt"
    
    if not rotary_emb_path.exists():
        raise FileNotFoundError(f"Rotary embeddings not found at {rotary_emb_path}")
    
    rotary_emb = torch.load(rotary_emb_path, map_location=device, weights_only=False)
    return rotary_emb


def eval_mmlu_with_replacement(model, tokenizer, replaced_layers, llama_layers_dir, 
                              limit=None, batch_size=16, use_cache=True, verbose=False):
    """
    使用指定层替换进行MMLU评估（和beam search完全一样）
    
    Args:
        model: Llamba模型
        tokenizer: tokenizer
        replaced_layers: 要替换的层索引列表
        llama_layers_dir: Llama层目录
        limit: 评估样本限制
        batch_size: 批次大小
        use_cache: 是否使用缓存
        verbose: 是否打印详细信息
    
    Returns:
        评估结果字典
    """
    limit_str = "full" if limit is None else str(limit)
    if verbose:
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
            if verbose:
                print(f"    ✅ MMLU completed in {eval_time:.2f}s: {mmlu_score:.4f}")
            
            return {
                'replaced_layers': replaced_layers,
                'score': mmlu_score,
                'time': eval_time,
                'success': True
            }
        else:
            if verbose:
                print(f"    ❌ No MMLU results found")
            return {
                'replaced_layers': replaced_layers,
                'score': 0.0,
                'time': 0.0,
                'success': False,
                'error': "No MMLU results found"
            }
        
    except Exception as e:
        if verbose:
            print(f"    ❌ MMLU evaluation failed: {e}")
        return {
            'replaced_layers': replaced_layers,
            'score': 0.0,
            'time': 0.0,
            'success': False,
            'error': str(e)
        }


class MMLUFitnessFunction:
    """
    MMLU适应度函数包装器
    
    使用单例模式，确保模型只加载一次
    """
    _instance = None
    _model = None
    _tokenizer = None
    _llama_layers_dir = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, llama_layers_dir: str, gpu_id: int = None, verbose: bool = True):
        """
        初始化模型（只调用一次）
        
        Args:
            llama_layers_dir: Llama层目录
            gpu_id: GPU ID
            verbose: 是否打印信息
        """
        if cls._model is not None:
            if verbose:
                print("  ℹ️  Model already loaded, skipping initialization")
            return
        
        # 设置GPU
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            if verbose:
                print(f"  🎮 Using GPU: {gpu_id}")
        
        cls._llama_layers_dir = llama_layers_dir
        
        if verbose:
            print("\n🧪 Loading Llamba model...")
        
        try:
            from modelscope_utils import get_model_modelscope
            
            cls._model, cls._tokenizer, _, _ = get_model_modelscope(
                'unaligned_llamba', 
                is_minimal=False
            )
            
            if verbose:
                print("  ✅ Llamba model loaded successfully")
                if torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    print(f"  📊 Memory: {mem_gb:.2f}GB allocated")
        
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            raise
    
    @classmethod
    def create_fitness_function(cls, limit: Optional[int] = None, 
                                verbose: bool = False) -> Callable[[List[int]], float]:
        """
        创建适应度函数
        
        Args:
            limit: 评估样本限制
            verbose: 是否打印详细信息
        
        Returns:
            fitness函数: layers -> score
        """
        if cls._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        def fitness_func(layers: List[int]) -> float:
            """评估层组合的适应度"""
            result = eval_mmlu_with_replacement(
                model=cls._model,
                tokenizer=cls._tokenizer,
                replaced_layers=layers,
                llama_layers_dir=cls._llama_layers_dir,
                limit=limit,
                batch_size=16,
                use_cache=True,
                verbose=verbose
            )
            return result['score'] if result['success'] else 0.0
        
        return fitness_func
    
    @classmethod
    def cleanup(cls):
        """清理资源"""
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()


# ==================== 便捷函数 ====================

def create_mmlu_fitness(llama_layers_dir: str,
                       limit: Optional[int] = None,
                       gpu_id: int = None,
                       verbose: bool = True) -> Callable[[List[int]], float]:
    """
    一站式创建MMLU适应度函数
    
    Args:
        llama_layers_dir: Llama层目录
        limit: 评估样本限制
        gpu_id: GPU ID
        verbose: 是否打印详细信息
    
    Returns:
        fitness函数
    """
    # 初始化（如果还没初始化）
    MMLUFitnessFunction.initialize(
        llama_layers_dir=llama_layers_dir,
        gpu_id=gpu_id,
        verbose=verbose
    )
    
    # 创建fitness函数
    return MMLUFitnessFunction.create_fitness_function(
        limit=limit,
        verbose=verbose
    )


# ==================== 测试代码 ====================

def test_simple():
    """简单测试"""
    print("\n" + "="*70)
    print("测试MMLU适应度函数")
    print("="*70)
    
    llama_layers_dir = "/home/huzhuangfei/Code/GandA/Gather-and-Aggregate/extracted_llama_layers"
    
    # 创建fitness函数（limit=10，快速测试）
    print("\n创建fitness函数...")
    fitness_func = create_mmlu_fitness(
        llama_layers_dir=llama_layers_dir,
        limit=10,
        gpu_id=3,  # 使用GPU 3
        verbose=True
    )
    
    # 测试1: 单层
    print(f"\n{'='*70}")
    print("测试1: 评估 [17]")
    print("="*70)
    score1 = fitness_func([17])
    print(f"分数: {score1:.4f}")
    
    # 测试2: 2层
    print(f"\n{'='*70}")
    print("测试2: 评估 [13, 17]")
    print("="*70)
    score2 = fitness_func([13, 17])
    print(f"分数: {score2:.4f}")
    
    # 清理
    print("\n清理资源...")
    MMLUFitnessFunction.cleanup()
    
    print(f"\n{'='*70}")
    print("✅ 测试完成")
    print("="*70)


if __name__ == "__main__":
    test_simple()

