#!/usr/bin/env python3
"""
测试替换后的模型checkpoint

用法:
    # 快速测试（只推理几个样本）
    python test_checkpoint.py --checkpoint ./checkpoints/replaced_11_13_17_21

    # 完整MMLU评估
    python test_checkpoint.py --checkpoint ./checkpoints/replaced_11_13_17_21 --full_eval --limit 100
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def load_checkpoint(checkpoint_dir):
    """加载checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    # 加载元数据
    metadata_path = checkpoint_path / "checkpoint_info.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n📋 Checkpoint Information:")
        print(f"   Base Model: {metadata.get('base_model', 'Unknown')}")
        print(f"   Replaced Layers: {metadata.get('replaced_layers', [])}")
        print(f"   Number of Layers: {metadata.get('num_replaced_layers', 0)}")
        print(f"   Creation Time: {metadata.get('creation_time', 'Unknown')}")
        if 'mmlu_score' in metadata:
            print(f"   MMLU Score: {metadata['mmlu_score']:.4f}")
    
    # 加载模型
    model_path = checkpoint_path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\n📥 Loading model from: {model_path}")
    model = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   ✅ Model loaded ({model_path.stat().st_size / 1e9:.2f} GB)")
    
    # 加载tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    print(f"\n📥 Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"   ✅ Tokenizer loaded")
    
    return model, tokenizer, metadata if metadata_path.exists() else {}

def test_inference(model, tokenizer):
    """测试基本推理功能"""
    print("\n🧪 Testing basic inference...")
    
    device = next(model.parameters()).device
    model.eval()
    
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Machine learning is"
    ]
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {i}: '{prompt}'")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"    Output: {generated_text}")
                print(f"    ✅ Inference successful")
                
            except Exception as e:
                print(f"    ❌ Inference failed: {e}")
                return False
    
    print(f"\n  ✅ All inference tests passed!")
    return True

def run_mmlu_evaluation(model, tokenizer, limit=None):
    """运行MMLU评估"""
    print(f"\n📊 Running MMLU evaluation (limit={limit})...")
    
    try:
        # 导入评估工具
        sys.path.insert(0, str(Path(__file__).parent))
        from modelscope_utils import run_eval
        
        device = next(model.parameters()).device
        model.eval()
        
        # 运行评估
        results = run_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=['mmlu'],
            num_fewshot=5,
            batch_size=16,
            limit=limit,
            cache_requests=True
        )
        
        if results and 'results' in results:
            mmlu_results = results['results'].get('mmlu', {})
            acc = mmlu_results.get('acc,none', 0.0)
            acc_norm = mmlu_results.get('acc_norm,none', 0.0)
            
            print(f"\n  📈 MMLU Results:")
            print(f"     Accuracy: {acc:.4f}")
            print(f"     Normalized Accuracy: {acc_norm:.4f}")
            
            if limit:
                print(f"     (Evaluated on {limit} samples per task)")
            else:
                print(f"     (Full MMLU evaluation)")
            
            return acc
        else:
            print(f"  ❌ MMLU evaluation failed: No results returned")
            return None
            
    except Exception as e:
        print(f"  ❌ MMLU evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Test a replaced model checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--full_eval', action='store_true',
                       help='Run full MMLU evaluation')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples for MMLU (default: None for full eval)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID (default: 0)')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("="*80)
    print("🧪 Test Replaced Model Checkpoint")
    print("="*80)
    
    try:
        # 加载checkpoint
        model, tokenizer, metadata = load_checkpoint(args.checkpoint)
        
        # 测试基本推理
        inference_ok = test_inference(model, tokenizer)
        
        if not inference_ok:
            print("\n❌ Basic inference test failed!")
            return 1
        
        # 如果需要，运行MMLU评估
        if args.full_eval:
            acc = run_mmlu_evaluation(model, tokenizer, limit=args.limit)
            
            if acc is not None:
                # 与元数据中的分数比较
                if 'mmlu_score' in metadata:
                    expected_score = metadata['mmlu_score']
                    diff = abs(acc - expected_score)
                    print(f"\n  📊 Score Comparison:")
                    print(f"     Expected (from metadata): {expected_score:.4f}")
                    print(f"     Actual (just measured): {acc:.4f}")
                    print(f"     Difference: {diff:.4f}")
                    
                    if diff < 0.01:
                        print(f"     ✅ Scores match!")
                    else:
                        print(f"     ⚠️  Scores differ significantly")
        
        print("\n" + "="*80)
        print("✅ Checkpoint test completed successfully!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

