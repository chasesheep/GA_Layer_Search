#!/usr/bin/env python3
"""
提取Llama模型的每一层和rotary_emb，保存为单独文件
避免重复加载整个模型，提高实验效率

使用方法：
python extract_layers.py --model_name llama --output_dir ./extracted_layers
python extract_layers.py --model_name unaligned_llamba --output_dir ./extracted_llamba_layers
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime
from pathlib import Path

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
        }
    return None

def print_memory_info(stage=""):
    """打印内存信息"""
    mem_info = get_memory_info()
    if mem_info:
        print(f"    📊 Memory {stage}:")
        print(f"       - Allocated: {mem_info['allocated_gb']:.2f}GB")
        print(f"       - Reserved: {mem_info['reserved_gb']:.2f}GB")

def extract_model_layers(model_name, output_dir):
    """
    提取模型的所有层和rotary_emb
    """
    print(f"🚀 Extracting layers from {model_name} model")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_path.absolute()}")
    
    try:
        from modelscope_utils import get_model_modelscope
        
        print(f"  - Loading {model_name} model...")
        model, tokenizer, num_heads, head_dim = get_model_modelscope(model_name, is_minimal=False)
        print("    ✅ Model loaded successfully")
        print_memory_info("(after model loading)")
        
        # 获取模型信息
        total_layers = len(model.backbone.layers)
        print(f"  - Total layers: {total_layers}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Head dimension: {head_dim}")
        
        # 提取rotary_emb
        print(f"\n🔄 Extracting rotary_emb...")
        rotary_emb = model.backbone.rotary_emb.cpu()
        rotary_emb_path = output_path / "rotary_emb.pt"
        torch.save(rotary_emb, rotary_emb_path)
        print(f"    ✅ Rotary embeddings saved to: {rotary_emb_path}")
        
        # 提取每一层
        print(f"\n🔄 Extracting {total_layers} layers...")
        layer_info = {
            'model_name': model_name,
            'total_layers': total_layers,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'extraction_time': datetime.now().isoformat(),
            'layers': []
        }
        
        for layer_idx in range(total_layers):
            print(f"  - Extracting layer {layer_idx}...")
            
            # 提取层并移动到CPU
            layer = model.backbone.layers[layer_idx].cpu()
            
            # 保存层
            layer_path = output_path / f"layer_{layer_idx:02d}.pt"
            torch.save(layer, layer_path)
            
            # 记录层信息
            layer_info['layers'].append({
                'layer_idx': layer_idx,
                'file_path': str(layer_path),
                'layer_type': str(type(layer)),
                'parameters': sum(p.numel() for p in layer.parameters()),
                'state_dict_keys': list(layer.state_dict().keys())
            })
            
            print(f"    ✅ Layer {layer_idx} saved to: {layer_path}")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        
        # 保存元数据
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(layer_info, f, indent=2, ensure_ascii=False)
        print(f"    ✅ Metadata saved to: {metadata_path}")
        
        # 删除模型以释放内存
        del model
        torch.cuda.empty_cache()
        print("    ✅ Model deleted, memory freed")
        print_memory_info("(after model deletion)")
        
        return True, layer_info
        
    except Exception as e:
        print(f"❌ Layer extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def verify_extracted_layers(output_dir):
    """
    验证提取的层文件
    """
    print(f"\n🔍 Verifying extracted layers in {output_dir}")
    print("=" * 60)
    
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.json"
    
    if not metadata_path.exists():
        print("❌ Metadata file not found")
        return False
    
    # 加载元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"  - Model: {metadata['model_name']}")
    print(f"  - Total layers: {metadata['total_layers']}")
    print(f"  - Extraction time: {metadata['extraction_time']}")
    
    # 验证rotary_emb
    rotary_emb_path = output_path / "rotary_emb.pt"
    if rotary_emb_path.exists():
        rotary_emb = torch.load(rotary_emb_path, map_location='cpu')
        print(f"    ✅ Rotary embeddings: {type(rotary_emb)}")
    else:
        print("    ❌ Rotary embeddings not found")
        return False
    
    # 验证每一层
    missing_layers = []
    for layer_info in metadata['layers']:
        layer_path = Path(layer_info['file_path'])
        if layer_path.exists():
            try:
                layer = torch.load(layer_path, map_location='cpu')
                print(f"    ✅ Layer {layer_info['layer_idx']:2d}: {type(layer)}")
            except Exception as e:
                print(f"    ❌ Layer {layer_info['layer_idx']:2d}: Failed to load - {e}")
                missing_layers.append(layer_info['layer_idx'])
        else:
            print(f"    ❌ Layer {layer_info['layer_idx']:2d}: File not found")
            missing_layers.append(layer_info['layer_idx'])
    
    if missing_layers:
        print(f"❌ Missing layers: {missing_layers}")
        return False
    else:
        print("✅ All layers verified successfully")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Extract model layers and rotary embeddings')
    parser.add_argument('--model_name', type=str, default='llama', 
                       choices=['llama', 'unaligned_llamba'],
                       help='Model name to extract layers from')
    parser.add_argument('--output_dir', type=str, default='./extracted_layers',
                       help='Output directory for extracted layers')
    parser.add_argument('--verify', action='store_true',
                       help='Verify extracted layers')
    
    args = parser.parse_args()
    
    print("🚀 Model Layer Extraction Tool")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please ensure GPU is accessible.")
        return 1
    
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    if args.verify:
        # 验证模式
        success = verify_extracted_layers(args.output_dir)
        return 0 if success else 1
    else:
        # 提取模式
        success, metadata = extract_model_layers(args.model_name, args.output_dir)
        
        if not success:
            print("❌ Layer extraction failed.")
            return 1
        
        print("\n" + "=" * 70)
        print("🎉 LAYER EXTRACTION COMPLETED")
        print("=" * 70)
        
        print(f"📊 Summary:")
        print(f"  - Model: {metadata['model_name']}")
        print(f"  - Total layers: {metadata['total_layers']}")
        print(f"  - Output directory: {args.output_dir}")
        
        print(f"\n💡 Next steps:")
        print("1. ✅ All layers extracted successfully")
        print("2. 📁 Layers saved as individual .pt files")
        print("3. 🔍 Use verify mode to check extraction: --verify")
        print("4. 🚀 Use extracted layers in replacement experiments")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
