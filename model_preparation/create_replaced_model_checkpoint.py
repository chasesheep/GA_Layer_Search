#!/usr/bin/env python3
"""
创建替换层后的完整模型checkpoint

这个脚本将Llamba模型和指定的Llama层组合，创建一个可以直接加载使用的完整模型checkpoint。

用法:
    python create_replaced_model_checkpoint.py \
        --layers 11 13 17 21 \
        --output_dir ./checkpoints/llamba_replaced_11_13_17_21 \
        --llama_layers_dir ../extracted_llama_layers

生成的checkpoint可以直接加载:
    model = torch.load('checkpoints/llamba_replaced_11_13_17_21/model.pt')
"""

import os
import sys
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

def get_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        }
    return None

def print_memory_info(stage=""):
    """打印内存信息"""
    mem_info = get_memory_info()
    if mem_info:
        print(f"  💾 GPU Memory {stage}:")
        print(f"       - Allocated: {mem_info['allocated_gb']:.2f}GB")
        print(f"       - Reserved: {mem_info['reserved_gb']:.2f}GB")

def load_llamba_model(device='cuda'):
    """加载Llamba模型（未对齐版本）"""
    print("📥 Loading Llamba model (unaligned)...")
    
    try:
        # 添加models模块路径 - 尝试多种可能的位置
        import sys
        from pathlib import Path
        
        # 可能的Gather-and-Aggregate目录位置
        possible_paths = [
            Path(__file__).parent.parent.parent / 'GandA' / 'Gather-and-Aggregate',  # 开发环境
            Path(__file__).parent.parent.parent / 'Gather-and-Aggregate',  # 其他情况
            Path(__file__).parent.parent,  # 如果models就在上级目录
            Path.cwd(),  # 当前工作目录
        ]
        
        gather_dir = None
        for path in possible_paths:
            if (path / 'models').exists():
                gather_dir = path
                print(f"    ℹ️  Found models directory at: {gather_dir}")
                sys.path.insert(0, str(gather_dir))
                break
        
        if gather_dir is None:
            print(f"    ⚠️  Warning: Could not find models directory")
            print(f"    💡 Tip: Make sure 'models' directory (containing llamba.py) is accessible")
            print(f"    💡 You can:")
            print(f"       1. Run from the original GandA/Gather-and-Aggregate directory")
            print(f"       2. Copy the 'models' directory to this project")
            print(f"       3. Set PYTHONPATH to include the directory containing 'models'")
        
        from modelscope_utils import get_model_modelscope
        model, tokenizer, num_heads, head_dim = get_model_modelscope('unaligned_llamba', is_minimal=False)
        
        print("    ✅ Llamba model loaded successfully")
        print_memory_info("(after loading Llamba)")
        
        return model, tokenizer, num_heads, head_dim
    except Exception as e:
        print(f"    ❌ Failed to load Llamba model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def load_extracted_layer(layers_dir, layer_idx, device='cuda'):
    """加载预提取的Llama层"""
    layer_path = Path(layers_dir) / f"layer_{layer_idx:02d}.pt"
    
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_path}")
    
    print(f"    📂 Loading Llama layer {layer_idx} from {layer_path}")
    layer = torch.load(layer_path, map_location='cpu')
    layer = layer.to(device)
    
    return layer

def load_extracted_rotary_emb(layers_dir, device='cuda'):
    """加载预提取的rotary_emb"""
    rotary_emb_path = Path(layers_dir) / "rotary_emb.pt"
    
    if not rotary_emb_path.exists():
        raise FileNotFoundError(f"Rotary embedding file not found: {rotary_emb_path}")
    
    print(f"    📂 Loading Llama rotary_emb from {rotary_emb_path}")
    rotary_emb = torch.load(rotary_emb_path, map_location='cpu')
    rotary_emb = rotary_emb.to(device)
    
    return rotary_emb

def replace_layers_in_model(model, replaced_layers: List[int], llama_layers_dir, device='cuda'):
    """
    在模型中替换指定的层
    
    Args:
        model: Llamba模型
        replaced_layers: 要替换的层索引列表
        llama_layers_dir: Llama层文件目录
        device: 设备
    
    Returns:
        替换后的模型
    """
    print(f"\n🔄 Replacing {len(replaced_layers)} layers: {replaced_layers}")
    
    # 替换rotary_emb
    print("\n  🔄 Replacing rotary_emb...")
    llama_rotary_emb = load_extracted_rotary_emb(llama_layers_dir, device=device)
    model.backbone.rotary_emb = llama_rotary_emb
    print("    ✅ rotary_emb replaced")
    
    # 替换指定的层
    print(f"\n  🔄 Replacing {len(replaced_layers)} transformer layers...")
    for layer_idx in replaced_layers:
        llama_layer = load_extracted_layer(llama_layers_dir, layer_idx, device=device)
        model.backbone.layers[layer_idx] = llama_layer
        print(f"    ✅ Layer {layer_idx} replaced")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
    
    print("\n  ✅ All layers replaced successfully")
    print_memory_info("(after replacement)")
    
    return model

def save_replaced_model_checkpoint(model, tokenizer, replaced_layers, output_dir, 
                                   num_heads=None, head_dim=None, metadata=None):
    """
    保存替换后的模型checkpoint
    
    Args:
        model: 替换后的模型
        tokenizer: tokenizer
        replaced_layers: 替换的层索引
        output_dir: 输出目录
        num_heads: 注意力头数
        head_dim: 头维度
        metadata: 额外的元数据
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving checkpoint to: {output_path}")
    
    # 保存state_dict（主要方式，避免pickle lambda问题）
    state_dict_path = output_path / "model_state_dict.pt"
    print(f"  📝 Saving state_dict to: {state_dict_path}")
    torch.save(model.state_dict(), state_dict_path)
    print(f"    ✅ State dict saved ({state_dict_path.stat().st_size / 1e9:.2f} GB)")
    
    # 尝试保存完整模型（可能失败due to lambda）
    model_path = output_path / "model.pt"
    print(f"  📝 Saving complete model to: {model_path}")
    try:
        torch.save(model, model_path)
        print(f"    ✅ Model saved ({model_path.stat().st_size / 1e9:.2f} GB)")
    except Exception as e:
        print(f"    ⚠️  Complete model save failed (lambda pickle issue): {e}")
        print(f"    💡 Use model_state_dict.pt instead (it's complete and works)")
    
    # 保存tokenizer
    tokenizer_path = output_path / "tokenizer"
    print(f"  📝 Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"    ✅ Tokenizer saved")
    
    # 保存元数据
    checkpoint_metadata = {
        'replaced_layers': replaced_layers,
        'num_replaced_layers': len(replaced_layers),
        'total_layers': 32,
        'base_model': 'Llamba (unaligned)',
        'replacement_source': 'Llama',
        'num_heads': num_heads,
        'head_dim': head_dim,
        'creation_time': datetime.now().isoformat(),
        'device': str(next(model.parameters()).device),
    }
    
    if metadata:
        checkpoint_metadata.update(metadata)
    
    metadata_path = output_path / "checkpoint_info.json"
    print(f"  📝 Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(checkpoint_metadata, f, indent=2)
    print(f"    ✅ Metadata saved")
    
    # 创建README
    readme_path = output_path / "README.txt"
    with open(readme_path, 'w') as f:
        f.write(f"""
Replaced Model Checkpoint
=========================

Base Model: Llamba (unaligned, 8B)
Replacement Source: Llama layers
Replaced Layers: {replaced_layers}
Number of Replaced Layers: {len(replaced_layers)}
Total Layers: 32
Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files:
------
- model.pt              : Complete model (can be loaded directly with torch.load)
- model_state_dict.pt   : Model state dict (requires model architecture to load)
- tokenizer/            : Tokenizer files
- checkpoint_info.json  : Detailed metadata
- README.txt            : This file

Usage:
------
# Load complete model
import torch
model = torch.load('model.pt')
model.eval()

# Or load from state dict
from modelscope_utils import get_model_modelscope
model, tokenizer, _, _ = get_model_modelscope('unaligned_llamba')
model.load_state_dict(torch.load('model_state_dict.pt'))

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

# Run inference
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

For evaluation:
---------------
cd /path/to/GA_Layer_Search/model_preparation
python test_specific_combination.py --checkpoint /path/to/this/checkpoint

""")
    print(f"    ✅ README saved")
    
    print(f"\n✅ Checkpoint saved successfully!")
    print(f"   Total size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e9:.2f} GB")

def main():
    parser = argparse.ArgumentParser(
        description='Create a checkpoint of Llamba model with replaced Llama layers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create checkpoint for layers 11, 13, 17, 21
  python create_replaced_model_checkpoint.py --layers 11 13 17 21 --output_dir ./checkpoints/replaced_11_13_17_21
  
  # Specify custom Llama layers directory
  python create_replaced_model_checkpoint.py --layers 13 17 --llama_layers_dir /path/to/extracted_llama_layers --output_dir ./checkpoints/replaced_13_17
  
  # Use specific GPU
  python create_replaced_model_checkpoint.py --layers 10 14 17 30 --gpu 1 --output_dir ./checkpoints/replaced_10_14_17_30
        """
    )
    
    parser.add_argument('--layers', nargs='+', type=int, required=True,
                       help='Layer indices to replace (e.g., 11 13 17 21)')
    parser.add_argument('--llama_layers_dir', type=str, 
                       default='../extracted_llama_layers',
                       help='Directory containing extracted Llama layers (default: ../extracted_llama_layers)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoint (e.g., ./checkpoints/replaced_11_13_17_21)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    parser.add_argument('--description', type=str, default='',
                       help='Additional description for this checkpoint')
    parser.add_argument('--score', type=float, default=None,
                       help='MMLU score of this configuration (if known)')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("🚀 Create Replaced Model Checkpoint")
    print("="*80)
    print(f"\n📋 Configuration:")
    print(f"   Replaced layers: {args.layers}")
    print(f"   Number of layers: {len(args.layers)}")
    print(f"   Llama layers dir: {args.llama_layers_dir}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   GPU: {args.gpu}")
    print(f"   Device: {device}")
    
    if args.description:
        print(f"   Description: {args.description}")
    if args.score:
        print(f"   MMLU Score: {args.score:.4f}")
    
    # 检查Llama层目录
    llama_layers_path = Path(args.llama_layers_dir)
    if not llama_layers_path.exists():
        print(f"\n❌ Error: Llama layers directory not found: {llama_layers_path}")
        print(f"   Please run extract_layers.py first to extract Llama layers.")
        return 1
    
    # 检查所有需要的层文件是否存在
    missing_layers = []
    for layer_idx in args.layers:
        layer_path = llama_layers_path / f"layer_{layer_idx:02d}.pt"
        if not layer_path.exists():
            missing_layers.append(layer_idx)
    
    if missing_layers:
        print(f"\n❌ Error: Missing layer files for layers: {missing_layers}")
        return 1
    
    print(f"\n✅ All required Llama layer files found")
    
    try:
        # 加载Llamba模型
        model, tokenizer, num_heads, head_dim = load_llamba_model(device=device)
        if model is None:
            return 1
        
        # 替换层
        model = replace_layers_in_model(
            model, 
            args.layers, 
            args.llama_layers_dir, 
            device=device
        )
        
        # 保存checkpoint
        metadata = {}
        if args.description:
            metadata['description'] = args.description
        if args.score:
            metadata['mmlu_score'] = args.score
        
        save_replaced_model_checkpoint(
            model=model,
            tokenizer=tokenizer,
            replaced_layers=args.layers,
            output_dir=args.output_dir,
            num_heads=num_heads,
            head_dim=head_dim,
            metadata=metadata
        )
        
        print("\n" + "="*80)
        print("✅ Checkpoint creation completed successfully!")
        print("="*80)
        print(f"\n📁 Checkpoint location: {Path(args.output_dir).absolute()}")
        print(f"\n🎯 Next steps:")
        print(f"   1. Test the checkpoint:")
        print(f"      python test_checkpoint.py --checkpoint {args.output_dir}")
        print(f"   2. Load in your code:")
        print(f"      model = torch.load('{args.output_dir}/model.pt')")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
