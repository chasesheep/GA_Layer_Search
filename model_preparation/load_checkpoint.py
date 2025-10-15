#!/usr/bin/env python3
"""
加载checkpoint的辅助函数

简化checkpoint加载过程
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer


def load_checkpoint(checkpoint_dir, device='cuda'):
    """
    加载替换层后的模型checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录路径
        device: 设备（'cuda'或'cpu'）
    
    Returns:
        model, tokenizer
    
    Example:
        model, tokenizer = load_checkpoint('model_checkpoints/best_4layer')
        inputs = tokenizer("Hello", return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=50)
    """
    checkpoint_path = Path(checkpoint_dir)
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    
    # 导入必要的模块
    import sys
    from pathlib import Path
    
    # 添加项目根目录到path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from model_preparation.modelscope_utils import get_model_modelscope
    
    # 1. 加载模型架构
    print("  📥 Loading model architecture...")
    model, _, _, _ = get_model_modelscope('unaligned_llamba', is_minimal=False)
    print("     ✅ Model architecture loaded")
    
    # 2. 加载权重
    state_dict_path = checkpoint_path / "model_state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict not found: {state_dict_path}")
    
    print(f"  📥 Loading weights from: {state_dict_path.name}")
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print("     ✅ Weights loaded")
    
    # 3. 移到设备
    model = model.to(device)
    model.eval()
    print(f"     ✅ Model ready on {device}")
    
    # 4. 加载tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    print(f"  📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("     ✅ Tokenizer loaded")
    
    print(f"\n✅ Checkpoint loaded successfully!")
    
    return model, tokenizer


def quick_test(checkpoint_dir, test_prompt="Hello, how are you?", max_new_tokens=50):
    """
    快速测试checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录
        test_prompt: 测试prompt
        max_new_tokens: 生成token数
    """
    print("="*70)
    print("快速测试Checkpoint")
    print("="*70)
    
    # 加载
    model, tokenizer = load_checkpoint(checkpoint_dir)
    
    # 推理
    print(f"\n📝 Test prompt: {test_prompt}")
    inputs = tokenizer(test_prompt, return_tensors="pt").to('cuda')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📤 Generated: {generated_text}")
    print(f"\n✅ Test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and test checkpoint')
    parser.add_argument('checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--prompt', type=str, default="Hello, how are you?", 
                       help='Test prompt')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Max new tokens to generate')
    
    args = parser.parse_args()
    
    quick_test(args.checkpoint_dir, args.prompt, args.max_tokens)

