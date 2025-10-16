#!/usr/bin/env python3
"""
加载checkpoint的辅助函数（混合架构版）
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLAMA_LAYERS_DIR = PROJECT_ROOT / "extracted_llama_layers"


def _load_metadata(checkpoint_path: Path):
    """Load checkpoint metadata if available."""
    metadata_path = checkpoint_path / "checkpoint_info.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as err:
        print(f"  ⚠️  Failed to parse metadata ({metadata_path}): {err}")
        return {}


def _resolve_llama_layers_dir(llama_layers_dir, metadata):
    """Determine the directory containing extracted Llama layers."""
    candidates = [
        llama_layers_dir,
        metadata.get("llama_layers_dir"),
        DEFAULT_LLAMA_LAYERS_DIR,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        "Unable to determine Llama layers directory. "
        "Please pass --llama_layers_dir or ensure extracted layers exist."
    )


def _load_extracted_layer(layers_dir: Path, layer_idx: int):
    layer_path = layers_dir / f"layer_{layer_idx:02d}.pt"
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_path}")
    return torch.load(layer_path, map_location="cpu")


def _load_extracted_rotary_emb(layers_dir: Path):
    rotary_emb_path = layers_dir / "rotary_emb.pt"
    if not rotary_emb_path.exists():
        raise FileNotFoundError(f"Rotary embedding file not found: {rotary_emb_path}")
    return torch.load(rotary_emb_path, map_location="cpu")


def _prepare_llama_layers(model, replaced_layers, layers_dir: Path):
    """Ensure the model architecture matches the checkpoint by injecting Llama layers."""
    if not replaced_layers:
        return

    print(f"  🔄 Preparing architecture for layers {replaced_layers} from {layers_dir}")
    rotary_emb = _load_extracted_rotary_emb(layers_dir)
    model.backbone.rotary_emb = rotary_emb

    for layer_idx in replaced_layers:
        llama_layer = _load_extracted_layer(layers_dir, layer_idx)
        model.backbone.layers[layer_idx] = llama_layer


def load_checkpoint(checkpoint_dir, device='cuda', llama_layers_dir=None):
    """
    加载替换层后的模型checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    print(f"📂 Loading checkpoint from: {checkpoint_path}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(device, str) and device.isdigit():
        device = f'cuda:{device}'
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Please specify --device cpu or ensure GPU is accessible.")
    print(f"  🎯 Target device: {device}")

    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from model_preparation.modelscope_utils import get_model_modelscope

    print("  📥 Loading model architecture...")
    model, _, _, _ = get_model_modelscope('unaligned_llamba', is_minimal=False)
    print("     ✅ Model architecture loaded")

    metadata = _load_metadata(checkpoint_path)
    replaced_layers = metadata.get('replaced_layers', [])
    resolved_layers_dir = _resolve_llama_layers_dir(llama_layers_dir, metadata)

    if replaced_layers:
        _prepare_llama_layers(model, replaced_layers, resolved_layers_dir)

    state_dict_path = checkpoint_path / "model_state_dict.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict not found: {state_dict_path}")

    print(f"  📥 Loading weights from: {state_dict_path.name}")
    try:
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
    except TypeError:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        guidance = "Check whether the required Llama layers were injected (use --llama_layers_dir)"
        raise RuntimeError(f"Failed to load state dict: {err}\nHint: {guidance}") from err
    print("     ✅ Weights loaded")

    model = model.to(device)
    model.eval()
    print(f"     ✅ Model ready on {device}")

    tokenizer_path = checkpoint_path / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    print(f"  📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("     ✅ Tokenizer loaded")

    print(f"\n✅ Checkpoint loaded successfully!")

    return model, tokenizer


def quick_test(checkpoint_dir, test_prompt="Hello, how are you?", max_new_tokens=50,
               device=None, llama_layers_dir=None):
    print("="*70)
    print("快速测试Checkpoint")
    print("="*70)

    model, tokenizer = load_checkpoint(
        checkpoint_dir,
        device=device,
        llama_layers_dir=llama_layers_dir,
    )
    model_device = next(model.parameters()).device

    print(f"\n📝 Test prompt: {test_prompt}")
    encoded = tokenizer(test_prompt, return_tensors="pt")
    encoded = {k: v.to(model_device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]

    generation_kwargs = {
        "input_ids": input_ids,
        "max_length": input_ids.shape[1] + max_new_tokens,
    }

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📤 Generated: {generated_text}")
    print(f"\n✅ Test passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load and test checkpoint (hybrid)')
    parser.add_argument('checkpoint_dir', type=str, help='Checkpoint directory')
    parser.add_argument('--prompt', type=str, default="Hello, how are you?",
                       help='Test prompt')
    parser.add_argument('--max_tokens', type=int, default=50,
                       help='Max new tokens to generate')
    parser.add_argument('--device', type=str, default=None,
                       help="Device for loading the checkpoint, e.g. 'cuda:0', '7', or 'cpu'")
    parser.add_argument('--llama_layers_dir', type=str, default=None,
                       help="Directory containing extracted Llama layers")

    args = parser.parse_args()

    quick_test(
        args.checkpoint_dir,
        args.prompt,
        args.max_tokens,
        device=args.device,
        llama_layers_dir=args.llama_layers_dir,
    )
