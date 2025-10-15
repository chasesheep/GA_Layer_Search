#!/usr/bin/env python3
"""
ModelScope-compatible utilities for the Gather-and-Aggregate experiments
This module provides functions to load models via ModelScope instead of Hugging Face
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run_eval(model, tokenizer, tasks, num_fewshot=0, batch_size=32, limit=None, cache_requests=False):
    """Evaluation function - same as original utils.py"""
    results = simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size),
        limit=limit,
        tasks=tasks,
        num_fewshot=num_fewshot,
        device="cuda",
        log_samples=False,
        batch_size=batch_size,
        verbosity="ERROR",
        cache_requests=cache_requests,
    )
    return results

def get_model_modelscope(model_name, is_minimal=False):
    """
    Modified get_model function that uses ModelScope for Llama models
    This function loads models via ModelScope instead of Hugging Face
    
    Note: Requires 'models' directory in project root containing llamba.py
    """
    torch.cuda.empty_cache()
    
    if model_name == 'llama':
        # Use ModelScope for Llama 3.1 8B Instruct
        from modelscope import snapshot_download
        
        model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
        print(f"Loading Llama model via ModelScope: {model_id}")
        
        # Download model (will use cache if already downloaded)
        # Use project root modelscope_cache directory
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.join(project_root, "modelscope_cache"))
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"  # 使用 Flash Attention 2
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.config.use_cache = False
        num_heads, head_dim = 32, 128

        # Alias the layers to match the Mamba naming scheme
        model.backbone = model.model
        for layer in model.backbone.layers:
            layer.layer_idx = layer.self_attn.layer_idx
            layer.mixer = layer.self_attn
            layer.mixer.out_proj = layer.mixer.o_proj
            
    elif model_name == 'llama_tokenizer':
        # Just get the tokenizer for other models
        from modelscope import snapshot_download
        import os
        
        model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.join(project_root, "modelscope_cache"))
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return tokenizer
        
    elif model_name == 'falcon':
        # Falcon model - use original Hugging Face
        model = AutoModelForCausalLM.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-mamba-7b-instruct')
        num_heads, head_dim = 2 * model.config.hidden_size, 1

    elif model_name == 'llamba':
        # Llamba model - use Hugging Face for model, ModelScope for tokenizer
        import sys
        from pathlib import Path
        # Add project models directory to path
        models_dir = Path(__file__).parent.parent / 'models'
        if models_dir.exists():
            sys.path.insert(0, str(models_dir.parent))
        from models.llamba import LlambaLMHeadModel
        from modelscope import snapshot_download
        import os
        
        model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B-untied", strict=True)
        
        # Use ModelScope for tokenizer (since we need Llama tokenizer)
        model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.join(project_root, "modelscope_cache"))
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        num_heads, head_dim = 32, 128

    elif model_name == 'zamba' or model_name == 'zamba2':
        # Zamba2-7B model - use official AutoModelForCausalLM
        # Following official example from https://huggingface.co/Zyphra/Zamba2-7B
        print("Loading Zamba2-7B using AutoModelForCausalLM...")
        
        # 获取 token
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        
        tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B", token=token, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            "Zyphra/Zamba2-7B", 
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            token=token,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # 使用 Flash Attention 2，与 Llama 对齐
        )
        
        num_heads, head_dim = 32, 224

        # Alias the layers to match the Mamba naming scheme
        # Check if model has 'model' attribute (for compatibility)
        if hasattr(model, 'model') and not hasattr(model, 'backbone'):
            model.backbone = model.model
        elif not hasattr(model, 'backbone'):
            # If neither exists, we might need to handle this differently
            # For now, just create a reference to the model itself
            print("⚠️  Warning: Model structure may differ from expected")
            if hasattr(model, 'transformer'):
                model.backbone = model.transformer
            else:
                model.backbone = model  # fallback

    elif model_name == "unaligned_llamba":
        # Unaligned Llamba model - use Hugging Face for model, ModelScope for tokenizer
        import sys
        from pathlib import Path
        # Add project models directory to path
        models_dir = Path(__file__).parent.parent / 'models'
        if models_dir.exists():
            sys.path.insert(0, str(models_dir.parent))
        from models.llamba import LlambaLMHeadModel
        from modelscope import snapshot_download
        import os
        
        # Load model from Hugging Face (no special permissions needed)
        model = LlambaLMHeadModel.from_pretrained("goombalab/Llamba-8B-untied-unaligned", strict=True)
        
        # Use ModelScope for tokenizer (since we need Llama tokenizer and HF requires permissions)
        model_id = "LLM-Research/Meta-Llama-3.1-8B-Instruct"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.environ.get('MODELSCOPE_CACHE', os.path.join(project_root, "modelscope_cache"))
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        num_heads, head_dim = 32, 128

    else:
        raise ValueError(f"Unknown model {model_name}")
    
    # For Zamba2, model is already on cuda with bfloat16 from device_map
    # For other models, move to cuda and convert dtype
    if model_name not in ['zamba', 'zamba2']:
        model = model.to(torch.bfloat16).to('cuda')
    
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer, num_heads, head_dim

def get_minimal_model_modelscope(model_name, layer_idx):
    """Get minimal model using ModelScope"""
    model, tokenizer, num_heads, head_dim = get_model_modelscope(model_name)
    model.backbone.layers = model.backbone.layers[0:layer_idx+1]
    return model, tokenizer, num_heads, head_dim

def keep_heads(model, layer_idx, heads, num_heads, head_dim):
    """
    Keep only the specified heads in the output projection of the specified layer.
    Zeroing the output projection of the other heads essentially removes their contribution to the output.
    """
    # Extract the weights of the layer's output projection
    weights = model.backbone.layers[layer_idx].mixer.out_proj.weight.data
    weights_view = weights.view(-1, num_heads, head_dim)

    # Create a mask to keep the specified heads
    mask = torch.zeros_like(weights_view)
    mask[:, heads, :] = 1.0

    # Apply the mask (in-place)
    weights_view *= mask

def remove_heads(model, layer_idx, num_heads, head_dim, heads):
    """
    Remove the specified heads from the output projection of the specified layer.
    Zeroing the output projection of the specified heads essentially removes their contribution to the output.
    """
    # Extract the weights of the layer's output projection
    weights = model.backbone.layers[layer_idx].mixer.out_proj.weight.data
    weights_view = weights.view(-1, num_heads, head_dim)

    # Create a mask to remove the specified heads
    mask = torch.ones_like(weights_view)
    mask[:, heads, :] = 0.0

    # Apply the mask (in-place)
    weights_view *= mask