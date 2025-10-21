"""
复现原始版本的MOHAWK，即用fineweb-EDU和OpenHermes-2.5做蒸馏数据
实现函数：
损失计算、数据集加载、评估函数、学习率调度器、模型加载
"""
import os
import math
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple
# 计算损失
import torch.nn.functional as F
def compute_distillation_loss(student_logits, teacher_logits, attention_mask, temperature=2.0):
    """
    计算蒸馏损失
    :param student_logits: 学生模型的logits
    :param teacher_logits: 教师模型的logits
    :param attention_mask: 注意力掩码
    :param temperature: 温度参数
    :return: 蒸馏损失
    """
    assert student_logits.size() == teacher_logits.size(), "学生和教师的logits尺寸必须相同"
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
    kl_per_token = F.kl_div(student_probs, teacher_probs, reduction='none').sum(-1)# 计算每个token的KL散度
    effect_tokens = attention_mask.sum().clamp_min(1)
    loss = (kl_per_token * attention_mask).sum() / effect_tokens
    return loss * (temperature**2)


# 数据集加载
from torch.utils.data import Dataset, DataLoader,IterableDataset

from datasets import load_dataset
class FinewebDatasetPacker(IterableDataset):
    """
    将streaming数据集中的'text' 打包为固定长度的token块
    每条样本返回：
    - input_ids: (seq_len,)
    - attention_mask: (seq_len,)
    """

    def __init__(self, dataset, tokenizer, seq_len: int, add_eos: bool = True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        self.eos_id = tokenizer.eos_token_id if add_eos else 0

    def __iter__(self):
        buffer: list[int] = []
        for sample in self.dataset:
            text = sample.get('text', '') 
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if self.add_eos and self.eos_id is not None:
                ids.append(self.eos_id)
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:# 每次取出一个固定长度的chunk
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long)
                }
def _rank_world_from_env() -> Tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world
# def build_fineweb_dataloader(
#         tokenizer,
#         batch_size: int,
#         seq_len: int,
#         seed: int = 42,
#         buffer_size: int = 100_000,
#         num_workers: int = 0,
# ):
#     """
#     streaming方式加载fineweb-EDU数据集，并打包为定长token块
#     """
#     dataset = load_dataset("aynetdia/fineweb-edu-score-4-dedup", split="train", streaming=True)
#     dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
#     rank, world_size = _rank_world_from_env()
#     if world_size > 1:
#         dataset = dataset.shard(num_shards=world_size, index=rank)
#     packed_dataset = FinewebDatasetPacker(dataset, tokenizer, seq_len)
#     return DataLoader(
#         packed_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )
from huggingface_hub import snapshot_download
def build_fineweb_dataset(tokenizer, seq_len: int, local_cache_dir: str = "/data3/wuyou/hf_datasets/fineweb-edu"):
    """
    优先从本地缓存加载数据集，避免网络超时。
    如果本地不存在，则先下载再加载。
    """
    dataset_name = "aynetdia/fineweb-edu-score-4-dedup"
    # 检查本地目录是否已包含数据集文件
    if not os.path.exists(local_cache_dir) or not any(f.endswith('.parquet') for f in os.listdir(os.path.join(local_cache_dir, "data"))):
        print(f"Dataset not found locally. Downloading {dataset_name} to {local_cache_dir}...")
        # 使用 snapshot_download 下载整个数据集到指定目录
        snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=local_cache_dir)
        print("Download complete.")
    else:
        print(f"Loading dataset from local cache: {local_cache_dir}")
    # 从本地的 parquet 文件流式加载
    # 注意：这里的路径要指向包含 parquet 文件的 'data' 子目录
    data_files = os.path.join(local_cache_dir, "data", "train-*.parquet")
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
    rank, world_size = _rank_world_from_env()
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    packed_dataset = FinewebDatasetPacker(dataset, tokenizer, seq_len)
    return packed_dataset
    
def build_fineweb_dataloader(
        tokenizer,
        batch_size: int,
        seq_len: int,
        seed: int = 42,
        buffer_size: int = 100_000,
        num_workers: int = 2,
        # 定义一个本地缓存目录
        local_cache_dir: str = "/data3/wuyou/hf_datasets/fineweb-edu",
):
    """
    优先从本地缓存加载数据集，避免网络超时。
    如果本地不存在，则先下载再加载。
    """
    dataset_name = "aynetdia/fineweb-edu-score-4-dedup"
    # 检查本地目录是否已包含数据集文件
    if not os.path.exists(local_cache_dir) or not any(f.endswith('.parquet') for f in os.listdir(os.path.join(local_cache_dir, "data"))):
        print(f"Dataset not found locally. Downloading {dataset_name} to {local_cache_dir}...")
        # 使用 snapshot_download 下载整个数据集到指定目录
        snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=local_cache_dir)
        print("Download complete.")
    else:
        print(f"Loading dataset from local cache: {local_cache_dir}")
    # 从本地的 parquet 文件流式加载
    # 注意：这里的路径要指向包含 parquet 文件的 'data' 子目录
    data_files = os.path.join(local_cache_dir, "data", "train-*.parquet")
    dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    rank, world_size = _rank_world_from_env()
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    packed_dataset = FinewebDatasetPacker(dataset, tokenizer, seq_len)
    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
def convert_to_text(conversations: list[dict]) -> str:
    """
    将OpenHermes样本字典转换为文本字符串
    """
    lines = []
    for turn in conversations:
        role = turn.get("from", "")
        content = turn.get("value", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\n"

def _dynamic_pad_collate_fn(batch: list[Dict[str, Any]], tokenizer) -> Dict[str, torch.Tensor]:
    """
    动态padding的collate函数
    """
    max_len = max(x["input_ids"].size(0) for x in batch)
    batch_size = len(batch)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        seq_len = x["input_ids"].size(0)
        input_ids[i, :seq_len] = x["input_ids"]
        attention_mask[i, :seq_len] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}
    
def build_openhermes_dataloader(
        tokenizer,
        batch_size: int,
        seq_len: int,
        seed: int = 42,
        num_workers: int = 4,
):
    """
    保留每条对话一个样本，不跨样本打包
    """
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    def preprocess_fn(sample):
        text = convert_to_text(sample.get("conversations", []))
        return {"text": text}
    dataset = dataset.map(preprocess_fn, remove_columns=[col for col in dataset.column_names if col != "text"])
    def tok_fn(batch):
        enc = tokenizer(batch["text"], add_special_tokens=False, truncation=True, max_length=seq_len)
        return {"input_ids": enc["input_ids"]}
    dataset = dataset.map(tok_fn, batched=True, remove_columns=["text"])
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0)
    def collate_fn(batch):
        return _dynamic_pad_collate_fn(batch, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

from huggingface_hub import hf_hub_download
import json
from transformers import BitsAndBytesConfig
def get_model(model_name, is_minimal=False):
    torch.cuda.empty_cache()
    
    if model_name == 'llama':
        # from models.modeling_llama import LlamaForCausalLM
        # model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', attn_implementation="flash_attention_2")
        # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
        # model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        # model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct', attn_implementation="flash_attention_2")
        # model = AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
        # tokenizer = AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B-Instruct')
        print("Loading Llama model...")
        # model = AutoModelForCausalLM.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct', attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_type=torch.bfloat16,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            'unsloth/Meta-Llama-3.1-8B-Instruct',
            attn_implementation="flash_attention_2",
            # quantization_config=bnb_config,
        )

        tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')
        model.config.use_cache = False
        # num_heads, head_dim = 32, 128

        # Alias the layers to match the Mamba naming scheme
        model.backbone = model.model
        for layer in model.backbone.layers:
            layer.layer_idx = layer.self_attn.layer_idx
            layer.mixer = layer.self_attn
            layer.mixer.out_proj = layer.mixer.o_proj
            
    elif model_name == 'llamba':
        # from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel
        from models.llamba import LlambaLMHeadModel
        print("Loading Llamba model...")
        # model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-1B", strict=True, torch_dtype=torch.bfloat16)
        # model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-1B", strict=True)
        # model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B", strict=True)
        # model = LlambaLMHeadModel.from_pretrained("cartesia-ai/Llamba-8B-untied", strict=True)
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        # tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        # num_heads, head_dim = 32, 128

        repo_id = "cartesia-ai/Llamba-8B"
        cfg_path = hf_hub_download(repo_id, filename="config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if "d_model" not in cfg and "hidden_size" in cfg:
            cfg["d_model"] = cfg["hidden_size"]
        if "vocab_size" not in cfg and "tokenizer_vocab_size" in cfg:
            cfg["vocab_size"] = cfg["tokenizer_vocab_size"]
        model = LlambaLMHeadModel.from_pretrained(repo_id, config=cfg, strict=True)
    else:
        raise ValueError(f"Unknown model {model_name}")
    
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    return model, tokenizer
import math
from torch.optim.lr_scheduler import LambdaLR
def build_wsd_scheduler(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.03,
        decay_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
        decay_style: str = 'linear',
):
    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = int(total_steps * decay_ratio)
    stable_steps = total_steps - warmup_steps - decay_steps
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        elif current_step < total_steps:
            decay_step = current_step - warmup_steps - stable_steps
            if decay_style == 'linear':
                return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * (decay_step / decay_steps))
            elif decay_style == 'cosine':
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
                return max(min_lr_ratio, cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio)
    
    return LambdaLR(optimizer, lr_lambda, -1)
import inspect 
def model_forward_logits(model, batch: Dict[str, torch.Tensor]):
    """
    兼容不接受attention_mask的模型（llamba）
    返回logits张量
    """
    sig = inspect.signature(model.forward)
    # print(f"model{model.__class__.__name__} forward args:", sig.parameters.keys())
    if 'attention_mask' in sig.parameters:
        out = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"])
    else:
        out = model(input_ids = batch["input_ids"])
    return out.logits if hasattr(out, 'logits') else out

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
def run_eval(model, tokenizer, tasks, limit=10, num_fewshot=0, batch_size=4, device='cpu'):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=batch_size)
    results = simple_evaluate(
        model=lm,
        limit=limit,
        tasks=tasks,
        num_fewshot=num_fewshot,
        device=device,
        log_samples=False,
        batch_size=batch_size,
        verbosity="ERROR",
        cache_requests=True,
    )
    return results["results"]["mmlu"]["acc,none"]

import gc
# 辅助函数：用训练中的 FSDP 模型导出完整权重，在 rank0 构建非FSDP副本做评测
def eval_on_rank0(accelerator, train_model, tokenizer, tasks, limit):
    # 1) 所有进程同步进入评测阶段
    accelerator.wait_for_everyone()

    # 2) 所有进程一起参与 state_dict 聚合（这是集体通信，不能只在 rank0 调）
    state_dict = accelerator.get_state_dict(train_model)

    # 3) 非主进程丢弃结果并等待主进程评测结束
    if not accelerator.is_main_process:
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        return None

    # 用学生config新建“空模型”，避免从Hub再加载一份权重
    from models.llamba import LlambaLMHeadModel
    base_cfg = accelerator.unwrap_model(train_model).config
    eval_model = LlambaLMHeadModel(config=base_cfg)
    missing, unexpected = eval_model.load_state_dict(state_dict, strict=False)
    del state_dict

    # 选其一：CPU评估（最稳）
    # device_str = "cpu"
    # eval_model.to("cpu").eval()

    # 或：放到rank0的GPU评估，结束后彻底释放
    device_str = f"cuda:{accelerator.local_process_index}"
    eval_model.to(device_str).eval()

    acc = run_eval(eval_model, tokenizer, tasks=tasks, limit=limit, device=device_str)

    del eval_model; gc.collect(); torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    return acc

def eval_ddp(model, tokenizer, tasks, limit=10, device='cuda'):
    acc = run_eval(model, tokenizer, tasks=tasks, limit=limit, device=device)
    return acc