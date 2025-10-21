"""
使用分布式框架accelerate+FSDP进行蒸馏训练
"""
import time
import torch
from accelerate import Accelerator
import csv
import os
os.environ.setdefault("ACCELERATE_USE_CPU_OBJECT_GATHER", "1") # 启用CPU端对象聚合，减少GPU内存占用
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") # 启用可扩展内存分配，减少碎片化
import json
from safetensors.torch import save_file
from mohawk_utils_distill import get_model, build_fineweb_dataloader, build_openhermes_dataloader, compute_distillation_loss, build_wsd_scheduler, run_eval, model_forward_logits, eval_on_rank0
class config:
    lr = 1e-5
    weight_decay = 1e-1
    max_grad_norm = 1.0
    beta1 = 0.9
    beta2 = 0.95
    train_epochs = 3
    batch_size = 1
    seq_len = 2048
    eval_interval = 100
    eval_limit = 10
    save_interval = 100
    temperature = 2.0
    total_steps = 100000
    gradient_checkpointing = True
    resume_dir = None  # "./ckpt_distill_1020/step_100"
replace_layers = [11, 13, 17, 29]  # 替换的层索引
# replace_layers = [9]
log_file = "distill_log_1020_2.csv"
teacher_model, tokenizer = get_model("llama")
student_model, _ = get_model("llamba")
# 手动为学生模型开启激活检查点
if config.gradient_checkpointing:
    student_model.backbone.gradient_checkpointing = True
teacher_model.eval()
dataloader = build_fineweb_dataloader(
    tokenizer,
    batch_size=config.batch_size,
    seq_len=config.seq_len,
    num_workers=0,
)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, config.beta2))
scheduler = build_wsd_scheduler(
    optimizer,
    config.total_steps,
)
for p in teacher_model.parameters():
    p.requires_grad_(False)

student_model.train()
# 层替换：只复制state_dict，不共享模块对象；在FSDP包装前完成
with torch.no_grad():
    # rotary_emb 复制权重
    if hasattr(student_model.backbone, "rotary_emb") and hasattr(teacher_model.backbone, "rotary_emb"):
        student_model.backbone.rotary_emb.load_state_dict(
            teacher_model.backbone.rotary_emb.state_dict(), strict=False
        )
    # 指定层复制权重
    for layer_idx in replace_layers:
        student_layer = student_model.backbone.layers[layer_idx]
        teacher_layer = teacher_model.backbone.layers[layer_idx]
        student_layer.load_state_dict(teacher_layer.state_dict(), strict=False)
accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=16)
student_model, optimizer, dataloader, scheduler = accelerator.prepare(student_model, optimizer, dataloader, scheduler)
# 这段被注释调的代码是FSDP1.0版本的写法，保留以供参考
# 5) 先 prepare 模型与数据（不要传 optimizer/scheduler）
# student_model,optimizer, dataloader = accelerator.prepare(student_model, optimizer, dataloader)
# student_model = accelerator.prepare_model(student_model)

# # 3) 基于已 wrap 的模型参数创建优化器（避免 data_ptr 映射丢失）
# optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# optimizer = accelerator.prepare_optimizer(optimizer)

# # 4) dataloader 最后 prepare
# dataloader = accelerator.prepare_data_loader(dataloader)
# # 6) 用“已包装”的模型参数创建优化器与调度器（避免参数映射失败）
# # optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# scheduler = build_wsd_scheduler(optimizer, total_steps=config.total_steps)
is_main_process = accelerator.is_main_process
device = accelerator.device
teacher_model.to(device, dtype=torch.bfloat16)
global_step = 0
start_epoch = 0
if config.resume_dir is not None: # 从检查点恢复
    Accelerator.wait_for_everyone()
    # 恢复模型/优化器/调度器状态/随机状态（所有进程）
    accelerator.load_state(config.resume_dir)
    # 恢复全局步数
    resume_state_path = os.path.join(config.resume_dir, "train_state.json")
    if is_main_process and os.path.exists(resume_state_path):
        with open(resume_state_path, 'r') as f:
            train_state = json.load(f)
        global_step = int(train_state.get("global_step", 0))
        start_epoch = int(train_state.get("epoch", 0))
    # 广播 global_step 到所有进程
    global_step = accelerator.broadcast(torch.tensor(global_step, device=device), src=0).item()
    start_epoch = accelerator.broadcast(torch.tensor(start_epoch, device=device), src=0).item()
    accelerator.print(f"Resumed from {config.resume_dir}, epoch {start_epoch}, global_step {global_step}")
    accelerator.wait_for_everyone()

# 初始化日志文件（只在首次运行时创建，继续训练时不覆盖）
if is_main_process and log_file is not None and not os.path.exists(log_file):
    with open(log_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "distill_loss", "eval_acc"])


# 开始的时候先评估一次
# if global_step == 0:
#     eval_acc = eval_on_rank0(accelerator, student_model, tokenizer, tasks=["mmlu"], limit=config.eval_limit)
#     if log_file is not None and is_main_process:
#         print(f"Step {global_step}, Eval Acc: {eval_acc:.4f}")
#         with open(log_file, 'a') as f:
#             writer = csv.writer(f)
#             writer.writerow([0, '', eval_acc])
start_time = time.time()
for epoch in range(start_epoch, config.train_epochs):
    for step, batch in enumerate(dataloader):
        # step_start_time = time.time()
        with accelerator.accumulate(student_model):
            # start_time = time.time()
            with torch.no_grad():
                teacher_outputs = model_forward_logits(teacher_model, batch)
            # end_time = time.time()
            # print(f"Teacher forward time: {end_time - start_time:.4f} seconds")
            # start_time = time.time()
            student_outputs = model_forward_logits(student_model, batch)
            # end_time = time.time()
            # print(f"Student forward time: {end_time - start_time:.4f} seconds")
            attention_mask = batch['attention_mask'] if 'attention_mask' in batch else None
            # start_time = time.time()
            loss = compute_distillation_loss(
                student_outputs,
                teacher_outputs,
                attention_mask,
                temperature=config.temperature,
            )
            # end_time = time.time()
            # print(f"Loss computation time: {end_time - start_time:.4f} seconds")
            # start_time = time.time()
            accelerator.backward(loss)
            # end_time = time.time()
            # print(f"Backward time: {end_time - start_time:.4f} seconds")
            # step_end_time = time.time()
            # print(f"Total step time: {step_end_time - step_start_time:.4f} seconds")
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(student_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                end_time = time.time()
                print(f"Step {global_step} time: {end_time - start_time:.4f} seconds")
                start_time = time.time()
                if is_main_process:
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
                    with open(log_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, loss.item(), ""])
                if global_step % config.eval_interval == 0:
                    teacher_model.to('cpu')
                    eval_acc = eval_on_rank0(accelerator, student_model, tokenizer, tasks=["mmlu"], limit=config.eval_limit)
                    teacher_model.to(device, dtype=torch.bfloat16)
                if is_main_process and global_step % config.eval_interval == 0:
    
                    print(f"Step {global_step}, Eval Acc: {eval_acc:.4f}")
                    if log_file is not None:
                        with open(log_file, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([global_step, '', eval_acc])

                if  global_step % config.save_interval == 0:
                    teacher_model.to('cpu')
                    accelerator.wait_for_everyone()
                    save_path = f"./ckpt_distill_1020_2/step_{global_step}"
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
                    # 从 FSDP/Accelerate 收集完整 state_dict 所有进程都要执行（集体通信）
                    full_state_dict = accelerator.get_state_dict(student_model)
                    # 4) 只有主进程保存可独立加载的权重与分词器(不用save_pretrained, LlambaLMHeadModel的save_pretrained不是HuggingFace 的PreTrainedModel的实现，不能接收state_dict参数)
                    if accelerator.is_main_process:
                        
                        try:
                            weight_path = os.path.join(save_path, "model.safetensors")
                            save_file(full_state_dict, weight_path, metadata={"format": "pt_safetensors"})
                        except Exception as e:
                            #回退到原始的torch.save方法
                            weight_path = os.path.join(save_path, "pytorch_model.bin")
                            torch.save(full_state_dict, weight_path)
                        tokenizer.save_pretrained(save_path)
                        print(f"Model saved to {save_path}")
                        # 3) 额外保存训练计数（epoch/global_step），便于恢复
                        with open(os.path.join(save_path, "train_state.json"), "w") as f:
                            json.dump({"epoch": epoch, "global_step": global_step}, f)
                        print(f"Model saved to {save_path}")
                    del full_state_dict
                    teacher_model.to(device, dtype=torch.bfloat16)
                    accelerator.wait_for_everyone()
                
            if global_step >= config.total_steps:
                break