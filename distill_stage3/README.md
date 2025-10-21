# MOHAWK 第三阶段蒸馏
目前fsdp的配置如 `distill_stage3/fsdp_config.yaml`所示，使用accelerate来便捷配置

在`models/llamba.py`中添加了activation checkpointing的功能

运行方式如：
```
accelerate launch --config_file /home/wuyou/.cache/huggingface/accelerate/default_config.yaml mohawk_stage3_distill——v2.py
```
