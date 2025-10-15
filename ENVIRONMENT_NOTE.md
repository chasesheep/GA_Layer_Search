# 环境配置说明

## 推荐方案

由于Python依赖的复杂性，我们提供两种环境配置方案：

### 方案1：使用提供的环境配置（推荐）

如果您有访问我们服务器的权限，可以直接使用已配置好的环境：

```bash
# 激活已有环境
export PATH="/data/huzhuangfei/conda_envs/ganda_new/bin:$PATH"
cd GA_Layer_Search

# 直接使用
cd model_preparation
python extract_layers.py --model_name llama --output_dir ../extracted_llama_layers
```

### 方案2：自行配置环境

```bash
# 创建环境
conda create -n ga_layer_search python=3.10 -y
conda activate ga_layer_search

# 安装依赖
pip install -r requirements.txt

# 如遇依赖冲突，请联系作者
```

## 已知问题

- scipy 1.14.1 + numpy 1.26.4 在某些环境下可能出现兼容性问题
- 建议使用Python 3.10（其他版本未测试）
- GPU需要CUDA 12.1+

## 联系方式

如遇环境配置问题，请联系：
- **作者**: Zhuangfei Hu
- **邮箱**: [您的邮箱]

我们可以提供：
1. 已配置好的conda环境
2. 依赖troubleshooting支持
3. Docker镜像（如需要）

