# SpongeBob LLM 🧽

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-grey?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/transformers/index)

**SpongeBob** 是一个从零开始实现、训练的大型语言模型（LLM）项目，具备完整的训练流水线和生产级特性。项目展示了现代LLM从预训练到部署的全流程实现。

## ✨ 核心特性

### 🏗️ 完整的训练流程
- **三阶段训练**: 预训练 (Pretrain) → 有监督微调 (SFT) → 知识蒸馏 (Distill)
- **生产级训练系统**: 完整的checkpoint保存/恢复、梯度累积、混合精度训练
- **灵活的配置管理**: 统一的配置系统支持不同规模的模型实验

### 🔧 先进的模型架构
- **旋转位置编码 (RoPE)**: 支持长序列建模，更好的位置感知能力
- **分组查询注意力 (GQA)**: 优化推理效率，减少KV缓存内存占用
- **RMSNorm + SwiGLU**: 现代归一化和激活函数，提升训练稳定性
- **权重共享**: 嵌入层与输出层参数共享，减少模型参数量

### 📊 企业级工具链
- **完整的评估体系**: 包含PPL评估、交互式测试、基准测试
- **可视化监控**: 支持训练过程实时监控和指标追踪
- **模型服务化**: 准备就绪的API接口（未完成）和部署方案

## 🗂 项目架构

```
spongebob-llm/
├── Config.py               # 模型配置类 (LLMConfig)
├── model.py                # 模型核心架构 (SpongeBob, Attention, RMSNorm etc.)
├── dataset.py              # 数据集加载器 (PretrainDataset, SFTDataset)
├── train_tokenizer.py      # 训练tokenizer脚本
├── pretrain.py             # 预训练脚本（支持checkpoint恢复）
├── SFT.py                  # 有监督微调脚本（支持checkpoint恢复）
├── distill.py              # 知识蒸馏脚本（支持checkpoint恢复）
├── chat.py                 # 交互式对话界面
├── api_server.py           # RESTful API服务（待实现）
├── eval_ppl.py             # 困惑度评估脚本
├── results/                # 模型权重和checkpoint保存目录
└── spongebob_tokenizer/    # 分词器目录
```

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/your-username/SpongeBob_Project.git
cd SpongeBob_Project

# 安装依赖
pip install -r requirements.txt

# 准备分词器（将HuggingFace格式的分词器放入目录）
cp -r your_tokenizer_directory ./spongebob_tokenizer/
```

### 数据准备

**预训练数据格式** (`pretrain.jsonl`):
```json
{"text": "长文本内容..."}
```

**SFT数据格式** (`sft_data.jsonl`):
```json
{"conversations": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]}
```

### 训练流程

#### 1. 预训练阶段
```bash
python pretrain.py \
    --data_path datasets/pretrain.jsonl \
    --batch_size 80 \
    --learning_rate 5e-4 \
    --max_seq_len 512 \
    --save_dir results \
    --epochs 10 \
    --save_step 1000

# 从checkpoint恢复训练
python pretrain.py --resume_from results/latest_checkpoint.pth --epochs 15
```

#### 2. 有监督微调
```bash
python SFT.py \
    --data_path datasets/sft_data.jsonl \
    --pretrained_path results/pretrain_final.pth \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_seq_len 1024 \
    --epochs 3
```

#### 3. 知识蒸馏
```bash
python distill.py \
    --data_path datasets/distill_data.jsonl \
    --sft_path results/sft_final.pth \
    --batch_size 16 \
    --learning_rate 1e-6 \
    --special_token_weight 5.0
```

### 模型评估

#### 困惑度评估
```bash
python eval_ppl.py \
    --model_path results/sft_final.pth \
    --dataset_path datasets/eval_data.jsonl \
    --batch_size 8 \
    --output_file results/ppl_results.json
```

#### 交互式测试
```bash
python eval_model.py \
    --model_mode 1 \          # 0:预训练 1:SFT 2:蒸馏
    --save_dir results \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_seq_len 2048
```

## ⚙️ 配置说明

### 模型配置
在 `Config.py` 中灵活调整模型架构：

```python
config = LLMConfig(
    dim=512,                  # 隐藏层维度
    n_layers=8,               # Transformer层数
    n_heads=8,                # 注意力头数
    n_kv_heads=8,             # KV头数（GQA）
    vocab_size=64000,         # 词表大小
    max_seq_len=8192,         # 最大序列长度
    dropout=0.1,              # Dropout率
    rope_theta=10000,         # RoPE基频
    norm_eps=1e-5,           # 归一化epsilon
)
```

### 训练配置
支持丰富的训练参数调节：
- **学习率调度**: 余弦退火 + 线性warmup
- **梯度处理**: 梯度累积、梯度裁剪
- **精度控制**: BF16/FP16混合精度训练
- **内存优化**: 激活checkpointing、梯度检查点

## 📊 监控与可视化

### 训练监控
项目集成多种监控方案：

```bash
# 使用SwanLab（默认）
python pretrain.py --use_wandb=True

# 或使用TensorBoard
tensorboard --logdir results/tensorboard_logs
```

### 评估指标
- **训练损失曲线**: 实时监控收敛情况
- **学习率变化**: 可视化调度策略效果
- **困惑度指标**: 量化模型语言建模能力
- **生成质量**: 人工评估对话流畅度

## 🏭 生产部署

### API服务(未完成)
```python
# 启动模型服务
python api_server.py \
    --model_path results/sft_final.pth \
    --port 8080 \
    --workers 4
```

### 性能优化
- **量化推理**: 支持INT8/INT4量化，减少内存占用
- **批处理优化**: 动态批处理，提高吞吐量
- **缓存机制**: KV缓存优化，降低计算开销

## 🔬 技术深度

### 核心创新点
1. **完整的训练系统**: 从数据预处理到模型部署的端到端解决方案
2. **生产级可靠性**: 完善的checkpoint管理和错误恢复机制
3. **模块化设计**: 每个组件都可独立替换和扩展
4. **性能优化**: 针对训练和推理的多层次优化策略

### 架构优势
- **可扩展性**: 支持从百万参数到十亿参数级别的模型
- **可复现性**: 完整的实验记录和配置管理
- **可维护性**: 清晰的代码结构和完整的文档

## 🤝 贡献指南

我们欢迎各种形式的贡献！

### 开发流程
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 代码规范
- 遵循PEP 8编码规范
- 添加适当的类型注解
- 编写单元测试和文档
- 确保向后兼容性

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目在实现过程中参考了以下优秀开源项目：
- [Llama](https://github.com/facebookresearch/llama) - Meta的LLaMA模型
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face transformers库
- [Nanotron](https://github.com/huggingface/nanotron) - 轻量级训练框架

---

**SpongeBob LLM** - 让每个人都能理解和构建大型语言模型 🚀