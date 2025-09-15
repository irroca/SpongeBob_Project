# SpongeBob LLM 🧽

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-grey?style=for-the-badge&logo=huggingface)](https://huggingface.co/docs/transformers/index)

**SpongeBob** 是一个从零开始实现、训练的大型语言模型（LLM）项目。它不仅是另一个模型实现，更是一个完整、透明且易于理解的 LLM 构建 pipeline，涵盖了从**预训练**、**有监督微调（SFT）** 到**蒸馏训练**的全过程。

## ✨ 项目特点

-   **从零实现**：基于 PyTorch 原生实现了 Transformer 架构的所有核心组件（如 RMSNorm, RoPE, GQA），不依赖外部模型黑盒。
-   **完整的训练流程**：提供了连贯的 `Pretrain -> SFT -> Distill` 三阶段训练代码，完整复现现代 LLM 的训练范式。
-   **高效训练技巧**：支持**梯度累积**、**混合精度训练**（AMP）、**梯度裁剪**和**余弦学习率调度**，最大化利用硬件资源。
-   **对话功能**：通过 SFT 阶段训练出优秀的对话能力，并提供了流畅的交互式聊天界面。
-   **可扩展的配置**：所有模型超参数通过 `LLMConfig` 类集中管理，方便进行不同规模的实验。
-   **实验跟踪**：集成 Weights & Biases (WandB)，实时监控训练损失、学习率等关键指标。

## 📦 模型架构

SpongeBob 模型是一个基于 Transformer Decoder 的自回归语言模型，其核心设计包括：

-   **旋转位置编码（RoPE）**：为模型注入位置信息，更好地处理长序列。
-   **分组查询注意力（GQA）**：在保持模型性能的同时，提高推理效率，减少 KV 缓存。
-   **RMSNorm**：更高效的归一化层，替代传统的 LayerNorm。
-   **SwiGLU 激活函数**：在前馈网络中使用，提供更强的非线性表达能力。
-   **权重共享**：词嵌入层与输出层共享权重，减少参数量并提升训练稳定性。

## 🗂 项目结构

```
spongebob-llm/
├── dataset.py              # 数据集加载器 (PretrainDataset, SFTDataset)
├── model.py                # 模型核心架构 (SpongeBob, Attention, FeedForward, etc.)
├── Config.py               # 模型配置类 (LLMConfig)
├── pretrain.py             # 预训练脚本
├── SFT.py                  # 有监督微调脚本
├── distill.py              # 蒸馏训练脚本
├── eval_model.py           # 交互式模型推理/聊天脚本
├── results/                # 默认的模型权重保存目录
└── spongebob_tokenizer/    # 分词器目录（需自行准备）
```

## 🚀 快速开始

### 1. 环境安装

建议使用 Python 3.10+ 和 PyTorch 2.0+。

```bash
# 克隆项目
git clone https://github.com/irroca/SpongeBob_Project.git
cd SpongeBob_Project

# 安装依赖（请根据你的CUDA版本安装PyTorch）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers wandb sklearn
```

### 2. 准备数据与分词器

-   **分词器**：将你的 Hugging Face 格式的分词器放在 `./spongebob_tokenizer/` 目录下。
-   **数据**：将预训练数据（每行一个 JSON `{"text": "..."}`）和 SFT 数据（每行一个 JSON `{"conversations": [...]}`）准备好。

### 3. 运行训练

训练分为三个清晰的阶段：

**a. 预训练 (Pretrain)**
```bash
python pretrain.py \
  --data_path "pretrain.jsonl" \
  --batch_size 80 \
  --learning_rate 5e-4 \
  --max_seq_len 512 \
  --save_dir "results"
```

**b. 有监督微调 (SFT)**
```bash
python SFT.py \
  --data_path "sft_512.jsonl" \
  --batch_size 84 \
  --learning_rate 5e-4 \
  --max_seq_len 512 \
  --save_dir "results"
```

**c. 蒸馏训练 (Distill)**
```bash
python distill.py \
  --data_path "r1_1024.jsonl" \
  --batch_size 28 \
  --learning_rate 1e-6 \
  --max_seq_len 1024 \
  --save_dir "results"
```

每个脚本都支持丰富的命令行参数，可通过 `--help` 查看详情。

### 4. 与模型对话

训练完成后，使用以下命令启动一个交互式的聊天程序：

```bash
python eval_model.py \
  --model_mode 1 \          # 0:预训练模型, 1:SFT模型, 2:蒸馏模型
  --save_dir "results" \
  --temperature 0.7 \
  --top_p 0.9
```

## ⚙️ 配置

你可以在 `Config.py` 中的 `LLMConfig` 类里修改模型的所有超参数，例如：

```python
config = LLMConfig(
    dim=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=8, # GQA
    vocab_size=64000,
    max_seq_len=8192,
    dropout=0.1,
    # ... 其他参数
)
```

## 📊 训练监控

本项目集成了 WandB。训练开始后，可以在终端输出的链接中打开 Dashboard，实时查看损失曲线、学习率变化等信息。

## 🧠 核心实现详解

-   **`model.py`**：包含了所有模型的底层实现，是理解本项目的最佳起点。
-   **`dataset.py`**：实现了动态 Loss Masking，尤其在 SFT 阶段只对助理回复部分计算损失，是训练对话模型的关键。
-   **`distill.py`**：通过增加特定 token（如 `<think>`, `</think>`）的损失权重，探索了一种简单的蒸馏或思维链训练方法。

## 🤝 贡献

欢迎任何形式的贡献！无论是提交 Issue、提出新功能建议，还是直接发起 Pull Request，都非常感谢。

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目在实现过程中参考了以下优秀项目：
-   [Llama 2](https://ai.meta.com/llama/)
-   [Transformer](https://arxiv.org/abs/1706.03762) 论文
-   [Hugging Face Transformers](https://github.com/huggingface/transformers)
