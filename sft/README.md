# Mini-GPT 指令微调

从 `checkpoints/tinystories/final_checkpoint_30000.pt` 分别开始 Full SFT、LoRA 和 QLoRA。Checkpoint 的 `iteration` 为 30000，原模型有 4 层、8 个 Q 头、2 个 KV 头，`context_len=256`，词表为 GPT-2 的 50,257 个 token。加载时依据 Checkpoint 配置创建原模型，并严格恢复 `model_state`。

| 方法 | 实现 | 默认可训练参数 |
| --- | --- | ---: |
| Full SFT | AdamW 更新全部预训练参数 | 28,748,544 |
| LoRA | 冻结基础模型，在各层 `w_q`、`w_v` 上学习 `B @ A * (alpha / rank)` | 26,624 |
| QLoRA | Attention 和 SwiGLU 的 28 个权重矩阵存为 NF4，在 `w_q`、`w_v` 上训练同样的 LoRA | 26,624 |

LoRA 使用原投影实际的输入和输出宽度。这里 Q 的形状为 `256 × 256`，V 为 `64 × 256`。默认 rank 8、alpha 16、dropout 0.05。QLoRA 默认 BF16 计算和双重量化，Embedding、RMSNorm 与输出投影保留冻结的 FP32。

## 数据和 Loss

[0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced) 提供英文故事指令。发布版本含 324,984 条 `train` 和 3,542 条 `validation`，各自的 `happy`/`sad` 各占一半。Parquet 字段为 `instruction`、`response`、`ending`，均为字符串。指令包含故事梗概、结局要求和 `### Story:` 标记。数据卡将许可证标记为 `openrail`，并指向上游的 [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)。

- 清理空白和故事标记后，训练与推理使用同一个 `Instruction + "\n\n### Story:\n"` 前缀。故事独立编码，再附加 GPT-2 EOS `50256`。
- 统计包含 EOS 的长度直方图和分位数，丢弃完整序列超过 `context_len` 的样本。右移一次，屏蔽所有 Instruction 标签和右侧 padding，保留故事与 EOS 标签。Loss 和 PPL 按实际参与监督的 token 加权。
- 先处理原始 `validation`，再处理原始 `train`。指令或故事正文在忽略大小写和空白差异后相同，就跳过后出现的候选。固定种子按 `ending` 均衡抽样，从原始 `validation` 生成互斥的验证和测试集。Debug 为 100/20/20，正式为 20,000/1,000/1,000。
- `data/sft/<name>/manifest.json` 记录种子、来源提交、文件哈希、筛选统计和最终划分。三种方法复用同一目录。评估会校验数据文件、划分与基础权重的指纹。划分的去重范围为本次指令数据，预训练语料可能已有相关故事。

## 文件

| 文件 | 用途 |
| --- | --- |
| `__init__.py`、`config.py` | 包入口和默认路径、命令行参数 |
| `prepare_data.py`、`dataset.py` | 下载或读取 Parquet，筛选和划分，统一编码、mask 与动态 padding |
| `model_utils.py` | 恢复原模型、校验权重来源、统计参数、加载四类模型 |
| `trainer.py` | Response Loss、token 加权梯度累积、裁剪、学习率调度、日志和保存 |
| `train_full.py`、`train_lora.py`、`train_qlora.py` | 三个训练入口 |
| `lora.py` | 手写低秩层、Adapter 注入、提取、加载与浮点合并 |
| `quantization.py` | CUDA 检查、原 Linear 和 SwiGLU 的 NF4 适配 |
| `generate.py`、`evaluate.py` | 四种模型的推理和统一评估 |

## 环境

从项目根目录运行。这里使用已有的 Python 3.11 Conda 环境，其中 PyTorch 为 `2.13.0+cu126`。

```powershell
$Py = 'D:\conda_envs\pytorch\python.exe'
& $Py -B -c "import torch, einops, numpy, tiktoken, pyarrow, huggingface_hub; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

上述依赖缺失时，只安装所缺的包。在线数据准备需要 `huggingface_hub`，本地 Parquet 需要 `pyarrow`。首次加载 `tiktoken` 的 GPT-2 词表需联网，可将 `TIKTOKEN_CACHE_DIR` 指向已有缓存。

```powershell
# 按缺少的依赖选择相应包名
& $Py -m pip install einops numpy tiktoken pyarrow huggingface_hub

# 仅 QLoRA 需要，可用新目录隔离依赖
$BnbDir = Join-Path $env:TEMP ('mini-gpt-bnb-' + [guid]::NewGuid().ToString('N'))
& $Py -m pip install --no-deps --target $BnbDir 'bitsandbytes==0.50.2'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$BnbDir;$env:PYTHONPATH" } else { $BnbDir }
```

本项目的 QLoRA 路径要求 NVIDIA CUDA、CUDA 版 PyTorch、Compute Capability 6.0 或以上，以及覆盖当前 CUDA 版本和 GPU 架构的 `bitsandbytes` 构建。默认 BF16 还要求 GPU 原生支持 BF16。Windows x86-64、CUDA 12.6 和 RTX 4060 Laptop（SM 8.9）可使用上述版本组合。 [bitsandbytes 安装文档](https://huggingface.co/docs/bitsandbytes/installation)

## 准备数据

```powershell
$Revision = 'a328d0fa75d06252cfef84a18b07e590d10518ee'
& $Py -B -m sft.prepare_data --profile debug --seed 42 --revision $Revision

# 正式实验的数据准备，输出 data/sft/formal
& $Py -B -m sft.prepare_data --profile formal --seed 42 --revision $Revision
```

下载受限时，可在[数据集文件页](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced/tree/a328d0fa75d06252cfef84a18b07e590d10518ee/data)下载两个分片，然后使用本地入口。

```powershell
& $Py -B -m sft.prepare_data --profile debug --seed 42 --revision $Revision `
  --train-parquet 'D:\downloads\train-00000-of-00001.parquet' `
  --validation-parquet 'D:\downloads\validation-00000-of-00001.parquet'
```

## PowerShell 训练

以下命令用 100 条 Debug 数据各跑 1 轮。默认配置适用于正式实验的初始对比，`--epochs 3 --batch-size 4 --grad-accum 4`，Full 学习率为 `2e-5`，两种 Adapter 为 `2e-4`。使用正式数据时将 `--data-dir` 设为 `data/sft/formal` 并使用新的 `--run-name`。`--max-steps 1` 可只验证一次优化器更新。

```powershell
$Data = 'data/sft/debug'
& $Py -B -m sft.train_full  --data-dir $Data --run-name debug-full  --device cuda --epochs 1
& $Py -B -m sft.train_lora  --data-dir $Data --run-name debug-lora  --device cuda --epochs 1
& $Py -B -m sft.train_qlora --data-dir $Data --run-name debug-qlora --device cuda --epochs 1
```

日志 `metrics.jsonl` 与权重一起保存。`--save-every N` 额外保存每 N 次更新后的推理权重。所有这些权重均面向推理，训练入口从预训练权重开始。

| 方法 | 最终权重 | 内容 |
| --- | --- | --- |
| Full SFT | `checkpoints/sft/full/<run-name>/final.pt` | 模型配置、完整权重和训练指标 |
| LoRA | `checkpoints/sft/lora/<run-name>/final.pt` | LoRA A/B、LoRA 配置和基础权重指纹 |
| QLoRA | `checkpoints/sft/qlora/<run-name>/final.pt` | LoRA A/B、LoRA/NF4 配置和基础权重指纹 |

## PowerShell 推理和评估

```powershell
$StoryPrompt = "Write a story about: a rabbit named Pip who finds a lantern.`nWith: happy ending"
$Full = 'checkpoints/sft/full/debug-full/final.pt'
$Lora = 'checkpoints/sft/lora/debug-lora/final.pt'
$QLora = 'checkpoints/sft/qlora/debug-qlora/final.pt'

& $Py -B -m sft.generate --model base  --prompt $StoryPrompt --max-new-tokens 128
& $Py -B -m sft.generate --model full  --prompt $StoryPrompt --max-new-tokens 128 --checkpoint $Full
& $Py -B -m sft.generate --model lora  --prompt $StoryPrompt --max-new-tokens 128 --checkpoint $Lora
& $Py -B -m sft.generate --model qlora --prompt $StoryPrompt --max-new-tokens 128 --checkpoint $QLora

& $Py -B -m sft.evaluate --data-dir $Data --device cuda --full-checkpoint $Full `
  --lora-checkpoint $Lora --qlora-checkpoint $QLora --seed 42 --max-new-tokens 128 `
  --temperature 0.8 --top-p 0.9 --output-name debug-comparison
```

生成支持 `--temperature 0` 贪心解码、`--no-cache` 无缓存路径、`--merge-lora` 在内存中合并浮点 LoRA。Adapter 恢复使用默认预训练权重，移动权重后可用 `--base-checkpoint` 指定新位置。评估可以用 `--models base full lora` 选择子集。带 `--output-name` 的报告写入 `checkpoints/sft/evaluations/<name>.json`。

## 实验记录模板

记录 GPU、数据 manifest SHA256、预训练 SHA256、更新次数、采样设置和模型版本。训练耗时包括每轮验证，GPU 峰值为 PyTorch 已分配显存，包含模型移至 GPU 后的训练和验证。Base 的训练耗时和峰值留空。

| 模型 | Validation Loss / PPL | Test Loss / PPL | 可训练参数 | 耗时 / 秒 | 峰值 / MiB | 主题 / 角色 / Happy / Sad |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Base |  |  | 0 |  |  |  |
| Full SFT |  |  | 28,748,544 |  |  |  |
| LoRA |  |  | 26,624 |  |  |  |
| QLoRA |  |  | 26,624 |  |  |  |

固定 Prompt 报告按不区分大小写的完整单词匹配主题和角色。结局测试使用同一个找小狗的情节，仅切换结局要求。人工按统一规则记录结果，主角找到小狗或冲突得到正向解决记 `happy`，小狗仍未找到且冲突以失落收尾记 `sad`，冲突未写完或结局含糊记 `unclear`。与目标相同记 1，否则记 0，同时保留原文与 `stop_reason`。

## 已运行的 Debug 验证

真实数据的训练源和验证源分别有 94,278 和 1,013 条超过 256 tokens。按同一提交和种子重复准备 Debug 数据，manifest 和三个划分文件的 SHA-256 完全一致。三种方法各完成 1 轮、7 次优化器更新，验证了 mask、padding、EOS、有限的 Loss 和梯度、权重重载与生成。LoRA 验证了合并前后 logits 一致到 `1e-4` 容差。QLoRA 在 RTX 4060 上验证了 28 个矩阵的 `uint8` 打包、NF4 量化状态和双重量化状态。

| 模型 | Validation Loss / PPL | Test Loss / PPL | 耗时 / 秒 | 峰值 / MiB |
| --- | --- | --- | ---: | ---: |
| Base | 2.3277 / 10.2539 | 2.4209 / 11.2564 | | |
| Full SFT | 2.2934 / 9.9086 | 2.3801 / 10.8057 | 3.07 | 1213.3 |
| LoRA | 2.3243 / 10.2200 | 2.4173 / 11.2152 | 2.08 | 857.3 |
| QLoRA | 2.3638 / 10.6312 | 2.4562 / 11.6601 | 2.19 | 848.3 |

实测权重的 run 名为 `smoke-20261112`，完整评估报告位于 `checkpoints/sft/evaluations/smoke-20261112.json`。固定 Prompt 中 Base、LoRA、QLoRA 各有 3 条在首 token 输出 EOS。正式训练和跨硬件验证尚未运行。

## 常见问题

- `输出目录已存在` 时使用新的 `--name`、`--run-name` 或 `--output-name`。保存采用独占创建。
- `CUDA out of memory` 时降低 `--batch-size` 和 `--eval-batch-size`，再调整 `--grad-accum`。原模型为整个序列计算词表 logits，较长样本会提高峰值。
- QLoRA 出现 DLL 或 CUDA 内核错误时，确认 `torch.version.cuda`、驱动和 wheel 对应的 GPU 架构。若设备不支持 BF16，训练可选 `--compute-dtype float16` 或 `float32`。RTX 4060 上 50,257 维词表输出投影采用 FP32，规避 `bitsandbytes` 的 NF4 反向对齐错误。
- `context_limit` 表示用尽上下文。过长的输入会显式报错，生成总长度最多为 256 tokens。
- 重新准备数据会改变 manifest 时，统一评估应指向三个训练 run 实际使用的数据目录。
