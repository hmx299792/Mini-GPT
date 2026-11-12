import platform

import torch
from torch import nn

from basics.model.modules import Linear, SwiGLU


DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}


class QuantizedSwiGLU(nn.Module):
    """适配原 SwiGLU 的三个裸 Parameter，保留其门控运算。"""

    def __init__(self, w1, w3, w2):
        super().__init__()
        self.w1 = w1
        self.w3 = w3
        self.w2 = w2

    def forward(self, inputs):
        gate = self.w1(inputs)
        return self.w2((gate * torch.sigmoid(gate)) * self.w3(inputs))


def check_backend(device, compute_dtype):
    if device.type != 'cuda' or not torch.version.cuda or not torch.cuda.is_available():
        raise RuntimeError('QLoRA 需要 NVIDIA CUDA 和 CUDA 版 PyTorch，请使用 --device cuda')
    if torch.cuda.get_device_capability(device) < (6, 0):
        raise RuntimeError('NF4 要求 GPU Compute Capability 至少为 6.0')
    with torch.cuda.device(device):
        if compute_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('GPU 不支持原生 BF16，请使用 --compute-dtype float16 或 float32')
    try:
        import bitsandbytes as bnb
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(f'无法导入 bitsandbytes，当前为 {platform.system()}、'
                           f'PyTorch {torch.__version__}、CUDA {torch.version.cuda}。'
                           '请参考 sft/README.md 安装兼容的构建。') from exc
    return bnb


def make_nf4_linear(weight, bnb, device, dtype, double_quant):
    out_features, in_features = weight.shape
    layer = bnb.nn.Linear4bit(in_features, out_features, bias=False, compute_dtype=dtype,
                             compress_statistics=double_quant, quant_type='nf4',
                             quant_storage=torch.uint8)
    layer.weight = bnb.nn.Params4bit(weight.detach().contiguous(), requires_grad=False,
                                    compress_statistics=double_quant, quant_type='nf4',
                                    quant_storage=torch.uint8, module=layer)
    # CPU 上装载的是预训练矩阵，首次迁移到 CUDA 时由 bitsandbytes 打包。
    layer.to(device)
    packed = layer.weight
    state = packed.quant_state
    if (not packed.bnb_quantized or state is None or state.quant_type != 'nf4' or
            packed.dtype != torch.uint8 or packed.numel() != (weight.numel() + 1) // 2 or
            (state.state2 is not None) != double_quant or packed.requires_grad):
        raise RuntimeError('检测到未正确打包的 NF4 权重')
    packed.original_numel = weight.numel()
    return layer


def quantize_model(model, config, device):
    if config['quant_type'] != 'nf4' or config['scope'] != 'attention_swiglu':
        raise ValueError('不支持此 Checkpoint 的量化格式')
    dtype = DTYPES[config['compute_dtype']]
    bnb = check_backend(device, dtype)
    model.requires_grad_(False)

    def convert(weight):
        return make_nf4_linear(weight, bnb, device, dtype, config['double_quant'])

    try:
        for block in model.layers:
            for name in ('w_q', 'w_k', 'w_v', 'w_o'):
                layer = getattr(block.attn, name)
                if not isinstance(layer, Linear):
                    raise ValueError('请先量化基础 Linear，再注入 LoRA')
                setattr(block.attn, name, convert(layer.weight))
            ffn = block.ffn
            if not isinstance(ffn, SwiGLU):
                raise ValueError('需要原始 SwiGLU 才能转换量化矩阵')
            block.ffn = QuantizedSwiGLU(convert(ffn.w1), convert(ffn.w3), convert(ffn.w2))
            block.rms1.to(device)
            block.rms2.to(device)
            block.attn.pe.to(device)
    except (RuntimeError, OSError, NotImplementedError) as exc:
        raise RuntimeError(f'bitsandbytes {bnb.__version__} 的 CUDA NF4 量化失败，'
                           f'PyTorch CUDA {torch.version.cuda}，设备 {torch.cuda.get_device_name(device)}') from exc
    model.embedding.to(device)
    model.output_norm.to(device)
    # GPT-2 词表的 50,257 个输出会触发此 CUDA 构建的 4-bit 反向对齐错误。
    model.project.to(device)
    return bnb.__version__
