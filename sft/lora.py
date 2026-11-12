import math

import torch
from torch import nn
from torch.nn import functional as F

from basics.model.modules import Linear


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        original = self.base(inputs)
        # B 从零开始，因此初始输出和基础层相同。A/B 使用 FP32。
        update = F.linear(F.linear(self.dropout(inputs).float(), self.lora_a), self.lora_b)
        return original + (update * self.scaling).to(original.dtype)


def inject_lora(model, config):
    rank, alpha, dropout, targets = (config[key] for key in ('rank', 'alpha', 'dropout', 'targets'))
    if (rank <= 0 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1 or
            not targets or len(set(targets)) != len(targets) or set(targets) - {'w_q', 'w_k', 'w_v', 'w_o'}):
        raise ValueError('LoRA 配置无效，请检查 rank、alpha、dropout 和 targets')
    model.requires_grad_(False)
    for block in model.layers:
        for name in targets:
            base = getattr(block.attn, name)
            if isinstance(base, LoRALinear):
                raise ValueError('模型已注入 LoRA')
            # 直接读取每层的实际 in/out_features，GQA 下 V 比 Q 窄。
            setattr(block.attn, name, LoRALinear(base, rank, alpha, dropout))


def adapter_parameters(model):
    return {f'{name}.{key}': getattr(module, key) for name, module in model.named_modules()
            if isinstance(module, LoRALinear) for key in ('lora_a', 'lora_b')}


def adapter_state(model):
    return {name: param.detach().cpu().clone() for name, param in adapter_parameters(model).items()}


@torch.no_grad()
def load_adapter(model, state):
    expected = adapter_parameters(model)
    if not expected or state.keys() != expected.keys():
        raise ValueError('Adapter 的参数名与注入的 LoRA 层不匹配')
    for name, param in expected.items():
        if state[name].shape != param.shape or state[name].dtype != param.dtype:
            raise ValueError(f'Adapter 参数形状或精度不匹配：{name}')
    for name, param in expected.items():
        param.copy_(state[name])


@torch.no_grad()
def merge_lora(model):
    """推理时将 FP32 LoRA 增量合并进内存中的基础 Linear。"""
    if model.training:
        raise ValueError('合并前请调用 model.eval()')
    adapters = [(parent, name, child) for parent in model.modules() for name, child in parent.named_children()
                if isinstance(child, LoRALinear)]
    if any(not isinstance(adapter.base, Linear) for _, _, adapter in adapters):
        raise ValueError('NF4 Adapter 需重新加载到浮点基础层后才能合并')
    for parent, name, adapter in adapters:
        delta = (adapter.lora_b @ adapter.lora_a) * adapter.scaling
        adapter.base.weight.add_(delta.to(adapter.base.weight.dtype))
        setattr(parent, name, adapter.base)
    return model
