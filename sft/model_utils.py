from pathlib import Path

import torch

from basics.model.transformer import transformer_lm

from .config import BASE_CHECKPOINT, PROMPT_VERSION, TOKENIZER, file_sha256


MODEL_KEYS = ('vocab_size', 'context_len', 'num_layers', 'd_model', 'num_heads',
              'num_kv_heads', 'rope_theta', 'd_ff')


def resolve_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前 PyTorch 未检测到 CUDA，请检查驱动和 CUDA 版 PyTorch')
    return device


def read_base_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'未找到原始预训练权重：{path}')
    # mmap 避免把只用于预训练的 optimizer_state 读入内存。
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    for key in ('model_state', 'config', 'iteration'):
        if key not in checkpoint:
            raise ValueError(f'预训练 Checkpoint 缺少 {key}：{path}')
    for key in MODEL_KEYS:
        if key not in checkpoint['config']:
            raise ValueError(f'预训练模型配置缺少 {key}：{path}')
    return checkpoint


def restore_model(config, state):
    kwargs = {key: config[key] for key in MODEL_KEYS}
    kwargs['context_length'] = kwargs.pop('context_len')
    # 在 meta 上仅创建结构，随后分配独立 CPU 存储并严格复制权重。
    with torch.device('meta'):
        model = transformer_lm(**kwargs)
    model.to_empty(device='cpu')
    model.load_state_dict(state, strict=True)
    # 原项目的 RoPE 频率是非持久 buffer，沿用原实现根据配置重新构造。
    for layer in model.layers:
        rope = layer.attn.pe
        layer.attn.pe = type(rope)(rope.theta, rope.d_k, rope.max_seq_len, device='cpu')
    return model


def load_base_model(path=BASE_CHECKPOINT):
    checkpoint = read_base_checkpoint(path)
    config = {key: checkpoint['config'][key] for key in MODEL_KEYS}
    model = restore_model(config, checkpoint['model_state'])
    identity = {'file': Path(path).name, 'sha256': file_sha256(path), 'iteration': checkpoint['iteration']}
    return model, config, identity


def parameter_counts(model):
    total = sum(getattr(param, 'original_numel', param.numel()) for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {'total_parameters': total, 'trainable_parameters': trainable}


def load_variant(method, checkpoint, base_checkpoint, device):
    if method == 'base':
        model, _, identity = load_base_model(checkpoint or base_checkpoint)
        model.requires_grad_(False)
        return model.to(device).eval(), {'base': identity, **parameter_counts(model)}
    if checkpoint is None:
        raise ValueError(f'{method} 需要提供 --checkpoint')
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1 or bundle.get('method') != method:
        raise ValueError(f'Checkpoint 的格式或类型与 {method} 不匹配')
    if (bundle['tokenizer'], bundle['prompt_version']) != (TOKENIZER, PROMPT_VERSION):
        raise ValueError('Checkpoint 的 Tokenizer 或 Prompt 模板不兼容')

    if method == 'full':
        model = restore_model(bundle['model_config'], bundle['model_state']).to(device)
    else:
        from .lora import inject_lora, load_adapter

        model, config, identity = load_base_model(base_checkpoint)
        if identity['sha256'] != bundle['metrics']['base']['sha256'] or config != bundle['model_config']:
            raise ValueError('Adapter 与指定的预训练权重不匹配，请检查 --base-checkpoint')
        if method == 'qlora':
            from .quantization import quantize_model

            quantize_model(model, bundle['quantization_config'], device)
        else:
            model.to(device)
        inject_lora(model, bundle['lora_config'])
        load_adapter(model, bundle['adapter_state'])
    return model.eval(), bundle['metrics']
