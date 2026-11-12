import argparse
import math
from pathlib import Path

import torch

from basics.generate import softmax_with_temperature, top_p_sampling

from .config import BASE_CHECKPOINT, METHODS, nonnegative_int, positive_int
from .dataset import format_prompt, get_tokenizer
from .lora import merge_lora
from .model_utils import load_variant, resolve_device


def add_sampling_arguments(parser):
    parser.add_argument('--max-new-tokens', type=positive_int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 表示贪心解码')
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=nonnegative_int, default=42)
    parser.add_argument('--no-cache', action='store_true')


def sampling_options(parser, args):
    if not math.isfinite(args.temperature) or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error('temperature 需为有限非负数，top-p 取值为 (0, 1]')
    return {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature,
            'top_p': args.top_p, 'seed': args.seed, 'use_cache': not args.no_cache}


@torch.no_grad()
def generate(model, tokenizer, instruction, device, *, max_new_tokens, temperature, top_p, seed, use_cache):
    prompt = tokenizer.encode_ordinary(format_prompt(instruction))
    budget = min(max_new_tokens, model.context_length - len(prompt))
    if budget <= 0:
        raise ValueError(f'Prompt 为 {len(prompt)} tokens，已用完上下文容量 {model.context_length}')
    model.eval()
    inputs = torch.tensor([prompt], device=device)
    # CPU 采样 RNG 独立于模型加载的随机数及 CUDA RNG，四种方法使用同一套参数。
    generator = torch.Generator().manual_seed(seed)
    generated, cache = [], None
    reason = 'context_limit' if budget < max_new_tokens else 'max_new_tokens'
    for _ in range(budget):
        if use_cache:
            logits, cache = model(inputs, past_kv=cache, use_cache=True)
        else:
            logits = model(inputs)
        last = logits[0, -1].cpu()
        if temperature == 0:
            token = int(last.argmax())
        else:
            scores = last.double() - last.max()
            probs = softmax_with_temperature(scores, temperature)
            if top_p < 1:
                probs = top_p_sampling(probs, top_p)
            token = int(torch.multinomial(probs, 1, generator=generator))
        if token == tokenizer.eot_token:
            reason = 'eos'
            break
        generated.append(token)
        new_input = torch.tensor([[token]], device=device)
        inputs = new_input if use_cache else torch.cat((inputs, new_input), dim=1)
    return {'text': tokenizer.decode(generated), 'generated_tokens': len(generated), 'stop_reason': reason}


def main():
    parser = argparse.ArgumentParser(description='Mini-GPT 统一指令生成')
    parser.add_argument('--model', choices=('base', *METHODS), default='base')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--prompt', required=True, help='英文故事指令')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--merge-lora', action='store_true', help='推理前在内存中合并浮点 LoRA')
    add_sampling_arguments(parser)
    args = parser.parse_args()
    options = sampling_options(parser, args)
    if args.merge_lora and args.model != 'lora':
        parser.error('--merge-lora 仅适用于 --model lora')
    if args.model != 'base' and args.checkpoint is None:
        parser.error(f'{args.model} 需要 --checkpoint')
    device = resolve_device(args.device)
    model, _ = load_variant(args.model, args.checkpoint, args.base_checkpoint, device)
    tokenizer = get_tokenizer()
    if tokenizer.n_vocab != model.vocab_size:
        raise ValueError('Checkpoint 词表大小与 GPT-2 Tokenizer 不匹配')
    if args.merge_lora:
        merge_lora(model)
    result = generate(model, tokenizer, args.prompt, device, **options)
    print(result['text'])


if __name__ == '__main__':
    main()
