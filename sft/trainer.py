import json
import math
import sys
import time

import torch
from torch.nn import functional as F
from tqdm import tqdm

from basics.trainer.utils import learning_rate_schedule

from .config import CHECKPOINT_ROOT, PROMPT_VERSION, TOKENIZER, training_parser, validate_training_args
from .dataset import IGNORE_INDEX, get_tokenizer, make_loader, read_manifest
from .lora import adapter_state, inject_lora
from .model_utils import load_base_model, parameter_counts, resolve_device


def response_nll(model, batch, device):
    inputs = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    logits = model(inputs)
    loss = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(),
                           ignore_index=IGNORE_INDEX, reduction='sum')
    count = int(batch['labels'].ne(IGNORE_INDEX).sum())
    return loss, count


def summarize_nll(nll, tokens):
    if tokens <= 0 or not math.isfinite(nll):
        raise FloatingPointError('Response token 计数为空，或 Loss 为 NaN/Inf')
    loss = nll / tokens
    return {'loss': loss, 'ppl': math.exp(loss) if loss < 709 else math.inf, 'response_tokens': tokens}


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    nll, tokens = 0.0, 0
    for batch in loader:
        loss, count = response_nll(model, batch, device)
        nll += loss.item()
        tokens += count
    return summarize_nll(nll, tokens)


def update_lr(step, total_steps, args):
    if total_steps == 1:
        return args.lr
    warmup = min(args.warmup_steps, total_steps - 1)
    if warmup:
        return learning_rate_schedule(step, args.lr, args.lr * args.min_lr_ratio, warmup, total_steps)
    return learning_rate_schedule(step - 1, args.lr, args.lr * args.min_lr_ratio, 0, total_steps - 1)


def save_weights(path, model, method, config, lora, quantization, metrics):
    bundle = {'format_version': 1, 'method': method, 'model_config': config, 'metrics': metrics,
              'tokenizer': TOKENIZER, 'prompt_version': PROMPT_VERSION}
    if method == 'full':
        bundle['model_state'] = model.state_dict()
    else:
        bundle.update(adapter_state=adapter_state(model), lora_config=lora)
        if method == 'qlora':
            bundle['quantization_config'] = quantization
    with path.open('xb') as handle:
        torch.save(bundle, handle)


def train(model, method, config, args, device, base, output, lora, quantization, bnb_version):
    tokenizer = get_tokenizer()
    if tokenizer.n_vocab != config['vocab_size']:
        raise ValueError('Checkpoint 词表大小与 GPT-2 Tokenizer 不匹配')
    manifest, manifest_sha = read_manifest(args.data_dir, config['context_len'])
    training = make_loader(args.data_dir, manifest, 'train', tokenizer, args.batch_size,
                           shuffle=True, seed=args.seed)
    validation = make_loader(args.data_dir, manifest, 'validation', tokenizer, args.eval_batch_size)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(args.beta1, args.beta2), eps=args.eps)
    total_steps = math.ceil(len(training) / args.grad_accum) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    output.mkdir(parents=True, exist_ok=False)
    counts = parameter_counts(model)
    step = 0
    last_validation = None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    def metrics():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return {**counts, 'base': base, 'device': str(device), 'torch_version': str(torch.__version__),
                'cuda_version': torch.version.cuda, 'bitsandbytes_version': bnb_version,
                'dataset_manifest_sha256': manifest_sha, 'optimizer_steps': step,
                'training_seconds': time.perf_counter() - started,
                'peak_gpu_allocated_bytes': (torch.cuda.max_memory_allocated(device)
                                             if device.type == 'cuda' else None),
                'validation': last_validation,
                'settings': {key: str(value) if hasattr(value, '__fspath__') else value
                             for key, value in vars(args).items()}}

    description = f'{method.upper()} SFT'
    with (
        (output / 'metrics.jsonl').open('x', encoding='utf-8') as log,
        tqdm(total=total_steps, desc=description, unit='step', dynamic_ncols=True, disable=None) as progress,
    ):
        def record(event):
            line = json.dumps(event, ensure_ascii=False)
            log.write(line + '\n')
            log.flush()
            with tqdm.external_write_mode(file=sys.stdout):
                print(line, flush=True)

        record({'event': 'start', 'method': method, **counts, 'planned_steps': total_steps,
                'context_len': config['context_len'], 'base': base, 'dataset_manifest_sha256': manifest_sha})
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_nll, group_tokens = 0.0, 0
            for micro_step, batch in enumerate(training, 1):
                loss, count = response_nll(model, batch, device)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise FloatingPointError(f'Loss 为 NaN/Inf，epoch={epoch}, micro_step={micro_step}')
                loss.backward()
                group_nll += loss_value
                group_tokens += count
                if micro_step % args.grad_accum and micro_step != len(training):
                    continue

                # 微批次故事长度不同。按整组实际监督 token 数归一化梯度。
                for param in trainable:
                    if param.grad is not None:
                        param.grad.div_(group_tokens)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm, error_if_nonfinite=True)
                step += 1
                lr = update_lr(step, total_steps, args)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step == 1 or step % args.log_every == 0 or step == total_steps:
                    update_metrics = summarize_nll(group_nll, group_tokens)
                    progress.set_postfix(epoch=f'{epoch}/{args.epochs}', loss=f'{update_metrics["loss"]:.4f}',
                                         lr=f'{lr:.2e}', refresh=False)
                    record({'event': 'update', 'epoch': epoch, 'step': step, 'lr': lr,
                            'gradient_norm': grad_norm.item(), **update_metrics})
                progress.update()
                group_nll, group_tokens = 0.0, 0
                if args.save_every and step % args.save_every == 0 and step < total_steps:
                    save_weights(output / f'step_{step:06d}.pt', model, method, config, lora, quantization, metrics())
                if step == total_steps:
                    break

            progress.set_description(f'{description} (validation)')
            last_validation = {'step': step, **evaluate_loss(model, validation, device)}
            progress.set_description(description, refresh=False)
            record({'event': 'validation', 'epoch': epoch, **last_validation})
            if step == total_steps:
                break

        progress.set_description(f'{description} (saving)')
        summary = metrics()
        save_weights(output / 'final.pt', model, method, config, lora, quantization, summary)
        progress.set_description(description, refresh=False)
        record({'event': 'complete', 'checkpoint': str(output / 'final.pt'), **summary})


def train_cli(method):
    parser = training_parser(method)
    args = parser.parse_args()
    validate_training_args(parser, args)
    output = CHECKPOINT_ROOT / method / args.run_name
    if output.exists():
        parser.error(f'输出目录已存在：{output}。请使用不同的 --run-name')
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model, config, base = load_base_model(args.base_checkpoint)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    lora, quantization, bnb_version = None, None, None
    if method == 'qlora':
        from .quantization import quantize_model

        quantization = {'quant_type': 'nf4', 'scope': 'attention_swiglu',
                        'compute_dtype': args.compute_dtype, 'double_quant': args.double_quant}
        bnb_version = quantize_model(model, quantization, device)
    else:
        model.to(device)
    if method != 'full':
        lora = {key: getattr(args, key) for key in ('rank', 'alpha', 'dropout', 'targets')}
        inject_lora(model, lora)
    train(model, method, config, args, device, base, output, lora, quantization, bnb_version)
