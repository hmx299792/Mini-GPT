import argparse
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CHECKPOINT = ROOT / 'checkpoints' / 'tinystories' / 'final_checkpoint_30000.pt'
DATA_ROOT = ROOT / 'data' / 'sft'
CHECKPOINT_ROOT = ROOT / 'checkpoints' / 'sft'
METHODS = ('full', 'lora', 'qlora')
TOKENIZER = 'gpt2'
PROMPT_VERSION = 'story-v1'


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('需要正整数')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('需要非负整数')
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def directory_name(value):
    if not value.isascii() or not value or not all(c.isalnum() or c in '-_' for c in value):
        raise argparse.ArgumentTypeError('目录名只能使用英文字母、数字、横线和下划线')
    return value


def training_parser(method):
    parser = argparse.ArgumentParser(description=f'Mini-GPT {method.upper()} SFT')
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT / 'debug')
    parser.add_argument('--run-name', required=True, type=directory_name)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=nonnegative_int, default=42)
    parser.add_argument('--epochs', type=positive_int, default=3)
    parser.add_argument('--batch-size', type=positive_int, default=4)
    parser.add_argument('--eval-batch-size', type=positive_int, default=4)
    parser.add_argument('--grad-accum', type=positive_int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5 if method == 'full' else 2e-4)
    parser.add_argument('--min-lr-ratio', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=nonnegative_int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--log-every', type=positive_int, default=10)
    parser.add_argument('--save-every', type=nonnegative_int, default=0, help='按更新步数保存推理权重，0 表示仅保存最终权重')
    parser.add_argument('--max-steps', type=positive_int, help='限制优化器更新次数')
    if method in ('lora', 'qlora'):
        parser.add_argument('--rank', type=positive_int, default=8)
        parser.add_argument('--alpha', type=float, default=16.0)
        parser.add_argument('--dropout', type=float, default=0.05)
        parser.add_argument('--targets', nargs='+', choices=('w_q', 'w_k', 'w_v', 'w_o'), default=['w_q', 'w_v'])
    if method == 'qlora':
        parser.add_argument('--compute-dtype', choices=('float32', 'float16', 'bfloat16'), default='bfloat16')
        parser.add_argument('--double-quant', action=argparse.BooleanOptionalAction, default=True)
    return parser


def validate_training_args(parser, args):
    for name in ('lr', 'max_grad_norm', 'eps'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name.replace("_", "-")} 必须为有限正数')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('--weight-decay 必须为有限非负数')
    if not (0 <= args.min_lr_ratio <= 1 and 0 <= args.beta1 < 1 and 0 <= args.beta2 < 1):
        parser.error('min-lr-ratio 取值为 [0, 1]，beta1 和 beta2 为 [0, 1)')
