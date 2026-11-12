import argparse
import json
import re
from pathlib import Path

from .config import BASE_CHECKPOINT, CHECKPOINT_ROOT, DATA_ROOT, METHODS, directory_name, positive_int
from .dataset import get_tokenizer, make_loader, read_manifest
from .generate import add_sampling_arguments, generate, sampling_options
from .model_utils import load_variant, resolve_device
from .trainer import evaluate_loss


MODELS = ('base', *METHODS)
ENDING_TOPIC = 'Write a story about: Mina tries to find her lost puppy before dark.'
FIXED_PROMPTS = (
    ('topic', 'Write a story about: a storm and a lighthouse.\nWith: happy ending',
     ('storm', 'lighthouse'), 'happy'),
    ('characters', 'Write a story about: a rabbit named Pip and a turtle named Nora.\nWith: happy ending',
     ('Pip', 'Nora'), 'happy'),
    ('happy_ending', f'{ENDING_TOPIC}\nWith: happy ending', ('Mina', 'puppy'), 'happy'),
    ('sad_ending', f'{ENDING_TOPIC}\nWith: sad ending', ('Mina', 'puppy'), 'sad'),
)


def fixed_prompt_results(model, tokenizer, device, sampling):
    results = []
    for name, instruction, keywords, ending in FIXED_PROMPTS:
        result = generate(model, tokenizer, instruction, device, **sampling)
        matches = {keyword: bool(re.search(rf'\b{re.escape(keyword)}\b', result['text'], re.I))
                   for keyword in keywords}
        results.append({'case': name, 'instruction': instruction, **result, 'keyword_matches': matches,
                        'all_keywords_present': all(matches.values()), 'expected_ending': ending})
    return results


def main():
    parser = argparse.ArgumentParser(description='评估 Base、Full SFT、LoRA、QLoRA 的 Response Loss/PPL 和指令生成')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    for method in METHODS:
        parser.add_argument(f'--{method}-checkpoint', type=Path)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT / 'debug')
    parser.add_argument('--batch-size', type=positive_int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-name', type=directory_name, help='可选 JSON 报告文件名，无需 .json')
    add_sampling_arguments(parser)
    args = parser.parse_args()
    sampling = sampling_options(parser, args)
    if len(args.models) != len(set(args.models)):
        parser.error('--models 不能重复')
    for method in args.models:
        if method != 'base' and getattr(args, f'{method}_checkpoint') is None:
            parser.error(f'缺少 --{method}-checkpoint')
    output = CHECKPOINT_ROOT / 'evaluations' / f'{args.output_name}.json' if args.output_name else None
    if output and output.exists():
        parser.error(f'报告已存在：{output}，请使用不同的 --output-name')

    device = resolve_device(args.device)
    tokenizer = get_tokenizer()
    results = {}
    loaders, report, base_sha = None, None, None
    for method in args.models:
        checkpoint = None if method == 'base' else getattr(args, f'{method}_checkpoint')
        model, metrics = load_variant(method, checkpoint, args.base_checkpoint, device)
        manifest, data_sha = read_manifest(args.data_dir, model.context_length)
        if model.vocab_size != tokenizer.n_vocab:
            raise ValueError(f'{method} 的词表大小与 GPT-2 Tokenizer 不匹配')
        if method != 'base' and metrics['dataset_manifest_sha256'] != data_sha:
            raise ValueError(f'{method} 使用的数据划分与 --data-dir 不一致')
        if base_sha and base_sha != metrics['base']['sha256']:
            raise ValueError('参与评估的模型来自不同的预训练权重')
        base_sha = metrics['base']['sha256']
        if loaders is None:
            loaders = {split: make_loader(args.data_dir, manifest, split, tokenizer, args.batch_size)
                       for split in ('validation', 'test')}
            report = {'base_sha256': base_sha, 'dataset_manifest_sha256': data_sha, 'device': str(device),
                      'sampling': sampling, 'models': results}
        elif data_sha != report['dataset_manifest_sha256']:
            raise ValueError('评估期间数据 manifest 发生变化')

        results[method] = {'checkpoint': str(checkpoint or args.base_checkpoint),
                           'total_parameters': metrics['total_parameters'],
                           'trainable_parameters': metrics['trainable_parameters'],
                           'training_seconds': metrics.get('training_seconds'),
                           'peak_gpu_allocated_bytes': metrics.get('peak_gpu_allocated_bytes'),
                           'training_steps': metrics.get('optimizer_steps'),
                           **{split: evaluate_loss(model, loader, device) for split, loader in loaders.items()},
                           'generations': fixed_prompt_results(model, tokenizer, device, sampling)}
        del model

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as handle:
            handle.write(payload + '\n')
    print(payload)


if __name__ == '__main__':
    main()
