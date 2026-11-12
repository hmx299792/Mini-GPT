import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from .config import (BASE_CHECKPOINT, DATA_ROOT, PROMPT_VERSION, TOKENIZER, directory_name,
                     file_sha256, nonnegative_int, positive_int)
from .dataset import clean_instruction, encode_parts, get_tokenizer
from .model_utils import read_base_checkpoint


DATASET_ID = '0rn0/tinystories-instruct-balanced'
ENDINGS = ('happy', 'sad')
FIELDS = ('instruction', 'response', 'ending')
PRESETS = {'debug': (100, 20, 20), 'formal': (20_000, 1000, 1000)}


def download_parquet(revision, cache_dir):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(DATASET_ID, revision=revision, token=False)
    files = {}
    for split in ('train', 'validation'):
        names = sorted(item.rfilename for item in info.siblings
                       if item.rfilename.endswith('.parquet') and
                       any(part == split or part.startswith((f'{split}-', f'{split}.', f'{split}_'))
                           for part in item.rfilename.split('/')))
        if not names:
            raise ValueError(f'{info.sha} 下没有识别到 {split} Parquet，请用本地文件参数指定分片')
        files[split] = [Path(hf_hub_download(DATASET_ID, name, repo_type='dataset',
                                            revision=info.sha, cache_dir=cache_dir, token=False))
                        for name in names]
    return files, info.sha


def normalized_key(value):
    return ' '.join(value.casefold().split())


def summarize_lengths(lengths):
    if not lengths:
        return {'histogram': {}}
    ordered = sorted(lengths.items())
    total = sum(lengths.values())
    result = {'min': ordered[0][0], 'max': ordered[-1][0], 'histogram': dict(ordered)}
    for percentile in (50, 90, 95, 99):
        rank = math.ceil(percentile * total / 100)
        for length, count in ordered:
            rank -= count
            if rank <= 0:
                result[f'p{percentile}'] = length
                break
    return result


def inspect_split(paths, split, tokenizer, context_len):
    import pyarrow.parquet as pq

    stats, lengths, endings = Counter(), Counter(), Counter()
    records = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = set(FIELDS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f'{path} 缺少字段 {sorted(missing)}')
        for batch in parquet.iter_batches(columns=FIELDS):
            for row in batch.to_pylist():
                index = stats['source_rows']
                stats['source_rows'] += 1
                if not all(isinstance(row[field], str) for field in FIELDS):
                    stats['invalid'] += 1
                    continue
                instruction = clean_instruction(row['instruction'])
                response = row['response'].strip().removesuffix('<|endoftext|>').rstrip()
                ending = row['ending'].strip().lower()
                if (not instruction or not response or ending not in ENDINGS or
                        '<|endoftext|>' in instruction or '<|endoftext|>' in response):
                    stats['invalid'] += 1
                    continue
                endings[ending] += 1
                prompt_ids, response_ids = encode_parts(tokenizer, instruction, response)
                length = len(prompt_ids) + len(response_ids)
                lengths[length] += 1
                if length > context_len:
                    stats['over_context'] += 1
                    continue
                records.append({'id': f'{split}:{index}', 'instruction': instruction,
                                'response': response, 'ending': ending, 'tokens_with_eos': length})
    return records, {'counts': dict(stats), 'ending_before_filter': dict(endings),
                     'eligible_before_dedup': len(records), 'lengths': summarize_lengths(lengths)}


def deduplicate(records, rng, seen_prompts, seen_responses):
    rng.shuffle(records)
    buckets = {ending: [] for ending in ENDINGS}
    for row in records:
        prompt, response = (normalized_key(row[field]) for field in FIELDS[:2])
        if prompt in seen_prompts or response in seen_responses:
            continue
        seen_prompts.add(prompt)
        seen_responses.add(response)
        buckets[row['ending']].append(row)
    return buckets


def select_split(buckets, size, split, rng):
    rows = []
    for index, ending in enumerate(ENDINGS):
        quota = (size + 1 - index) // 2
        if len(buckets[ending]) < quota:
            raise ValueError(f'{split} 的 {ending} 样本需要 {quota} 条，只剩 {len(buckets[ending])} 条')
        rows.extend(buckets[ending][:quota])
        del buckets[ending][:quota]
    rng.shuffle(rows)
    return rows


def prepare(files, output, context_len, sizes, seed, source):
    tokenizer = get_tokenizer()
    rng = random.Random(seed)
    seen_prompts, seen_responses = set(), set()
    stats, splits = {}, {}

    # 原始 validation 优先保留，其去重后的候选也用来排除 train 中的重合项。
    for source_split in ('validation', 'train'):
        records, report = inspect_split(files[source_split], source_split, tokenizer, context_len)
        buckets = deduplicate(records, rng, seen_prompts, seen_responses)
        report['eligible_by_ending_after_dedup'] = {key: len(rows) for key, rows in buckets.items()}
        stats[source_split] = report
        if source_split == 'validation':
            splits['validation'] = select_split(buckets, sizes[1], 'validation', rng)
            splits['test'] = select_split(buckets, sizes[2], 'test', rng)
        else:
            splits['train'] = select_split(buckets, sizes[0], 'train', rng)
    manifest = {'dataset': DATASET_ID, 'source': source, 'license_on_dataset_card': 'openrail',
                'seed': seed, 'tokenizer': TOKENIZER, 'prompt_version': PROMPT_VERSION,
                'context_len': context_len, 'policy': 'filter_whole_story_and_deduplicate_v1',
                'source_stats': stats, 'splits': {}}
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in splits.items():
        path = output / f'{split}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['splits'][split] = {'rows': len(rows), 'sha256': file_sha256(path),
                                     'ending': dict(Counter(row['ending'] for row in rows))}
    with (output / 'manifest.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'output': str(output), 'splits': manifest['splits'],
                      'source_stats': {key: report['counts'] for key, report in stats.items()}}, indent=2))


def prepare_parquet(files, revision, output, context_len, sizes, seed):
    source = {'revision': revision, 'parquet_sha256': {
        split: [file_sha256(path) for path in paths] for split, paths in files.items()}}
    prepare(files, output, context_len, sizes, seed, source)


def main():
    parser = argparse.ArgumentParser(description='准备 TinyStories 指令微调数据')
    parser.add_argument('--profile', choices=PRESETS, default='debug')
    parser.add_argument('--name', type=directory_name, help='data/sft 下的新目录名，默认与 profile 相同')
    parser.add_argument('--seed', type=nonnegative_int, default=42)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--revision', help='Hugging Face 分支或提交 SHA，在线默认 main，本地可用于记录来源')
    parser.add_argument('--train-parquet', type=Path, nargs='+')
    parser.add_argument('--validation-parquet', type=Path, nargs='+')
    for split in ('train', 'validation', 'test'):
        parser.add_argument(f'--{split}-size', type=positive_int)
    args = parser.parse_args()
    if bool(args.train_parquet) != bool(args.validation_parquet):
        parser.error('使用本地数据时必须同时提供 --train-parquet 和 --validation-parquet')
    output = DATA_ROOT / (args.name or args.profile)
    if output.exists():
        parser.error(f'输出目录已存在：{output}。请使用不同的 --name')
    sizes = tuple(getattr(args, f'{split}_size') or default
                  for split, default in zip(('train', 'validation', 'test'), PRESETS[args.profile]))
    checkpoint = read_base_checkpoint(args.base_checkpoint)
    context_len = checkpoint['config']['context_len']
    del checkpoint
    if args.train_parquet:
        files = {'train': args.train_parquet, 'validation': args.validation_parquet}
        prepare_parquet(files, args.revision, output, context_len, sizes, args.seed)
        return

    with TemporaryDirectory(prefix='mini-gpt-sft-', ignore_cleanup_errors=True) as cache_dir:
        try:
            files, revision = download_parquet(args.revision or 'main', cache_dir)
        except Exception as exc:
            raise SystemExit(f'下载数据失败：{exc}。可手动下载 Parquet 后用本地文件参数准备数据。') from exc
        prepare_parquet(files, revision, output, context_len, sizes, args.seed)


if __name__ == '__main__':
    main()
