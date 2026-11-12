import json
from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import PROMPT_VERSION, TOKENIZER, file_sha256


IGNORE_INDEX = -100
STORY_HEADER = '### Story:'


def get_tokenizer():
    return tiktoken.get_encoding(TOKENIZER)


def clean_instruction(instruction):
    return instruction.strip().removesuffix(STORY_HEADER).rstrip()


def format_prompt(instruction):
    instruction = clean_instruction(instruction)
    if not instruction:
        raise ValueError('Instruction 不能为空')
    return f'{instruction}\n\n{STORY_HEADER}\n'


def encode_parts(tokenizer, instruction, response):
    # 分开编码，确保推理前缀与训练前缀的 token 完全一致。
    prompt_ids = tokenizer.encode_ordinary(format_prompt(instruction))
    response_ids = tokenizer.encode_ordinary(response.strip())
    if not response_ids:
        raise ValueError('Response 不能为空')
    return prompt_ids, response_ids + [tokenizer.eot_token]


def encode_example(tokenizer, row, context_len):
    prompt, response = encode_parts(tokenizer, row['instruction'], row['response'])
    sequence = prompt + response
    if len(sequence) > context_len:
        raise ValueError(f"样本 {row.get('id', '')} 长度 {len(sequence)} 超过 context_len={context_len}")
    # logits[i] 预测下一个 token。最后一个 prompt 位置监督第一个故事 token。
    return torch.tensor(sequence[:-1]), torch.tensor([IGNORE_INDEX] * (len(prompt) - 1) + response)


class StoryDataset(Dataset):
    def __init__(self, path, tokenizer, context_len):
        with path.open(encoding='utf-8') as handle:
            self.examples = [encode_example(tokenizer, json.loads(line), context_len) for line in handle]
        if not self.examples:
            raise ValueError(f'空数据划分 {path}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(examples, pad_id):
    inputs, labels = zip(*examples)
    # 右侧 padding 不会被前面的有效位置看到，原模型的 causal mask 即可。
    return {
        'input_ids': pad_sequence(inputs, batch_first=True, padding_value=pad_id),
        'labels': pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    }


def read_manifest(directory, context_len):
    manifest_path = Path(directory) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['tokenizer'], manifest['prompt_version']) != (TOKENIZER, PROMPT_VERSION):
        raise ValueError('数据使用了不兼容的 Tokenizer 或 Prompt 模板')
    if manifest['context_len'] > context_len:
        raise ValueError(f'数据准备时的 context_len 超过当前模型的 {context_len}')
    return manifest, file_sha256(manifest_path)


def make_loader(directory, manifest, split, tokenizer, batch_size, shuffle=False, seed=42):
    path = Path(directory) / f'{split}.jsonl'
    expected = manifest['splits'][split]
    if file_sha256(path) != expected['sha256']:
        raise ValueError(f'数据文件与 manifest 不一致：{path}')
    dataset = StoryDataset(path, tokenizer, manifest['context_len'])
    if len(dataset) != expected['rows']:
        raise ValueError(f'样本数量与 manifest 不一致：{path}')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=partial(collate, pad_id=tokenizer.eot_token))
