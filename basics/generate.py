import torch
import tiktoken
import numpy as np
import os
import sys
from typing import List, Optional, Union
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#将 D:\mini_gpt 加入模块搜索路径

from basics.model.transformer import transformer_lm
from basics.check_points import load_checkpoint
from basics.model.modules import softmax

def softmax_with_temperature(logits: torch.Tensor, temperature: float=0.8) -> torch.Tensor:
    scaled_logits = logits / temperature
    prob = softmax(scaled_logits, dim=-1)
    return prob

def top_p_sampling(probabilities: torch.Tensor, p: float=0.9):
    sorted_probs, sorted_indicies = torch.sort(probabilities, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) #torch.cumsum()是cumulative sum（累积求和）

    mask = (cumulative_probs - sorted_probs) < p
    filtered_probs = sorted_probs * mask
    filtered_probs = filtered_probs / torch.sum(filtered_probs, dim=-1, keepdim=True)

    #把顺序改回去
    output_probs = torch.zeros_like(filtered_probs)
    output_probs.scatter_(-1, index=sorted_indicies, src=filtered_probs)

    return output_probs

def generate_text(
        model: torch.nn.Module,
        tokenizer,
        prompt: str,
        max_tokens: int=256,
        temperature: float=0.8,
        top_p: float=0.9,
        device: str='cuda',
        eos_token: str='<|endoftext|>',
        use_cache: bool=True
) -> str:
    model.to(device)
    model.eval()

    #提示词编码为tokenID
    prompt_tokens = tokenizer.encode(prompt) #prompt_tokens是一个Python列表（list[int]）
    inputs = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0) #增加一个Batch维度，方便输入Transformer模型
    generate_tokens = prompt_tokens.copy() #先复制一份Prompt的Token ID列表，用来保存后续生成的所有Token

    #每次生成文本时重新创建KV Cache
    past_kv = None

    with torch.no_grad():
        for _ in range(max_tokens):
            #前向传播
            if use_cache:
                #第一次：Prefill，输入完整Prompt
                #之后：Decode，只输入新生成的Token
                logits, past_kv = model(
                    inputs,
                    past_kv=past_kv,
                    use_cache=True
                )
            else:
                #不使用KV Cache：每次重新计算完整输入
                logits = model(inputs)

            next_token_logits = logits[0, -1, :]
            probabilities = softmax_with_temperature(next_token_logits, temperature)

            if top_p < 1.0:
                probabilities = top_p_sampling(probabilities=probabilities, p=top_p)

            #在top-p中随机采样一个token
            next_token = torch.multinomial(probabilities, num_samples=1).item() #torch.multinomial抽取的是概率张量中的索引（index）即tokenID

            #序列拼接新token
            generate_tokens.append(next_token)

            # #检查新token是不是结束标志token
            # if eos_token is not None:
            #     #将新生成的 Token ID 解码成字符串str
            #     decode_new_token = tokenizer.decode([next_token]) #列表[]中只有一个next_token的ID
            #     if decode_new_token.strip() == eos_token.strip(): #.strip()：去除字符串首尾的空格、换行符\n、制表符\t等
            #         break

            #或：
            if eos_token is not None:
                if next_token == tokenizer.eot_token:
                    break

            #已经生成完最后一个Token，无需再次前向传播
            if len(generate_tokens) - len(prompt_tokens) >= max_tokens:
                break

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device) #[[next_token]]创建二维张量

            if use_cache:
                if past_kv[0][0].size(-2) >= model.context_length: #past_kv[0][0]：取第 0 层缓存的 K 张量
                    break
                inputs = next_token_tensor
            else:
                #token张量拼接新token
                inputs = torch.cat([inputs, next_token_tensor], dim=1)

            #句子太长就截断
            if inputs.size(1) > model.context_length:
                inputs = inputs[:, -model.context_length:] #第一个冒号：所有B   第二个冒号：一直到结尾

    generate_text = tokenizer.decode(generate_tokens)

    return generate_text

def load_model_and_tokenizer(checkpoint_path: str, device: str='cuda'):
    tokenizer = tiktoken.get_encoding('gpt2')

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device #指定张量加载到哪个设备
    )

    config = checkpoint['config']

    # 提取模型结构参数
    model_config = {
        'vocab_size': config['vocab_size'],
        'context_length': config['context_len'],
        'num_layers': config['num_layers'],
        'd_model': config['d_model'],
        'num_heads': config['num_heads'],
        'num_kv_heads': config['num_kv_heads'],
        'rope_theta': config['rope_theta'],
        'd_ff': config['d_ff']
    }

    # 初始化模型
    model = transformer_lm(**model_config).to(device) # **model_config表示字典解包，把字典中的键值对转换成关键字参数

    model.load_state_dict(checkpoint['model_state'])

    model.eval()

    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description='用训练好的Transformer语言模型生成文本')

    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint检查点路径')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='输入的提示词')
    parser.add_argument('--max_tokens', type=int, default=256, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature(低温代表更具确定性)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p阈值')
    parser.add_argument('--n_samples', type=int, default=1, help='生成文本条数')
    parser.add_argument('--device', type=str, default='cuda', help='设备: auto, cpu, cuda, mps')
    parser.add_argument('--eos_token', type=str, default='<|endoftext|>', help='句子结束token')
    #KV Cache开关
    parser.add_argument('--no_cache', action='store_true', help='禁用KV Cache，使用普通自回归推理')
    
    args = parser.parse_args()


    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device


    #加载模型和分词器
    print('正在加载模型和分词器...')
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            device=device
        )
        print('模型和分词器加载成功！')
    except Exception as e:
        print(f"加载模型或分词器时发生错误: {e}")
        return

    print(f'模型共有{sum(p.numel() for p in model.parameters())}个参数')


    #生成文本
    print(f'\n生成{args.n_samples}条文本')
    print(f"最大token数={args.max_tokens}, 温度={args.temperature}, Top_p={args.top_p}")
    print("-" * 80)

    for i in range(args.n_samples):
        if args.n_samples > 1:
            print(f"\n第{i+1}条文本：")
            print("-" * 40)
        try:
            gene_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                eos_token=args.eos_token,
                use_cache=not args.no_cache #返回True
            )
            print(gene_text)
        except Exception as e:
            print(f'生成时出现错误{e}')

if __name__ == '__main__':
    main()