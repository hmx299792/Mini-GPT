import tiktoken
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import pathlib
import numpy as np


#从当前脚本所在位置向上返回两级，找到项目根目录
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
    #__file__ 表示当前 Python 文件的路径
    #pathlib.Path(__file__)：将文件路径转换成 Path 对象
    #.resolve()：将路径解析为绝对路径，并规范化路径
#定义data文件夹的路径
DATA_DIR = ROOT_DIR / "data" / "tinystory"
#定义原始训练、验证数据文件的路径
TRAIN_TXT_DATA_PATH = DATA_DIR / "TinyStoriesV2-GPT4-train.txt"
VAL_TXT_DATA_PATH = DATA_DIR / "TinyStoriesV2-GPT4-valid.txt"
#定义训练、验证数据文件的路径
TRAIN_DATA_PATH = DATA_DIR / "train.dat"
VAL_DATA_PATH = DATA_DIR / "valid.dat"


tokenizer = tiktoken.get_encoding("gpt2")
special_tokens = {"<|endoftext|>"} #原始tinystory文本就有<|endoftext|>
vocab_size = tokenizer.n_vocab


def download_dataset():
    #根据data文件夹路径创建文件夹，.mkdir() 是 pathlib.Path 对象创建文件夹的方法
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    #定义需要下载的文件名
    filenames = [
        "TinyStoriesV2-GPT4-train.txt",
        "TinyStoriesV2-GPT4-valid.txt"
    ]
    for filename in filenames:
        local_path = DATA_DIR / filename
        if local_path.is_file(): #判断这个路径是否指向一个已经存在的普通文件
            print(f'数据集{filename}已经下载')
            continue
        print(f'正在下载{filename}')
        hf_hub_download(
            repo_id="roneneldan/TinyStories",
            repo_type="dataset",
            filename=filename,
            local_dir=DATA_DIR
        )
        '''
                    参数	作用
        repo_id	指定 Hugging Face 仓库
        repo_type="dataset"	表示下载的是 数据集仓库 中的文件
        filename=filename	指定下载哪个文件
        local_dir=DATA_DIR	指定下载文件的本地目录
        '''
        print(f'{filename}下载完成')

def test_tokenizer():
    print('\n——测试tokenizer——')
    print(f'分词器名称：{tokenizer.name} 词表大小：{vocab_size}')

    test_texts = [
        "Once upon a time, there was a little robot.",
        "Hello world! <|endoftext|> Some more text.",
        "<|endoftext|>",
        "你好，世界！"
    ]

    for text in test_texts:
        print(f'\n原文{text}')

        encoded = tokenizer.encode(text, allowed_special=special_tokens)
        print(f'编码：{encoded}')#encoded是这个文本的一堆token ID

        # 词表 本质上建立了 tokenID 和 字节串 之间的映射：
        byte_tokens = [
            tokenizer.decode_single_token_bytes(token_id) for token_id in encoded
        ] #得到这个文本每个token对应的字节串
        print('字节串分词：', byte_tokens)
        str_tokens = [
            b.decode('utf-8', errors='replace') for b in byte_tokens
        ] #得到字节串再解码回的真正字符
        print('直接可读分词：', str_tokens)

        #1.Token ID → 字节串（bytes）2.将字节串拼接，再通过 UTF-8 解码 → 原始字符串（str）
        decoded = tokenizer.decode(encoded)
        print(f'解码：{decoded}')

        print('是否完全还原：', decoded == text)

def encode_txt_as_numpy_array(tokenizer, path_to_txt, save_path):#path_to_txt：原始文本文件的路径（输入路径）；save_path：分词后生成的 .dat 文件的路径（输出路径）
    #统计原始数据总行数
    with open(path_to_txt, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    #统计原始数据总Token数量
    total_tokens = 0
    with open(path_to_txt, 'r', encoding="utf-8") as f:
        #desc="Counting tokens"：description（描述）的缩写，用来设置进度条前面的提示文字 即“正在统计token数量”
        for line in tqdm(f, total=num_lines, desc="正在统计token数量"):
            total_tokens += len(tokenizer.encode(line, allowed_special=special_tokens))
    print(f"token总数：{total_tokens}")
    if total_tokens == 0:
        raise ValueError(f"输入文件：{path_to_txt} 为空或没有Token")

    #创建memmap[memory map（内存映射）]：在磁盘上创建一个能够存储Token ID的一维数组，并通过np.memmap对它进行读写
    dtype = np.uint16 #指定数组中每个元素的数据类型为非负16位整数
    tokens_mm = np.memmap(
    save_path,           # 磁盘文件路径
    dtype=dtype,         # 元素类型：uint16
    mode='w+',           # 创建/覆盖文件，可读写
    shape=(total_tokens,) # 一维数组，长度total_tokens
    )

    #再次遍历原始数据做tokenize存到磁盘上
    pos = 0
    with open(path_to_txt, 'r', encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Tokenizing"):
            ids = tokenizer.encode(line, allowed_special=special_tokens)
            n = len(ids)
            tokens_mm[pos:pos+n] = ids
            pos += n
    tokens_mm.flush() #将内存映射数组中尚未同步的修改刷新到磁盘文件

def main():
    #1.下载数据集
    print("\n——下载数据集——")
    download_dataset()

    #测试分词器
    test_tokenizer()

    #用分词器把 原始数据 分词成给mini-gpt用的 token ID数据 
    print('训练数据：')
    encode_txt_as_numpy_array(tokenizer, TRAIN_TXT_DATA_PATH, TRAIN_DATA_PATH)
    print('验证数据：')
    encode_txt_as_numpy_array(tokenizer, VAL_TXT_DATA_PATH, VAL_DATA_PATH)
    print('所有分词完毕！')

#判断当前文件是不是程序的直接运行入口，运行main()
if __name__ == '__main__': #'__main__'：标识程序的主执行模块
    main()