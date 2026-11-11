
import tiktoken
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import pathlib
import numpy as np
import gzip
import shutil


# ==================== 路径配置 ====================

# 从当前脚本所在目录向上两级，找到项目根目录
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

# OWT 数据单独存放，避免覆盖 TinyStories
DATA_DIR = ROOT_DIR / "data" / "openwebtext"

# 原始压缩文件
TRAIN_GZ_PATH = DATA_DIR / "owt_train.txt.gz"
VAL_GZ_PATH = DATA_DIR / "owt_valid.txt.gz"

# 解压后的文本文件
TRAIN_TXT_DATA_PATH = DATA_DIR / "owt_train.txt"
VAL_TXT_DATA_PATH = DATA_DIR / "owt_valid.txt"

# 分词后的 Token ID 数据
TRAIN_DATA_PATH = DATA_DIR / "train.dat"
VAL_DATA_PATH = DATA_DIR / "valid.dat"


# ==================== 分词器 ====================

tokenizer = tiktoken.get_encoding("gpt2")

special_tokens = {"<|endoftext|>"}

vocab_size = tokenizer.n_vocab


# ==================== 下载数据集 ====================

def download_dataset():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filenames = [
        "owt_train.txt.gz",
        "owt_valid.txt.gz"
    ]

    for filename in filenames:

        local_path = DATA_DIR / filename

        if local_path.is_file():
            print(f"{filename} 已经下载")
            continue

        print(f"正在下载 {filename}")

        hf_hub_download(
            repo_id="stanford-cs336/owt-sample",
            repo_type="dataset",
            filename=filename,
            local_dir=DATA_DIR
        )

        print(f"{filename} 下载完成")


# ==================== 解压数据集 ====================

def decompress_dataset():

    files = [
        (TRAIN_GZ_PATH, TRAIN_TXT_DATA_PATH),
        (VAL_GZ_PATH, VAL_TXT_DATA_PATH)
    ]

    for gz_path, txt_path in files:

        if txt_path.is_file():
            print(f"{txt_path.name} 已经解压")
            continue

        print(f"正在解压 {gz_path.name}")

        # gzip.open 读取压缩文件
        with gzip.open(gz_path, "rb") as f_in:

            # open 创建解压后的文本文件
            with open(txt_path, "wb") as f_out:

                # 将解压后的内容写入文本文件
                shutil.copyfileobj(f_in, f_out)

        print(f"{txt_path.name} 解压完成")


# ==================== 测试 Tokenizer ====================

def test_tokenizer():

    print("\n——测试 GPT-2 Tokenizer——")

    print(f"分词器名称：{tokenizer.name}")
    print(f"词表大小：{vocab_size}")

    test_texts = [
        "The development of artificial intelligence.",
        "Machine learning is a field of computer science.",
        "Hello world! <|endoftext|> Some more text."
    ]

    for text in test_texts:

        encoded = tokenizer.encode(
            text,
            allowed_special=special_tokens
        )

        decoded = tokenizer.decode(encoded)

        print(f"\n原文：{text}")
        print(f"Token ID：{encoded}")
        print(f"解码：{decoded}")
        print(f"是否完全还原：{decoded == text}")


# ==================== 文本编码 ====================

def encode_txt_as_numpy_array(
    tokenizer,
    path_to_txt,
    save_path
):

    # 如果已经生成 .dat，跳过
    if save_path.is_file():
        print(f"{save_path.name} 已经存在，跳过分词")
        return

    # 第一步：统计文本行数
    with open(path_to_txt, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)

    # 第二步：统计总 Token 数量
    total_tokens = 0

    with open(path_to_txt, "r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            total=num_lines,
            desc="正在统计Token数量"
        ):

            ids = tokenizer.encode(
                line,
                allowed_special=special_tokens
            )

            total_tokens += len(ids)

    print(f"Token 总数：{total_tokens:,}")

    if total_tokens == 0:
        raise ValueError(
            f"输入文件 {path_to_txt} 为空或没有Token"
        )

    # 第三步：创建内存映射文件
    dtype = np.uint16

    tokens_mm = np.memmap(
        save_path,
        dtype=dtype,
        mode="w+",
        shape=(total_tokens,)
    )

    # 第四步：再次遍历文本并写入 Token ID
    pos = 0

    with open(path_to_txt, "r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            total=num_lines,
            desc="正在Tokenizing"
        ):

            ids = tokenizer.encode(
                line,
                allowed_special=special_tokens
            )

            n = len(ids)

            tokens_mm[pos:pos+n] = ids

            pos += n

    # 第五步：刷新到磁盘
    tokens_mm.flush()

    del tokens_mm

    print(f"分词完成：{save_path}")
    print(f"数据类型：{dtype}")
    print(f"文件大小：{save_path.stat().st_size / (1024**3):.2f} GB")


# ==================== 主程序 ====================

def main():

    print("\n========== OpenWebText 数据处理 ==========")

    # 1. 下载
    print("\n——下载 OpenWebText——")
    download_dataset()

    # 2. 解压
    print("\n——解压 OpenWebText——")
    decompress_dataset()

    # 3. 测试分词器
    test_tokenizer()

    # 4. 训练集编码
    print("\n——训练集 Tokenization——")

    encode_txt_as_numpy_array(
        tokenizer,
        TRAIN_TXT_DATA_PATH,
        TRAIN_DATA_PATH
    )

    # 5. 验证集编码
    print("\n——验证集 Tokenization——")

    encode_txt_as_numpy_array(
        tokenizer,
        VAL_TXT_DATA_PATH,
        VAL_DATA_PATH
    )

    print("\n========== 所有数据处理完成 ==========")


if __name__ == "__main__":
    main()