import torch
from tqdm import tqdm
import wandb # Weights & Biases，实验跟踪/可视化工具
import numpy as np
import os # 操作系统接口模块（文件/目录、环境变量等）
import pathlib # 面向对象的路径处理模块
import sys # 系统相关模块（解释器、标准输入输出、退出等）
import time # 时间模块（计时、延时等）
import json # JSON 数据编解码模块

from basics.model.transformer import transformer_lm
from basics.trainer.AdamW import AdamW
from basics.trainer.utils import cross_entropy, learning_rate_schedule, gradient_clipping
from basics.trainer.data_loading import data_loading
from basics.check_points import save_checkpoint, load_checkpoint
from basics.config import get_config

def get_device(device_arg):
    '''判断用什么设备'''
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg
    #返回用的什么设备

def get_dataset_memmap(path, dtype=np.uint16): #np.uint16：无符号 16 位整数数据类型（没负数）
    '''从内存映射获取token ID数据集'''
    if not os.path.exists(path): #os.path：提供路径操作功能
        raise FileNotFoundError(f"数集未找到：{path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset
    #返回数据集

def main():
    #参数
    args = get_config()

    #设备
    device = get_device(args.device)
    print(f'使用设备为{device}')

    #WandB：用于实时记录、可视化和对比模型训练过程中的 Loss、学习率、GPU 使用情况及超参数等信息。
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)#vars(args)：将Namespace对象中的属性转换成字典，交给WandB保存为实验配置   [vars：variables（变量）]
                #WandB 就会记录本次实验的配置，方便以后比较不同实验
        )
        print(f'wandb初始化：{wandb.run.name}') #在终端打印WandB当前实验运行（Run）的名称，表示实验已初始化

    #检查点目录
    os.makedirs(args.save_ckp_path, exist_ok = True)

    #初始化模型
    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        rope_theta=args.rope_theta,
        d_ff=args.d_ff
    ).to(device=device)

    total_params = sum(p.numel() for p in model.parameters()) #numel() 是 number of elements（元素数量） 的缩写，用于统计张量中有多少个元素
    #返回模型中所有注册的可学习参数（nn.Parameter）
    print(f'模型初始化完成，共有{total_params}个参数')

    #将模型信息记录到wandb
    if not args.no_wandb:
        wandb.log({'模型/总参数量': total_params}) #wandb.log() 接收一个字典（dict）作为参数

    #初始化优化器
    optimizer = AdamW(
        para=model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    #加载数据集
    if args.data_dir is None:
        raise ValueError('必须通过--data_dir指定数据目录')
    #将数据目录路径与文件名拼接，得到训练集和验证集文件的完整路径
    train_data_path = os.path.join(args.data_dir, 'train.dat')#os模块提供路径拼接函数
    valid_data_path = os.path.join(args.data_dir, 'valid.dat')
    train_data = get_dataset_memmap(train_data_path)
    valid_data = get_dataset_memmap(valid_data_path)
    print(f'训练数据集总大小：{len(train_data)}个token')
    print(f'验证数据集总大小：{len(valid_data)}个token')

    #如果指定了Checkpoint（检查点），则从该检查点恢复训练
    start_iter = 0
    if args.resume_ckp_path:
        print(f'从检查点{args.resume_ckp_path}恢复训练')
        start_iter = load_checkpoint(args.resume_ckp_path, model, optimizer)
        print(f'从第{start_iter}次开始训练')

    #训练循环
    model.train()#训练模式
    train_losses = []

    #创建训练进度条
    pbar = tqdm(
        iterable=range(start_iter, args.train_steps),
        desc='训练中',
        initial=start_iter,
        total=args.train_steps
    )

    for iter_num in pbar:
        #学习率调度
        lr = learning_rate_schedule(iter_num, args.max_lr, args.min_lr, args.warm_up_it, args.cosine_it)
        for para_group in optimizer.param_groups:
            para_group['lr'] = lr

        #采样一个批次的数据
        inputs, targets = data_loading(train_data, args.batch_size, args.context_len, device)

        #清空梯度
        optimizer.zero_grad() #PyTorch的loss.backward()默认会累积梯度，而不是覆盖之前的梯度；所以要手动清空

        #前向传播
        logits = model(inputs) #得到输出logits

        #计算交叉熵损失
        logits_flat = logits.view(-1, logits.size(-1)) #(b,s,v) -> (b*s,v)
        targets_flat = targets.view(-1) #(b,s) -> (b*s)
        loss = cross_entropy(logits_flat, targets_flat)

        #反向传播
        loss.backward() #利用链式法则计算损失函数loss对模型中所有需要训练的参数的梯度，并将梯度保存到参数的.grad属性中

        #梯度裁剪
        gradient_clipping(model.parameters(), args.clip_grad_norm)

        #用AdamW更新参数
        optimizer.step() #把参数更新，优化器状态更新


        #记录损失
        train_losses.append(loss.item()) #loss.item()：将只包含一个元素的Tensor转换为Python数值（通常是float）
        #保存Python数值更简单、更省资源

        #日志记录
        if iter_num % args.log_interval == 0:
            ave_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses) #如果记录的损失数量大于100，就取最近的100个
            ppl = np.exp(ave_loss) #perplexity困惑度

            #更新tqdm训练进度
            pbar.set_postfix(
                {
                    #以下返回的都是字符串形式
                    'Loss': f'{loss.item():.4f}', #保留四位小数
                    'Ave_Loss': f'{ave_loss:.4f}', #保留四位小数
                    'PPL': f'{ppl:.2f}', #保留两位小数
                    'LR': f'{lr:.2e}' #科学计数法保留两位小数
                }
            )

            #把以上数据记录到wandb中
            if not args.no_wandb:
                wandb.log(
                    {
                        'train/loss': loss.item(),
                        'train/Ave_Loss': ave_loss,
                        'train/PPL': ppl,
                        'train/LR': lr,
                        'iteration': iter_num
                    }
                )

        #在训练中到一定步数进行 验证val 
        if iter_num % args.val_interval == 0 and iter_num > 0: #避免在iter_num==0第一步时候就验证
            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(args.val_batches):
                    val_inputs, val_targets = data_loading(valid_data, args.batch_size, args.context_len, device)

                    val_logits = model(val_inputs)
                    val_logits_flat = val_logits.view(-1, val_logits.size(-1))
                    val_targets_flat = val_targets.view(-1)

                    val_loss = cross_entropy(val_logits_flat, val_targets_flat)
                    val_losses.append(val_loss.item())

            ave_val_loss = np.mean(val_losses)
            val_ppl = np.exp(ave_val_loss)

            #tqdm.write()可以在输出验证信息的同时，尽量不干扰正在运行的tqdm训练进度条
            tqdm.write(f'Validation | Ave Loss: {ave_val_loss:.4f} | PPL: {val_ppl:.2f}')

            if not args.no_wandb:
                wandb.log(
                    {
                        'val/loss': ave_val_loss,
                        'val/ppl': val_ppl,
                        'iteration': iter_num
                    }
                )

            model.train()

        #保存checkpoint检查点
        if iter_num % args.save_interval == 0 and iter_num > 0:
            checkpoint_path = os.path.join(args.save_ckp_path, f'checkpoint_{iter_num}.pt')
            save_checkpoint(model, optimizer, iter_num, checkpoint_path, args) #保存某次存档的模型/优化器状态为字典

    #关掉tqdm进度条
    pbar.close()

    #保存训练完毕的最终检查点
    final_checkpoint_path = os.path.join(args.save_ckp_path, f'final_checkpoint_{args.train_steps}.pt')
    save_checkpoint(model, optimizer, args.train_steps, final_checkpoint_path, args)
    print(f'最终检查点保存在了{final_checkpoint_path}')

    #关掉wandb
    if not args.no_wandb:
        wandb.finish() #结束当前WandB实验运行（Run），并完成日志同步等收尾工作

    print('模型训练完毕！')

if __name__ == '__main__':
    main()