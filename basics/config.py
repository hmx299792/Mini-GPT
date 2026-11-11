import argparse # 命令行参数解析模块

def get_config(): #命令行参数配置系统：不仅能设置模型超参数，还能配置优化器、训练过程、数据路径和实验日志
    parser = argparse.ArgumentParser(description='训练Mini-GPT') #ArgumentParser()：创建一个命令行参数解析器，可以在命令行修改配置参数

    #模型参数
    parser.add_argument('--vocab_size', type=int, default=50257, help='词表大小') #type：将输入转换为 Python 整数
    parser.add_argument('--d_model', type=int, default=256, help='隐藏维度')
    parser.add_argument('--d_ff', type=int, default=768, help='前馈神经网络维度')
    parser.add_argument('--context_len', type=int, default=256, help='最大句子长度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_kv_heads', type=int, default=8, help='KV注意力头数，必须能整除num_heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='旋转位置编码基数')

    #优化器参数
    parser.add_argument('--max_lr', type=float, default=3e-4, help='最大学习率')
    parser.add_argument('--min_lr', type=float, default=3e-5, help='最小学习率')
    parser.add_argument('--warm_up_it', type=int, default=500, help='预热步数')
    parser.add_argument('--cosine_it', type=int, default=30000, help='余弦退火步数')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='权重衰减')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪范数阈值')

    #训练参数
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--train_steps', type=int, default=30000, help='总训练步数')#先设置6000观察损失，之后再加总步数
    parser.add_argument('--log_interval', type=int, default=1, help='日志记录间隔')
    parser.add_argument('--val_interval', type=int, default=500, help='验证间隔')#指每隔多少个 iteration/step 进行一次验证
    parser.add_argument('--val_batches', type=int, default=50, help='验证批次数量')#指每次验证时，抽取多少个batch的数据来评估
    parser.add_argument('--save_interval', type=int, default=1000, help='Checkpoint检查点保存间隔')
    parser.add_argument('--save_ckp_path', type=str, default='./checkpoints', help='Checkpoint检查点保存目录')
    parser.add_argument('--resume_ckp_path', type=str, default=None, help='恢复训练的Checkpoint检查点路径')
    
    #数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据文件夹路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备: auto, cpu, cuda, mps')
    
    #实验跟踪工具参数
    parser.add_argument('--wandb_project', type=str, default='从零构建GPT', help='Wandb项目名称')
    parser.add_argument('--wandb_run_name', type=str, default='mini_gpt_30k', help='Wandb运行名称')
    parser.add_argument('--no_wandb', action='store_true', help='禁用Wandb日志记录')#输入--no_wandb就会关掉wandb

    args = parser.parse_args()
    #parse_args()是ArgumentParser对象的方法；将所有参数封装成一个argparse.Namespace对象：一个专门存放配置参数的对象，可以通过 对象.属性名 来访问参数值
    #argparse.Namespace：一个专门存放配置参数的对象，可以通过 对象.属性名 来访问参数值即args.batch_size
    
    return args