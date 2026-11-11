import torch
import torch.nn as nn
import math
from collections.abc import Iterable #从 Python 标准库 collections.abc 中导入 Iterable，用于表示“可迭代对象”

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    #硬标签交叉熵损失
    get_logit = logits.gather(dim=-1, index=targets.unsqueeze(-1)) #unsqueeze(-1)使targets和logits的维度数量一致
    #logits:(b,s,v) targets:(b,s)->(b,s,1) 在最后一个维度中，找到logits的索引位置等于targets的那项；也就是targets给的是ID位置，我要logits中这个位置上的logit原始分数（不一定是分数最高的)
    #一般算时候是(b*s, v)与(b*s)计算交叉熵
    logsumexp = torch.logsumexp(input=logits, dim=-1, keepdim=True) #保证张量维数不变
    loss = -get_logit + logsumexp

    return loss.mean() #整批数据的平均损失：一个数字元素
#一个batch里共享一个权重矩阵，一批数据中一个张量的每个(s,d)都被这个一样的矩阵操作，所以算损失时候也是这一批损失一起算再求均值

def learning_rate_schedule(t: int, max_lr: float, min_lr: float, warmup_t: int, cosine_annealing_t: int) -> float:
    assert warmup_t < cosine_annealing_t, '迭代步数输入错误！'

    if t < warmup_t:
        lr = max_lr * (t / warmup_t)
    elif warmup_t <= t <= cosine_annealing_t:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi*(t-warmup_t)/(cosine_annealing_t-warmup_t)))
    else:
        lr = min_lr

    return lr

def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    l2_add = 0.0
    for g in grads:
        l2_add += (g.data**2).sum()
    l2_norm = torch.sqrt(l2_add)

    with torch.no_grad():
        for g in grads:
            g.mul_(min(1.0, max_l2_norm/(l2_norm+eps))) #.mul_：原地乘（修改）