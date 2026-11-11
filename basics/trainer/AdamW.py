import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from collections.abc import Callable, Iterable #抽象基类（Abstract Base Classes）
from typing import Optional #类型注解；一个值可以是某个类型，也可以是 None

class AdamW(Optimizer):
    def __init__(self, para: Iterable[nn.Parameter], lr: float=1e-3, betas: tuple[float, float]=(0.9, 0.95), eps: float=1e-8, weight_decay: float=0.01):
        #[]都是在描述元组内数据类型   ()创建一个实际的元组对象

        #把初始化超参数打包成字典：
        defaults = {
            'lr': lr,
            'beta1': betas[0],
            'beta2': betas[1],
            'eps': eps,
            'lamb': weight_decay
        }
        super().__init__(params=para, defaults=defaults)

    #训练流程里的optimizer.step()就是调用下面这个def step()：
    def step(self, closure: Callable | None = None): #closure 是一个让优化器在 step() 内部重新计算损失和梯度的回调函数
        loss = None
        if closure is not None:
            with torch.enable_grad(): #在这个代码块内部开启 PyTorch 的自动梯度记录功能（Autograd）
                loss = closure() #closure 不是损失函数本身，而通常是一个包含前向传播、计算损失和反向传播的函数
        '''
        Optimizer自带的实例属性param_groups，是一个列表（list）
        列表中的每个元素都是一个字典（dict），每个字典保存了一组需要训练的参数，以及这组参数对应的优化器超参数
        分组情况是调用AdamW时候自己设置的
        
        [
        {"params": [...layer1 参数...], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        {"params": [...layer2 参数...], "lr": 5e-4, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
        ...
        ]
        (超参数是defaults里自己定义的)

        同一个权重矩阵中的所有元素共享优化器超参数和更新步数，但每个元素独立维护自己的梯度、一阶动量和二阶动量
        '''
        for group in self.param_groups:
            for p in group['params']: #一个p就是一个权重矩阵（或者偏置什么的）
                if p.grad is None:
                    continue
                # .data：绕过pytorch的自动求导机制
                # .data 是 PyTorch 的 torch.Tensor 类自带的属性
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW不接受稀疏矩阵Sparse Matrix！')
                    #稀疏矩阵：绝大多数元素都是 0，只有少数元素非零可以只存储非零元素及其位置，从而节省内存和计算资源
                    #RuntimeError通常在程序语法正确、但运行时检测到不该发生的情况时抛出

                lr = group['lr']
                beta1 = group['beta1']
                beta2 = group['beta2']
                eps = group['eps']
                lamb = group['lamb']

                #self.state 是 PyTorch 的 torch.optim.Optimizer 父类自带的属性
                #它的数据结构是collections.defaultdict（带默认值的字典）
                state = self.state[p] #p就是self.state的键；state就是self.state里p对应的值
                #找到state里存的每个参数的一阶矩动量，没有就创建一个和梯度张亮形状一致的新张量用来存动量
                m_pre = state.get('m', torch.zeros_like(grad))
                #找到state里存的每个参数的二阶矩（梯度平方均值），没有就创建一个和梯度张亮形状一致的新张量用来存二阶矩（梯度平方均值）
                v_pre = state.get('v', torch.zeros_like(grad))
                t = state.get('t', 1.0) #当前步数
                state['m'] = beta1 * m_pre + (1-beta1) * grad #把新的m更新到字典里
                state['v'] = beta2 * v_pre + (1-beta2) * torch.square(grad) #把新的v更新到字典里
                lr_t = lr * math.sqrt(1-beta2**t) / (1-beta1**t)
                eps_t = eps * math.sqrt(1-beta2**t)
                p.data -= lr * lamb * p.data
                p.data -= lr_t * state['m'] / (torch.sqrt(state['v']) + eps_t)

                state['t'] = t + 1
                #此时就完成了参数的更新，并更新了状态t m v到字典里
        
        return loss