import torch
import torch.nn as nn

def save_checkpoint(model, optimizer, iteration, out, args):
    '''
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int 迭代次数
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    torch.save(
        obj={
            #.state_dict()：获取模型或优化器当前的状态以字典形式返回（字典嵌套字典）
            'model_state': model.state_dict(), #键对应的值是一个字典
            'optimizer_state': optimizer.state_dict(),
            'iteration': iteration,
            'config': vars(args)
        },
        f=out
    )

def load_checkpoint(src, model, optimizer):
    '''
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    '''
    #读取之前某一次保存的checkpoint，
    check_point = torch.load(f=src, map_location='cpu') #ckp就是torch.save保存的字典
    #把checkpoint中保存的字典中的参数值，重新覆盖到当前模型对应的参数上。
    model.load_state_dict(check_point['model_state']) #.load_state_dict()：将保存的状态字典加载到模型或优化器中
    optimizer.load_state_dict(check_point['optimizer_state'])
    return check_point['iteration']