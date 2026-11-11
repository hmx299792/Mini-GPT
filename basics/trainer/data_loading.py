import torch
import numpy.typing as npt

#Token 数据集中抽取一个 Batch 的样本
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    #输出两个张量，一个x，一个右移的y
    dataset_len = len(dataset)
    #创建空的x张量
    inputs = torch.empty(batch_size, context_length, dtype=torch.long) #torch.long等价于torch.int64
    #创建空的y张量
    targets = torch.empty(batch_size, context_length, dtype=torch.long)

    for i in range(batch_size):
        #torch.randint(low, high, size)
        start_idx = torch.randint(0, dataset_len-context_length, (1,)).item() #.item()：将单元素 Tensor 转换为 Python 数值
        input_seq = dataset[start_idx: start_idx+context_length]
        target_seq = dataset[start_idx+1: start_idx+context_length+1]
        inputs[i] = torch.tensor(input_seq, dtype=torch.long) #torch.tensor是把数组转化成张量
        targets[i] = torch.tensor(target_seq, dtype=torch.long)

    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    return (inputs, targets)