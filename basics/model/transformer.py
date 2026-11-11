import torch
import torch.nn as nn
from . import modules #要modules.Linear, modules.Embedding, modules.RMSNorm等类，需要使用modules.前缀
# from .modules import *  直接用Linear, Embedding, RMSNorm等类，不需要导入所有模块

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None, num_kv_heads: int = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.rms1 = modules.RMSNorm(d_model)
        self.rms2 = modules.RMSNorm(d_model)

        self.attn = modules.GroupedQueryAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta, num_kv_heads=num_kv_heads)
        self.ffn = modules.SwiGLU(d_model=d_model, d_ff=d_ff)

        # if hasattr(self.attn, 'pe') and hasattr(self.attn.pe, 'inv_freq'): #hasattr 是 "has attribute" 的缩写，即"是否有属性"
        #     #buf 是 buffer（缓冲区）
        #     buf = self.attn.pe.inv_freq
        #     try:
        #         #把已经注册的 inv_freq 从默认的持久化 Buffer 改成非持久化 Buffer
        #         del self.attn.pe_buffers['inv_freq'] #删除原来的 inv_freq buffer
        #         self.attn.pe.register_buffer('inv_freq', buf, persistent = False)
        #     #except：捕获异常；Exception：Python 中大多数常见异常的基类，例如 ValueError、TypeError、KeyError
        #     except Exception:
        #         pass #发现有异常就pass

        #把整个模型搬到某处或者把整个模型数据类型改变
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool=False):
        if use_cache:
            attn_output, present_kv = self.attn(self.rms1(x), past_kv, use_cache=True)
        else:
            attn_output = self.attn(self.rms1(x))
            present_kv = None
        x = x + attn_output
        x = x + self.ffn(self.rms2(x))

        if use_cache:
            return x, present_kv

        return x

class transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, rope_theta: float, d_ff: int, device = None, dtype = None, num_kv_heads: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding = modules.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta, num_kv_heads=num_kv_heads) 
                for _ in range(num_layers)
            ]
        )
        self.output_norm = modules.RMSNorm(d_model=d_model)
        self.project = modules.Linear(in_features=d_model, out_features=vocab_size)
        
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool=False):
        x = self.embedding(x) #嵌入
        if not use_cache:
            for layer in self.layers: #对layers的容器中的每个模型layer
                x = layer(x) #进行transformer)block
            x = self.output_norm(x) #输出时候再归一化
            x = self.project(x) #输出没有softmax的logits
            return x

        if past_kv is None:
            past_kv = [None] * self.num_layers

        if len(past_kv) != self.num_layers:
            raise ValueError("past_kv的长度必须等于num_layers")

        new_past_kv = []

        for i, layer in enumerate(self.layers):
            x, present_kv = layer(x, past_kv=past_kv[i], use_cache=True)
            new_past_kv.append(present_kv)
        x = self.output_norm(x)
        x = self.project(x)
        return x, new_past_kv #每一次new_past_kv变成下一次的past_kv