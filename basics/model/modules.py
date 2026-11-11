import torch
import torch.nn as nn
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None): #device，dtype可选存储
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))#nn.Parameter 是 PyTorch 中用来把一个张量标记为模块可学习参数的类
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T #矩阵乘法

    def _init_weight(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std) #截断正态分布初始化权重

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings#词表vocab大小
        self.embedding_dim = embedding_dim#嵌入维度dim大小
        self.device = device
        self.dtype = dtype
        self.embed_weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self._init_weight()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype == torch.long:
            pass
        else:
            token_ids = token_ids.long()
        return self.embed_weight[token_ids] #根据token_ids索引embedding矩阵，返回对应的嵌入向量

    def _init_weight(self):
        nn.init.trunc_normal_(self.embed_weight, mean = 0, std=1, a=-3, b=3) #截断正态分布初始化权重

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g_weight = nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))#每一个xi对应一个γi(g_weight)
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #RMSNorm 内部的 RMS 计算建议使用 FP32；输出时候再将结果转换回输入精度
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)#分母
        out = x / rms * self.g_weight #归一化后乘以γi
        return out.to(dtype=in_dtype) #返回输入精度

    def _init_weight(self):
        nn.init.ones_(self.g_weight) #初始化γi为1

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device = device, dtype = dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device = device, dtype = dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device = device, dtype = dtype))
        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x @ self.w1.T
        x1 = (x1 * torch.sigmoid(x1)) #silu激活函数
        x2 = x @ self.w3.T
        x = (x1 * x2) @ self.w2.T
        return x

    def _init_weight(self):
        std = (2 / (self.d_model + self.d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w2, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, std=std, a=-3 * std, b=3 * std)

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None, dtype = None):
        #theta就是base基底
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.freq_arrange = 1 / (self.theta**(torch.arange(0, d_k, 2).to(device=device, dtype=torch.float) / self.d_k)) #torch.float默认32位
        self.register_buffer(name='inv_freq', tensor=self.freq_arrange, persistent=False) #注册一个buffer“逆频率”，不参与梯度更新

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        # x: (B, h, s, d_k)  或 (B, s, d) 取决于你在哪一层调用，这里假设是 (B, h, s, d_k)
        B = x.size(0)
        S = x.size(-2)

        # 1)准备positions
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)#如果没有提供token_positions，就生成一个默认的token_positions
        else:
            #如果传进来是(B,s)，取一行
            if token_positions.dim() == 2:
                token_positions = token_positions[0, :]#取第一行

        # 2)相位：最后是(s, d_k/2)
        theta = token_positions[:, None] * self.inv_freq[None, :] # s ->扩充 (s, 1); d_k/2 ->扩充 (1, d_k/2) -->利用广播机制相乘 (S, d_k/2)

        # 3)cos/sin：(s, d_k) -> (1, 1, s, d_k)以便对(B, h, s, d_k)广播
        #theta.cos()和theta.sin()分别计算余弦和正弦值，然后使用repeat_interleave将每个元素重复两次，以便与d_k的维度匹配。最后使用unsqueeze在前面添加两个维度，使其形状为(1, 1, s, d_k)，以便在后续操作中与输入张量进行广播。
        cos = theta.cos().repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, s, d_k)
        sin = theta.sin().repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, s, d_k)

        #4)旋转x：(B, h, s, d_k)
        x_rotated = self.rotate_tensor(x)

        #5)应用RoPE公式
        x = (x * cos) + (x_rotated * sin)
        return x

    def rotate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, '... (s r) -> ... s r', r=2)#解绑"：将相邻两个元素分组：(...,s*r) -> (...,s,r=2)
        x_even, x_odd = x.unbind(dim=-1)#unbind(dim=-1)表示沿着最后一个维度拆分张量，并删除该维度：裂开成两组
        x = torch.stack((-x_odd, x_even), dim=-1)#交换顺序，并对奇数索引元素取负
        x = rearrange(x, '... s r -> ... (s r)')#恢复原来的维度：(...,s,r=2) -> (...,s*r)
        return x

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True).values #.values是为了获取最大值的张量，而不是索引
    #torch.max() 会返回两个结果：values, indices = torch.max(scaled_logits, dim=-1, keepdim=True)
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim = True)
    return x

#缩放点积注意力
def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    attention_score = q @ k.transpose(-2, -1) / (q.size(-1)**0.5)
    #attention_score = einsum(q, k, '... i d, ... j d -> ... i j')
    #增加掩码
    if mask is not None:#有掩码操作时候
        attention_score = attention_score.masked_fill_(mask == False, float('-inf')) #mask为False的地方填充-inf；mask == False与mask == 0等价
    mask_attention = softmax(attention_score, dim=-1)
    attention_output = mask_attention @ v
    #attention_output = einsum(mask_attention, v, '... s_q s_k, ... s_k d_v -> ... s_q d_v')
    return attention_output

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, position_embedding: nn.Module = RotaryPositionEmbedding, max_seq_len = None, theta = None, token_positions = None, device = None, dtype = None, use_causal_mask: bool = True, num_kv_heads: int = None): #num_kv_heads：kv头数
        super().__init__()
        self.pe = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask

        assert d_model % num_heads == 0, 'd_model必须被num_heads整除！'
        self.d_k = d_model // num_heads

        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert self.num_kv_heads > 0
        assert num_heads % self.num_kv_heads == 0, 'num_heads必须被num_kv_heads整除！'
        self.num_kv_groups = num_heads // self.num_kv_heads #GQA中每个KV Head被多少个Q Head共享

        self.w_q = Linear(d_model, self.num_heads * self.d_k, device=device, dtype=dtype)
        self.w_k = Linear(d_model, self.num_kv_heads * self.d_k, device=device, dtype=dtype)
        self.w_v = Linear(d_model, self.num_kv_heads * self.d_k, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)
        
        if position_embedding is not None and max_seq_len is not None and theta is not None:
            self.pe = position_embedding(theta, self.d_k, max_seq_len, device=device, dtype=dtype)
        self.token_positions = token_positions

    def causal_mask(self, seq_len, past_len=0):
        mask = torch.tril(torch.ones(seq_len, seq_len + past_len, dtype=torch.bool), diagonal=past_len).unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len+past_len) 矩形掩码
        return mask

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool=False):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)

        #训练时通常设置use_cache=False，推理生成时设置use_cache=True
        past_len = 0
        if use_cache and past_kv is not None: #use_cache是True，past_kv是not None
            past_len = past_kv[0].size(-2) #past_kv里存k_rotated和v；k_r形状为(b,h_kv,s_past,d_k) s_past就是推理时候前面的token长度

        token_positions = self.token_positions

        #cache模式下修正rope位置索引：
        if use_cache:
            token_positions = torch.arange(
                past_len,
                past_len + q.size(-2), #prefill时候是prompt长度，逐个token推理时候是1
                device=q.device
            )

        # 如果没有传入 max_seq_len 或 theta，那么 self.pe = None，导致 q_rotated 和 k_rotated 没有定义
        if self.pe is not None:
            q_rotated = self.pe(q, token_positions)
            k_rotated = self.pe(k, token_positions)
        else:
            q_rotated = q
            k_rotated = k

        #KV Cache：拼接历史K、V
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            k_rotated = torch.cat([past_k, k_rotated], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        present_kv = None #默认不保存
        #保存kv cache（还未扩展kv head）
        if use_cache:
            present_kv = (k_rotated, v) #保存到元组中

        # GQA：将KV Heads扩展到与Q Heads相同
        if self.num_kv_groups > 1:
            k_rotated = k_rotated.repeat_interleave(self.num_kv_groups, dim=1) #每个k_r扩充num_kv_groups个，和q数量一致
            v = v.repeat_interleave(self.num_kv_groups, dim=1) #每个v扩充num_kv_groups个，和q数量一致

        mask = None
        if self.use_causal_mask:
            mask = self.causal_mask(q.size(-2), past_len if use_cache else 0).to(q.device)

        attention = scale_dot_product_attention(q_rotated, k_rotated, v, mask)
        attention = rearrange(attention, 'b h s d -> b s (h d)', h=self.num_heads) #einops 的维度名称不能包含下划线 _
        output = self.w_o(attention)

        if use_cache:
            return output, present_kv

        return output