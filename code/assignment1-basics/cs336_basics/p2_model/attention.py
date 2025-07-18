from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
from .linear import Linear

def scaled_dot_product_attention(
    q: Float[Tensor, "batch nheads seq d_head"], 
    k: Float[Tensor, "batch nheads seq d_head"], 
    v: Float[Tensor, "batch nheads seq d_head"], 
    mask: Optional[Float[Tensor, "batch nheads seq seq"]] = None
) -> Float[Tensor, "batch nheads seq d_head"]:
    d_head = q.shape[-1]
    att: Float[Tensor, "batch nheads seq seq"] = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)  # (batch, nheads, seq, seq)
    if mask is not None:
        att = att.masked_fill(mask == 0, float('-inf'))
    # 展开成树后，对同一个父节点下的叶子节点进行softmax
    att = F.softmax(att, dim=-1)
    out: Float[Tensor, "batch nheads seq d_head"] = att @ v  # (batch, nheads, seq, d_head)
    return out

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, theta: float = 10000.0, max_seq_len: int = 2048) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_head: int = d_model // num_heads
        self.q_proj: Linear = Linear(d_model, d_model)
        self.k_proj: Linear = Linear(d_model, d_model)
        self.v_proj: Linear = Linear(d_model, d_model)
        self.o_proj: Linear = Linear(d_model, d_model)
        self.use_rope: bool = use_rope
        self.theta: float = theta  # RoPE的theta参数
        self.max_seq_len: int = max_seq_len  # 新增max_seq_len参数
        # 预生成 inv_freq
        self.register_buffer("inv_freq",1.0 / (self.theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head)))

    def forward(self, in_features: torch.Tensor, token_positions: Optional[torch.Tensor] = None):
        """
        优化可读性，分别计算q, k, v，不使用一次性矩阵乘法
        如果use_rope为True，则对q和k应用RoPE编码
        """
        seq_len = in_features.shape[-2]
        
        # 分别计算q, k, v
        q = self.q_proj(in_features)  # (batch, seq_len, d_model)
        k = self.k_proj(in_features)  # (batch, seq_len, d_model)
        v = self.v_proj(in_features)  # (batch, seq_len, d_model)

        # 重新排列为多头注意力的形状
        q = rearrange(q, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        k = rearrange(k, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)
        v = rearrange(v, "... seq_len (h d_head) -> ... h seq_len d_head", h=self.num_heads)

        # 如果需要RoPE编码，则对q和k应用
        if self.use_rope:
            if token_positions is None: # 自动生成 [0, 1, 2, ..., seq_len-1]
                token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(in_features.shape[0], -1)
            q = self.apply_rope(q, token_positions)
            k = self.apply_rope(k, token_positions)

        # 构造因果mask
        casual_mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool() 
        casual_mask = casual_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

        # 计算注意力输出 ~casual_取反 
        output = scaled_dot_product_attention(q, k, v, ~casual_mask)

        # 恢复输出形状
        output = rearrange(output, "... h seq_len d_head ->  ... seq_len (h d_head)")
        return self.o_proj(output)
    
    def apply_rope(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        对输入x应用RoPE旋转位置编码。
        x: (batch, nheads, seq, d_head)
        token_positions: (batch, seq)
        """
        device = x.device
        d_head = self.d_head
        inv_freq = torch.as_tensor(self.inv_freq, device=x.device, dtype=x.dtype)
        pos = token_positions.unsqueeze(1).unsqueeze(-1)
        freqs = pos * inv_freq
        # 计算cos和sin
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # x: (batch, nheads, seq, d_head)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rope_even = x1 * cos - x2 * sin
        x_rope_odd = x1 * sin + x2 * cos
        # 交错合并
        x_rope = torch.stack([x_rope_even, x_rope_odd], dim=-1).flatten(-2)
        return x_rope