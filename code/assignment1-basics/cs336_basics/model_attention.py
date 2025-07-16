
from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float
from torch import Tensor

from .model_basic import Linear

def scaled_dot_product_attention(
    q: Float[Tensor, "batch nheads seq d_head"], 
    k: Float[Tensor, "batch nheads seq d_head"], 
    v: Float[Tensor, "batch nheads seq d_head"], 
    mask: Optional[Float[Tensor, "batch nheads seq seq"]] = None
) -> Float[Tensor, "batch nheads seq d_head"]:
    """
    计算缩放点积注意力
    q, k, v: (batch, nheads, seq, d_head)
    mask: (batch, nheads, seq, seq) 或 None
    返回: (batch, nheads, seq, d_head)
    """
    d_head = q.shape[-1]
    att: Float[Tensor, "batch nheads seq seq"] = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)  # (batch, nheads, seq, seq)
    if mask is not None:
        att = att.masked_fill(mask == 0, float('-inf'))
    # 展开成树后，对同一个父节点下的叶子节点进行softmax
    att = F.softmax(att, dim=-1)
    out: Float[Tensor, "batch nheads seq d_head"] = att @ v  # (batch, nheads, seq, d_head)
    return out

# 多头自注意力机制
# 看不出来有什么问题，如何打印log呢？
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.d_head: int = d_model // num_heads
        self.q_proj: Linear = Linear(d_model, d_model)
        self.k_proj: Linear = Linear(d_model, d_model)
        self.v_proj: Linear = Linear(d_model, d_model)
        self.o_proj: Linear = Linear(d_model, d_model)

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"], 
        mask: Optional[Float[Tensor, "batch num_heads seq seq"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        batch, seq, d_model = x.shape
        # 先做线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 拆分多头，和 reference 保持一致
        from einops import rearrange
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


class MultiheadSelfAttentionWithRoPE(nn.Module):
    """
    带RoPE（旋转位置编码）的多头自注意力机制。
    """
    def __init__(self, d_model: int, num_heads: int, theta: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.theta = theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        token_positions: Optional[Tensor] = None,
        mask: Optional[Float[Tensor, "batch num_heads seq seq"]] = None,
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        x: (batch, seq, d_model)
        token_positions: (batch, seq) 或 None
        mask: (batch, num_heads, seq, seq) 或 None
        """
        batch, seq, d_model = x.shape
        q :Tensor = self.q_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)  # (batch, nheads, seq, d_head)
        k :Tensor = self.k_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        v :Tensor = self.v_proj(x).view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        # 应用RoPE
        if token_positions is not None:
            q = self.apply_rope(q, token_positions)
            k = self.apply_rope(k, token_positions)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (batch, nheads, seq, seq)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (batch, nheads, seq, d_head)
        out = out.transpose(1, 2).contiguous().reshape(batch, seq, d_model)
        return self.o_proj(out)

    def apply_rope(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        对输入x应用RoPE旋转位置编码。
        x: (batch, nheads, seq, d_head)
        token_positions: (batch, seq)
        """
        # 只对最后一个维度d_head做RoPE
        # 参考Llama实现
        # 先构造旋转因子
        device = x.device
        d_head = self.d_head
        seq = x.shape[2]
        # 计算RoPE的频率
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
        # token_positions: (batch, seq) -> (batch, 1, seq, 1)
        pos = token_positions.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq, 1)
        freqs = pos * inv_freq  # (batch, 1, seq, d_head//2)
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





"""
scaled_dot_product_attention 的逻辑讲解：

1. 输入：
   - q, k, v: 形状为 (batch, nheads, seq, d_head)，分别表示查询（query）、键（key）、值（value）张量。
   - mask: 形状为 (batch, nheads, seq, seq) 或 None，用于掩码（mask）不需要关注的位置（如自回归时未来信息）。

2. 步骤：
   a) 首先计算注意力分数（att），即 q 与 k 的转置做矩阵乘法，然后除以 sqrt(d_head) 进行缩放，防止梯度消失或爆炸。
      att = (q @ k.transpose(-2, -1)) / sqrt(d_head)
      结果形状为 (batch, nheads, seq, seq)，表示每个query与所有key的相关性。

   b) 如果提供了mask，则将mask为0的位置填充为负无穷（-inf），这样softmax后这些位置概率为0，实现掩码功能。

   c) 对最后一个维度（即所有key）做softmax，得到归一化的注意力权重。

   d) 用注意力权重加权求和v，得到输出out，形状为 (batch, nheads, seq, d_head)。

3. 输出：
   - 返回加权后的v，形状为 (batch, nheads, seq, d_head)，即每个query根据注意力分数聚合value信息。

简而言之，scaled_dot_product_attention 实现了 Transformer 中最核心的“注意力”机制：每个位置的query根据与所有key的相关性（经过缩放和softmax）加权聚合所有value的信息。
"""
