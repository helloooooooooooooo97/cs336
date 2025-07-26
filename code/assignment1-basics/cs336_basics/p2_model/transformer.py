from torch._tensor import Tensor
from torch._tensor import Tensor
import torch.nn as nn
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

from .attention import MultiheadSelfAttention 
from .embedding import Embedding
from .rms_norm import RMSNorm
from .swiglu import  SwiGLU
from .linear import Linear

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.ln1: RMSNorm = RMSNorm(d_model)
        self.attn: MultiheadSelfAttention = MultiheadSelfAttention(d_model, num_heads, True, theta, max_seq_len)
        self.ln2: RMSNorm = RMSNorm(d_model)
        self.ffn: SwiGLU = SwiGLU(d_model, d_ff)

    def forward(
        self, 
        x: Float[Tensor, "batch seq d_model"], 
        mask: Optional[Float[Tensor, "batch num_heads seq seq"]] = None
    ) -> Float[Tensor, "batch seq d_model"]:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float 
    ) -> None:
        super().__init__()
        self.token_emb: Embedding = Embedding(vocab_size, d_model)
        # 不要用nn.ModuleList类型注释，直接用List[TransformerBlock]
        self.layers: list[TransformerBlock] = [TransformerBlock(d_model, n_heads, d_ff, max_seq_len, theta) for _ in range(n_layers)]
        self.ln_f: RMSNorm = RMSNorm(d_model)
        self.lm_head: Linear = Linear(d_model, vocab_size, bias=False)

    def forward(
        self, 
        idx: Int[Tensor, "batch seq"], 
        mask: Optional[Float[Tensor, "batch num_heads seq seq"]] = None
    ) -> Float[Tensor, "batch seq vocab_size"]:
        # 输入 idx: (batch, seq)
        x = self.token_emb(idx)  # (batch, seq, d_model)
        for layer in self.layers:
            x = layer(x, mask)  # (batch, seq, d_model)
        x = self.ln_f(x)  # (batch, seq, d_model)
        logits = self.lm_head(x)  # (batch, seq, vocab_size)
        return logits 