from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

from .model_attention import MultiheadSelfAttention 
from .model_basic import Linear, Embedding

"""
SwiGLU是一种用于Transformer中前馈神经网络（FFN）部分的激活结构，全称为SwiGLU（Swish-Gated Linear Unit）。

其核心思想如下：
1. 输入x分别通过两个线性变换（w1和w2）；
2. w1(x)经过SwiSH激活（即F.silu）；
3. 激活后的w1(x)与w2(x)逐元素相乘；
4. 最后再通过w3线性变换还原回原始维度。

这样做的好处是引入了门控机制和非线性激活，使模型表达能力更强，效果优于传统的ReLU或GLU结构。

silu（Swish Linear Unit）是一种激活函数，其数学表达式为：silu(x) = x * sigmoid(x)。
其中sigmoid(x) = 1 / (1 + exp(-x))，所以silu(x)会对输入x进行非线性变换。

silu的优点是：相比ReLU等激活函数，silu在x为负时不会直接置零，而是平滑地衰减，有助于梯度流动和模型收敛。

在PyTorch中，可以通过F.silu(x)或者torch.nn.functional.silu(x)来调用silu激活函数。

# 纬度变化逻辑说明：
# 假设输入 x 的形状为 (..., d_model)
# 1. self.w1(x) 和 self.w2(x) 都是 Linear(d_model, d_ff)，输出形状为 (..., d_ff)
# 2. F.silu(self.w1(x)) 结果形状为 (..., d_ff)
# 3. F.silu(self.w1(x)) * self.w2(x) 是逐元素相乘，结果形状为 (..., d_ff)
# 4. self.w3(...) 是 Linear(d_ff, d_model)，输出形状为 (..., d_model)
# 因此，整个SwiGLU结构输入输出的最后一维都是 d_model，中间隐层为 d_ff


"""
# forward不改变shape
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # w1和w2都是将输入从d_model投影到d_ff的线性层
        self.w1: Linear = Linear(d_model, d_ff)
        self.w2: Linear = Linear(d_model, d_ff)
        # w3是将d_ff还原回d_model的线性层
        self.w3: Linear = Linear(d_ff, d_model)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # x: (..., d_model)，输入的特征张量
        # 返回: (..., d_model)，输出的特征张量
        # 计算SwiGLU: w3(silu(w1(x)) * w2(x))
        # 其中silu是Swish激活函数，*表示逐元素相乘，不是做矩阵乘法
        w1_out: Float[Tensor, "... d_ff"] = self.w1(x)
        w2_out: Float[Tensor, "... d_ff"] = self.w2(x)
        # silu(x) = x * sigmoid(x)
        silu_w1: Float[Tensor, "... d_ff"] = silu(w1_out)
        return self.w3(silu_w1 * w2_out)

def silu(x: torch.Tensor) -> torch.Tensor:
    """
    计算SiLU激活函数（Swish Linear Unit）: silu(x) = x * sigmoid(x)
    支持任意形状的输入张量。
    """
    return x * torch.sigmoid(x)


# forward不改变shape
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1: RMSNorm = RMSNorm(d_model)
        self.attn: MultiheadSelfAttention = MultiheadSelfAttention(d_model, num_heads)
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


"""
1. 为什么输出 shape 是 (batch, seq, vocab_size)？
这是因为Transformer 语言模型的输出本质上是“对每个位置的下一个 token 的概率分布”，即：
输入：一个 batch 的 token 序列，shape 是 (batch, seq)
输出：每个 batch、每个序列位置，都要预测下一个 token 的概率分布（未归一化 logits），
所以输出 shape 是 (batch, seq, vocab_size)
具体解释
对于每个输入序列（比如一句话），模型要对每个 token 位置都输出一个预测（即下一个 token 的概率分布）。
vocab_size 是词表大小，每个位置都要输出一个长度为 vocab_size 的向量，表示“下一个 token 是词表中每个词的概率/分数”。
这样才能用于训练（比如 cross entropy loss），也能用于生成（采样/贪心/beam search）。
"""
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048
    ) -> None:
        super().__init__()
        self.token_emb: Embedding = Embedding(vocab_size, d_model)
        # 不要用nn.ModuleList类型注释，直接用List[TransformerBlock]
        self.layers: list[TransformerBlock] = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
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