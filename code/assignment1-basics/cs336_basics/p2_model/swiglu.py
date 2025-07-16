from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .linear import Linear

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

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