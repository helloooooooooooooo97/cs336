
from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True) -> None:
        super().__init__()
        self.weight: Float[Tensor, "d_out d_in"] = nn.Parameter(torch.empty(d_out, d_in))  # 权重参数，形状为(d_out, d_in)
        if bias:
            self.bias: Optional[Float[Tensor, "d_out"]] = nn.Parameter(torch.empty(d_out))  # 偏置参数，形状为(d_out,)
        else:
            self.bias = None
        # 使用Kaiming均匀分布初始化权重参数，a=5**0.5是LeCun推荐的参数
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # 如果有偏置参数，则将偏置初始化为0
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        # x: (..., d_in)
        # weight: (d_out, d_in)
        # bias: (d_out,)
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y