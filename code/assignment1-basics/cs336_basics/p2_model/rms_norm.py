from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight