
from torch._tensor import Tensor
from torch._tensor import Tensor
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

# 将token id 转换为 token 向量，这里的layer需要训练吗？
# [1,2,3]转换成对应的的嵌入向量[[token ID = 1 对应的向量],[token ID = 2 对应的向量],[token ID = 3 对应的向量]]
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight: Float[Tensor, "num_embeddings embedding_dim"] = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=embedding_dim ** -0.5)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... embedding_dim"]:
        # x: (...), int64，表示token的索引
        # 返回: (..., embedding_dim)，每个token索引对应一个embedding向量
        return self.weight[x]