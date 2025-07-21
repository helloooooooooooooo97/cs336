from torch._tensor import Tensor
import torch

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算输入张量x在指定维度上的softmax
    参数:
        x: 输入张量
        dim: 归一化的维度，默认为最后一维
    返回:
        与x形状相同的softmax结果
    """
    # 为了数值稳定性，先减去最大值
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum: Tensor = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum

