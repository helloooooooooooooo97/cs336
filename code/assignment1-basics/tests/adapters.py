from __future__ import annotations

import os
from torch._tensor import Tensor
from torch._tensor import Tensor
from torch._tensor import Tensor
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

import cs336_basics.p1_tokenizer as my_tokenizer 
import cs336_basics.p2_model as my_model
import numpy as np

# pass
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    # 使用Linear类来实现线性变换
    linear = my_model.Linear(d_in, d_out, bias=False)
    linear.weight.data = weights
    return linear(in_features)

    raise NotImplementedError

# pass
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = my_model.Embedding(vocab_size, d_model)
    embedding.weight.data = weights
    return embedding(token_ids)
    raise NotImplementedError

# pass
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = my_model.SwiGLU(d_model, d_ff)
    # 设置权重
    swiglu.w1.weight.data = w1_weight    # diff * model 
    swiglu.w2.weight.data = w3_weight    # diff * model  
    swiglu.w3.weight.data = w2_weight    # model * diff
    # 前向传播
    return swiglu(in_features)
    raise NotImplementedError

# pass
def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    该函数实现了 Scaled Dot-Product Attention（缩放点积注意力）机制，是 Transformer 架构中的核心组件之一。
    其主要作用是根据输入的查询（Q）、键（K）、值（V）以及可选的掩码（mask），计算注意力分数，并输出加权后的值向量。

    具体流程如下：
    1. 计算 Q 和 K 的点积，并除以 sqrt(d_k) 进行缩放，得到注意力分数（attn_scores）。
    2. 如果提供了 mask，则将不需要关注的位置分数设为负无穷，防止其参与 softmax。
    3. 对注意力分数进行 softmax，得到注意力权重（attn_weights）。
    4. 用注意力权重对 V 进行加权求和，得到最终的输出。

    该机制能够让模型根据输入序列中不同位置的信息动态调整关注的重点，从而提升建模能力。
    """
    return my_model.scaled_dot_product_attention(Q, K, V, mask)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attn = my_model.MultiheadSelfAttention(d_model, num_heads)
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.o_proj.weight.data.copy_(o_proj_weight)

    return attn(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """ 
    attn = my_model.MultiheadSelfAttention(d_model, num_heads, True,theta, max_seq_len)
    attn.q_proj.weight.data.copy_(q_proj_weight)
    attn.k_proj.weight.data.copy_(k_proj_weight)
    attn.v_proj.weight.data.copy_(v_proj_weight)
    attn.o_proj.weight.data.copy_(o_proj_weight)
    return attn(in_features,token_positions)

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    使用 my_model.MultiheadSelfAttention 里的 apply_rope 实现 RoPE
    """
    attn = my_model.MultiheadSelfAttention(d_model=d_k, num_heads=1, use_rope=True, theta=theta, max_seq_len=max_seq_len)
    orig_shape = in_query_or_key.shape
    *batch_dims, seq_len, d_k_ = in_query_or_key.shape
    x = in_query_or_key.reshape(-1, seq_len, d_k_)  # (batch, seq, d_k)
    x = x.unsqueeze(1)  # (batch, 1, seq, d_k)
    pos = token_positions.reshape(-1, seq_len)  # (batch, seq)
    # 调用apply_rope
    out: Tensor = attn.apply_rope(x, pos)  # (batch, 1, seq, d_k)
    out = out.squeeze(1)  # (batch, seq, d_k)
    out = out.reshape(*batch_dims, seq_len, d_k_)
    return out


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = my_model.TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    block.attn.q_proj.weight.data = weights['attn.q_proj.weight']
    block.attn.k_proj.weight.data = weights['attn.k_proj.weight']
    block.attn.v_proj.weight.data = weights['attn.v_proj.weight']
    block.attn.o_proj.weight.data = weights['attn.output_proj.weight']
    block.ln1.weight.data = weights['ln1.weight']
    block.ffn.w1.weight.data = weights['ffn.w1.weight']
    block.ffn.w2.weight.data = weights['ffn.w3.weight']
    block.ffn.w3.weight.data = weights['ffn.w2.weight']
    block.ln2.weight.data = weights['ln2.weight']
    return block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    model = my_model.Transformer(vocab_size, d_model, num_layers, num_heads, d_ff,context_length,rope_theta)
    # 加载权重
    model.token_emb.weight.data = weights['token_embeddings.weight']
    for i in range(num_layers):
        prefix = f'layers.{i}.'
        block = model.layers[i]
        block.attn.q_proj.weight.data = weights[prefix+'attn.q_proj.weight']
        block.attn.k_proj.weight.data = weights[prefix+'attn.k_proj.weight']
        block.attn.v_proj.weight.data = weights[prefix+'attn.v_proj.weight']
        block.attn.o_proj.weight.data = weights[prefix+'attn.output_proj.weight']
        block.ln1.weight.data = weights[prefix+'ln1.weight']
        block.ffn.w1.weight.data = weights[prefix+'ffn.w1.weight']
        block.ffn.w2.weight.data = weights[prefix+'ffn.w3.weight']
        block.ffn.w3.weight.data = weights[prefix+'ffn.w2.weight']
        block.ln2.weight.data = weights[prefix+'ln2.weight']
    model.ln_f.weight.data = weights['ln_final.weight']
    model.lm_head.weight.data = weights['lm_head.weight']
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rmsnorm = my_model.RMSNorm(d_model, eps)
    rmsnorm.weight.data = weights
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    import torch
    return torch.nn.functional.silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    import torch
    # 1. 随机采样batch_size个起点
    max_start = len(dataset) - context_length -1  # 要求起点 start + context_length <= len(dataset) -1  
    starts = np.random.randint(0, max_start + 1, size=batch_size) # [0,max_start)采样
    
    # 2. 构造inputs和labels
    inputs = np.stack([dataset[i : i+context_length] for i in starts])
    labels = np.stack([dataset[i+1 : i+context_length+1] for i in starts])
    
    # 3. 转为torch tensor并放到device
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    
    return inputs, labels
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    # 实现softmax函数，需考虑数值稳定性
    import torch
    # 减去最大值以防止溢出
    max_vals, _ = in_features.max(dim=dim, keepdim=True)
    exp_x = torch.exp(in_features - max_vals)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    softmax = exp_x / sum_exp
    return softmax
    raise NotImplementedError


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # inputs: (batch_size, vocab_size)
    # targets: (batch_size,)
    # 1. 数值稳定性处理
    logits = inputs - inputs.max(dim=1, keepdim=True).values
    log_probs = logits - torch.log(torch.exp(logits).sum(dim=1, keepdim=True))
    # 2. 取出每个样本的正确类别的log概率
    nll = -log_probs[torch.arange(inputs.size(0)), targets]
    # 3. 求平均
    return nll.mean()
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    手动实现L2范数梯度裁剪，确保所有参数的梯度L2范数不超过max_l2_norm。
    """
    import torch

    # 1. 收集所有有梯度的参数
    params: list[torch.nn.Parameter] = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return

    # 2. 计算所有参数梯度的L2范数（与PyTorch官方实现一致）
    norm_list: list[torch.Tensor] = [p.grad.detach().norm(2) for p in params]
    total_norm: float = torch.norm(torch.stack(norm_list), 2).item()

    # 3. 计算缩放因子并裁剪
    clip_coef: float = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    返回一个实现了AdamW优化器的torch.optim.Optimizer子类。
    这里我们自定义一个AdamW实现，支持权重衰减（decoupled weight decay）。
    """
    import torch

    class MyAdamW(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        ):
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = group["betas"]

                    state["step"] += 1

                    # Decoupled weight decay
                    if group["weight_decay"] != 0:
                        p.data = p.data.add(-group["weight_decay"] * group["lr"], p.data)

                    # Adam update
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

                    step_size = group["lr"] / bias_correction1

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            return loss

    return MyAdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦退火学习率调度（带线性预热）的参数和迭代次数，返回该迭代下的学习率。
    先缓慢增加到max_learning_rate，再缓慢衰减到min_learning_rate，最后保持最小值
    """
    if it < warmup_iters:
        # Warm-up 阶段：线性增加学习率
        lr = (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine Annealing 阶段：余弦函数衰减
        t = it - warmup_iters
        T = cosine_cycle_iters - warmup_iters
        cos_value = np.cos(np.pi * t / T)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        # Post-annealing 阶段：学习率保持最小值
        lr = min_learning_rate
    return lr

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化保存到磁盘或文件对象。
    """
    import torch

    # 构建要保存的字典
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    # 判断 out 是路径还是文件对象
    if isinstance(out, (str, os.PathLike)):
        torch.save(checkpoint, out)
    else:
        torch.save(checkpoint, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    加载序列化的checkpoint，恢复模型和优化器的状态，并返回迭代次数。
    """
    import torch

    # 判断 src 是路径还是文件对象
    if isinstance(src, (str, os.PathLike)):
        checkpoint = torch.load(src, map_location="cpu")
    else:
        checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> my_tokenizer.BPETokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return my_tokenizer.BPETokenizer(vocab, merges, special_tokens)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练BPE分词器，返回vocab和merges。
    """
    from cs336_basics.p1_tokenizer import BPETrainer

    # 使用BPETrainer进行BPE训练
    trainer = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens, **kwargs)
    vocab, merges = trainer.train(input_path)
    return vocab, merges