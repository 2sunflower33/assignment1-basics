from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
from collections import Counter
import numpy.typing as npt
import torch
from torch import Tensor
import regex as re 
from collections import defaultdict
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Iterator
import torch.nn as nn 
import math 
from einops import rearrange, einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        args = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty( (num_embeddings, embedding_dim), **args ))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

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

    embedding_layer = Embedding(vocab_size, d_model)
    embedding_layer.load_state_dict({'weight': weights})
    return embedding_layer.forward(token_ids)



class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        args = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **args))
        std = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
        return

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

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
    linear_layer = Linear(d_in, d_out)
    linear_layer.load_state_dict({'weight': weights})
    return linear_layer.forward(in_features)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff: int, device=None, dtype=None):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **args)
        self.w3 = Linear(d_model, d_ff, **args)
        self.w2 = Linear(d_ff, d_model, **args)
    
    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2( self.SiLU( self.w1(x) ) * self.w3(x)) 

def get_optimal_d_ff(d_model: int, multiplier: float = 8.0 / 3) -> int:
    """
    Calculate d_ff as approximately multiplier * d_model, 
    rounded to the nearest multiple of 64 for hardware efficiency.
    """
    target_d_ff = int(multiplier * d_model)
    # Round to nearest multiple of 64
    return ((target_d_ff + 31) // 64) * 64

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
    # d_ff = get_optimal_d_ff(d_model)

    swiglu = SwiGLU(d_model, d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
 
    return swiglu.forward(in_features)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V:torch.Tensor, mask:Optional[torch.Tensor]=None) -> torch.Tensor:
        d_k = Q.shape[-1]
        qk = einsum(Q, K, '... d_m d_k, ... d_n d_k -> ... d_m d_n') / torch.sqrt(torch.tensor(d_k))  
        if mask is not None:
            qk = qk.masked_fill(~mask, -torch.inf) # don't do in-place assignment
        return einsum (softmax(qk, dim=-1), V, '... d_m d_n, ... d_n d_v -> ... d_m d_v')

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    dot_product = ScaledDotProductAttention()
    return dot_product(Q, K, V, mask) 


class MultiheadSelfAttentionV0(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        d_k = d_model
        d_v = d_model
        self.q_proj_weight = nn.Parameter(torch.empty( (d_k, d_model)))
        self.k_proj_weight = nn.Parameter(torch.empty( (d_k, d_model)))
        self.v_proj_weight = nn.Parameter(torch.empty( (d_v, d_model)))
        self.o_proj_weight = nn.Parameter(torch.empty( (d_model, d_v)))
    
    def forward(self, x:torch.Tensor):
        Q = einsum(x, self.q_proj_weight, "... seq_len d_model, d_k d_model -> ... seq_len d_k") # Q: ... seq_len d_k
        K = einsum(x, self.k_proj_weight, "... seq_len d_model, d_k d_model -> ... seq_len d_k")
        V = einsum(x, self.v_proj_weight, "... seq_len d_model, d_v d_model -> ... seq_len d_v")
        Q_mh = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads) # Q: ... seq_len h d_mh
        K_mh = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)
        V_mh = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)

        dot_product_atten = ScaledDotProductAttention()
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        mh_atten = dot_product_atten(Q_mh, K_mh, V_mh, mask) # ... h, seq_len, d_v
        mh_atten = rearrange(mh_atten, "... h seq_len d -> ... seq_len (h d)") # ... seq_len, d_model
        return einsum(mh_atten, self.o_proj_weight, "... seq_len d_v, d_model d_v -> ... seq_len d_model")

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        d_k = d_model
        d_v = d_model
        self.linear_qkv = Linear(d_model, d_k+d_k+d_v)
        # self.linear_q = Linear(d_model, d_k)
        # self.linear_k = Linear(d_model, d_k)
        # self.linear_v = Linear(d_model, d_v)
        self.linear_o = Linear(d_v, d_model)
    
    def forward(self, x:torch.Tensor):
        QKV = self.linear_qkv(x) # output size: d_k + d_k + d_v
        # QKV_mh = rearrange(QKV, "... seq_len (h d) -> ... h seq_len d", h=self.num_heads)
        # Q_mh, K_mh, V_mh = QKV_mh.chunk(3, -1)
        
        
        Q, K, V = QKV.chunk(3, -1)
        # Q = self.linear_q(x)
        # K = self.linear_k(x)
        # V = self.linear_v(x)
        Q_mh = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads) # Q: ... seq_len h d_mh
        K_mh = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)
        V_mh = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)

        dot_product_atten = ScaledDotProductAttention()
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        mh_atten = dot_product_atten(Q_mh, K_mh, V_mh, mask) # ... h, seq_len, d_v
        mh_atten = rearrange(mh_atten, "... h seq_len d -> ... seq_len (h d)") # ... seq_len, d_model
        return self.linear_o(mh_atten)
    

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
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
    mh_attention = MultiheadSelfAttention(d_model, num_heads=num_heads)
    mh_attention.load_state_dict({
        'linear_qkv.weight': torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0),
        'linear_o.weight': o_proj_weight
    })
    # mh_attention.load_state_dict({
    #     'linear_q.weight': q_proj_weight,
    #     'linear_k.weight': k_proj_weight,
    #     'linear_v.weight': v_proj_weight,
    #     'linear_o.weight': o_proj_weight
    # })
    # mh_attention.load_state_dict({
    #     'q_proj_weight': q_proj_weight,
    #     'k_proj_weight': k_proj_weight,
    #     'v_proj_weight': v_proj_weight,
    #     'o_proj_weight': o_proj_weight,
    # })
    return mh_attention(in_features)



class MultiheadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int, theta:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        assert d_model % num_heads == 0
        d_k = d_model
        d_v = d_model
        self.q_proj = Linear(d_model, d_k)
        self.k_proj = Linear(d_model, d_k)
        self.v_proj = Linear(d_model, d_v)
        self.output_proj = Linear(d_v, d_model)
        self.rope =  RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)

    
    def forward(self, x:torch.Tensor, positions):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q_mh = rearrange(Q, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads) # Q: ... seq_len h d_mh
        K_mh = rearrange(K, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)
        V_mh = rearrange(V, "... seq_len (h d) -> ... h seq_len d", h = self.num_heads)
        seq_len = x.shape[-2]

        Q_mh_rope = self.rope(Q_mh, positions)
        K_mh_rope = self.rope(K_mh, positions)

        dot_product_atten = ScaledDotProductAttention()
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        mh_atten = dot_product_atten(Q_mh_rope, K_mh_rope, V_mh, mask) # ... h, seq_len, d_v
        mh_atten = rearrange(mh_atten, "... h seq_len d -> ... seq_len (h d)") # ... seq_len, d_model
        return self.output_proj(mh_atten)
    

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
) -> Float[Tensor, " ... sequence_length d_out"]:
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
    mh_atten_rope = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    mh_atten_rope.load_state_dict({
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight
    })
    return mh_atten_rope(in_features, token_positions)

class RotaryPositionalEmbeddingEinsum(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        R = torch.zeros(max_seq_len, d_k // 2, 2, 2, device=device)
        for i in range(max_seq_len):    
            for k in range(d_k // 2):
                theta_ik = i / (theta ** (2 * k / d_k))
                cos_val = torch.cos(torch.tensor(theta_ik))
                sin_val = torch.sin(torch.tensor(theta_ik))
                
                R[i, k] = torch.tensor([
                    [cos_val, sin_val],
                    [-sin_val, cos_val]
                ])
        
        self.register_buffer('R', R, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        R_pos = self.R[token_positions]  
        x_pairs = rearrange(x, '... seq_len (d pair) -> ... seq_len d pair', pair=2)
        rotated = einsum(R_pos, x_pairs, '... seq_len d i j, ... seq_len d j -> ... seq_len d i')
        result = rearrange(rotated, '... seq_len d pair -> ... seq_len (d pair)')
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len:int, device=None):
        super().__init__()
        cos_val = torch.zeros(max_seq_len, d_k//2, device=device)
        sin_val = torch.zeros(max_seq_len, d_k//2, device=device)
        for i in range(max_seq_len):
            for k in range(d_k // 2):
                theta_ik = i / (theta ** (2 * k / d_k))
                cos_val[i, k] = torch.cos(torch.tensor(theta_ik))
                sin_val[i, k] = torch.sin(torch.tensor(theta_ik))
        self.register_buffer('cos_val', cos_val, persistent=False)
        self.register_buffer('sin_val', sin_val, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin_rot = self.sin_val[token_positions] # ... seq_len, d_k // 2
        cos_rot = self.cos_val[token_positions]
        x_pair = rearrange(x, "... seq_len (d i) -> ... seq_len d i", i=2)
        x_even = x_pair[..., 0] # ... seq_len, d_k//2 
        x_odd = x_pair[..., 1]
        x_rot_even = x_even * cos_rot - x_odd * sin_rot
        x_rot_odd = x_even * sin_rot + x_odd * cos_rot
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)
        return rearrange(x_rot, '... seq_len d i -> ... seq_len (d i)')

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope.forward(in_query_or_key, token_positions)

class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        args = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter( torch.ones( d_model, **args ) )
        self.d_model = d_model
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt( torch.mean( torch.square(x), dim=-1, keepdim=True ) + self.eps )
        res = self.weight * x / rms 
        return res.to(in_dtype)

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
    rms_layer = RMSNorm(d_model, eps)
    rms_layer.load_state_dict({'weight': weights})
    return rms_layer.forward(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError

class Transformer(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff) 
    
    def forward(self, x:torch.Tensor, token_positions):
        y = x + self.attn(self.ln1(x), token_positions)
        z = y + self.ffn(self.ln2(y))
        return z

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer = Transformer(d_model, num_heads, d_ff, max_seq_len, theta)
    transformer.load_state_dict(weights)

    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=in_features.device).expand(batch_size, -1)
    return transformer(in_features, token_positions)

class TranformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, context_length, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Transformer(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    def forward(self, x):
        batch_size, seq_len = x.shape
        token_positions = torch.arange(seq_len).expand(batch_size, -1)
        x = self.token_embeddings(x)
        for transformer in self.layers:
            x = transformer(x, token_positions)
        return  self.lm_head( self.ln_final (x))

# Number of parameters:
# embedding: vocab_size * d_model
# norm: d_model
# causal mt attention with RoPE: 3*d_model*d_model for kvq proj, d_model*d_model for output_proj
# SwiGLU: 3*d_model*d_ff
# Transformer block: (4*d_model**2 + 3*d_model*d_ff + 2*d_model) = n_trans
# Transformer LM: num_layers*n_trans + d_model * vocab_size
# 2b parameters in total

# 4 bytes per single precision floating number (float32) -> 8GB

# Number of matrix multiply FLOP needed for a forward pass
# embedding: indexing of seq_len
# causal mt attention: 4* (2*context_len*d_model*d_model) for projection
# scaled dot product attention: 2 * (2*context_len*context_len*d_model)
# SwiGLU: 3* (2*context_len*d_model*d_ff)
# 4T FLOPs for each forward pass

# The FFN SwiGLU needs most FLOPs

# As model gets larger, FFN takes proportionaly more FLOPs

# With increasing context length, attention takes more FLOPs  

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
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = TranformerLM(vocab_size, d_model, context_length, num_layers, num_heads, d_ff, rope_theta)
    transformer_lm.load_state_dict(weights)
    return transformer_lm(in_indices)

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
    raise NotImplementedError

def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    max_elem, _ = torch.max(in_features, dim=dim, keepdim=True)
    return torch.exp(in_features-max_elem) / torch.sum(torch.exp(in_features-max_elem), dim=dim, keepdim=True)

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
    return softmax(in_features, dim)


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
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError



class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab_i2b = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        vocab_count = len(vocab)
        if special_tokens:
            for special_token in special_tokens:
                encoded_special_token = special_token.encode('utf-8')
                if encoded_special_token not in self.vocab_i2b.values():
                    self.vocab_i2b[vocab_count] = encoded_special_token
                    vocab_count += 1

        self.merges_dict = {merge: i for i, merge in enumerate(merges)}            
        self.vocab_b2i = {v: k for k, v in self.vocab_i2b.items()}
        return 

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]]=None) -> 'Tokenizer':
        vocab = None
        merges = []
        with open(vocab_filepath, 'r', encoding='utf-8') as vf:
            vocab_json = json.load(vf)
            vocab = {int(k): v.encode('utf-8') for k, v in vocab_json.items()}
            # recover the errors from serialization
            vocab.update({ i: bytes([i]) for i in range(256)})

        with open(merges_filepath, 'r', encoding='utf-8') as mf:
            for line in mf:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                t1, t2 = line.split(' ', 1)
                merges.append((t1.encode('utf-8'), t2.encode('utf-8')))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self._encode_pretoken(text)
        self.special_tokens.sort(key=lambda x: (len(x), x), reverse=True)
        escape_tokens = [re.escape(token) for token in self.special_tokens]
        pattern = "(" +  "|".join(escape_tokens) + ")"
        
        res = []
        segments = re.split(pattern, text)
        for segment in segments:
            if segment in self.special_tokens:
                res += [self.vocab_b2i[segment.encode('utf-8')]]
            else:
                res += self._encode_pretoken(segment)
        return res
    
    def _encode_pretoken(self, text: str) -> List[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        res = []
        for match in re.finditer(PAT, text):
            try:
                # the pre-token which should be handled independently
                pre_token_bytes = match.group(0).encode('utf-8')
                pre_token_list = [bytes([b]) for b in pre_token_bytes]

                to_merge = True
                while to_merge:
                    merge_priority = len(self.merges_dict) + 1
                    merge_idx = 0
                    to_merge = False
                    # find which pair to merge next, if any
                    for i in range(len(pre_token_list)-1):
                        bytes_to_merge =(pre_token_list[i], pre_token_list[i+1])
                        if bytes_to_merge in self.merges_dict and self.merges_dict[bytes_to_merge] < merge_priority:
                            merge_priority = self.merges_dict[bytes_to_merge]
                            merge_idx = i
                            to_merge = True
                    if to_merge:
                        pre_token_list = pre_token_list[:merge_idx] + [ pre_token_list[merge_idx] + pre_token_list[merge_idx+1] ] + pre_token_list[merge_idx+2:]
                res += [self.vocab_b2i[b] for b in pre_token_list]
            except:
                print(pre_token_bytes)
                print(pre_token_list)
                raise KeyError
        return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for id in self.encode(text):
                yield id
    
    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab_i2b[id] for id in ids).decode('utf-8', errors='replace')


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
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

    return Tokenizer(vocab, merges, special_tokens)


# from datasets import load_dataset

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    print("downloading")
    dataset = load_dataset("Skylion007/openwebtext", split="train")

    SEP = "<|endoftext|>"  # Or any unique special token you defined
    special_tokens = [SEP]

    def stream_segments(dataset):
        for example in dataset:
            yield example["text"].strip()
    
    content_stream = stream_segments(dataset)

    escape_tokens = [re.escape(token) for token in special_tokens]
    pattern = "|".join(escape_tokens)

    vocab = { i: bytes([i]) for i in range(256)}
    vocab_count = len(vocab)
    for token in special_tokens:
        vocab[vocab_count] = token.encode('utf-8')
        vocab_count += 1
    merges = []

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #  calculate pre-token
    print("start pre-tokenization")
    pre_token_counter = defaultdict(int) 

    chunk_counter = 0
    for chunk in content_stream:
        chunk_counter += 1
        if chunk_counter % 10000 == 0:
            print("processesd chunks " + str(chunk_counter))
        segments = re.split(pattern, chunk)
        for segment in segments:
            for match in re.finditer(PAT, segment):
                pre_token_counter[match.group(0)] += 1
    print("finish pre-tokenization")

    print("start byte counter")
    # convert to tuples of pre-token bytes
    pre_token_bytes_counter = {}
    for pt in pre_token_counter.keys():
        pre_token_bytes_counter[tuple(bytes([b]) for b in pt.encode('utf-8'))] = pre_token_counter[pt]
    print("finish byte counter")

    # intial frequency counter
    freq_counter = defaultdict(int)
    for tokens in pre_token_bytes_counter.keys():
        for pair in zip(tokens, tokens[1:]):
            token_pair = (pair[0], pair[1]) # bytes
            freq_counter[token_pair] += pre_token_bytes_counter[tokens]

    print("finish initialize frequency counter")

    while vocab_count < vocab_size:
        if len(freq_counter) == 0:
            print('all vocab merged')
            break
        max_pair = max(freq_counter.items(), key=lambda x: (x[1], x[0]))
        max_pair_token = max_pair[0]
        merges.append(max_pair_token)
        merged_token = max_pair_token[0] + max_pair_token[1] 
        vocab[vocab_count] = merged_token
        vocab_count += 1
        if (vocab_count % 100) == 0:
            print(str(vocab_count) + " vocab has been processed")
            print(max_pair)
        
        new_counter = defaultdict(int) 
        for tokens in pre_token_bytes_counter.keys():
            tokens_count = pre_token_bytes_counter[tokens]
            new_tokens = []
            i = 0
            ede_token = False
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == max_pair_token[0] and tokens[i+1] == max_pair_token[1]:
                    freq_counter[(tokens[i], tokens[i+1])] -= tokens_count 
                    if len(new_tokens) > 0:
                        freq_counter[(new_tokens[-1], tokens[i])] -= tokens_count
                        freq_counter[(new_tokens[-1], merged_token)] += tokens_count
                    new_tokens.append(merged_token)
                    if i < len(tokens)-2:
                        freq_counter[(tokens[i+1], tokens[i+2])] -= tokens_count
                        freq_counter[(merged_token, tokens[i+2])] += tokens_count
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_counter[tuple(new_tokens)] += pre_token_bytes_counter[tokens]
        pre_token_bytes_counter = new_counter

    return (vocab, merges)
        
def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: os.PathLike,
    merges_path: os.PathLike,
) -> None:
    """
    Dump the vocabulary to JSON and the merges to a plain-text file.

    * vocab.json  →  {"0": "<byte-string>", "1": "...", ...}
    * merges.txt  →  token1␠token2 per line, order preserved
    """
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {str(idx): tok.decode("utf-8", errors="replace") for idx, tok in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(
                f"{left.decode('utf-8', errors='replace')} "
                f"{right.decode('utf-8', errors='replace')}\n"
            )

if __name__=="__main__":
    here = Path(__file__).parent 
    input_path = here /"TinyStories-train.txt"
    vocab_size = 32_000
    special_tokens = ['<|endoftext|>']

    vocab_out   = here / "vocab_openweb.json"
    merges_out  = here / "merges_openweb.txt"

    print(f"Training BPE on {input_path} …")
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    print(f"Saving vocab ➜ {vocab_out}")
    print(f"Saving merges➜ {merges_out}")
    save_vocab_and_merges(vocab, merges, vocab_out, merges_out)
    print("✔ Done.")

             
             
             
             