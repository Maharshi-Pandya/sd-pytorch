import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class SelfAttentionBlock(nn.Module):
    """
    Self Attention for calculating attention between tokens
    of the same sequence

    attention = softmax((Q * K.T) / sqrt(d_k)) * V
    """

    def __init__(
            self, 
            n_heads: int, 
            d_embed: int,
            # whether to add bias for input/output projection matrix
            in_proj_bias: bool = True,
            out_proj_bias: bool = True
        ):
        super(SelfAttentionBlock, self).__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.dim_head = d_embed // self.n_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # shape x: (n, seq_len, d_embed)

        inp: torch.Tensor = self.in_proj(x)
        q, k, v = inp.chunk(3, dim=-1)      # 3 times (bs, seq, d_embed)

        # (bs, seq, d_embed) -> (bs, n_heads, seq, d_embed // n_heads)
        q = rearrange(q, "bs sl (nh dh) -> bs nh sl dh", nh = self.n_heads)
        k = rearrange(k, "bs sl (nh dh) -> bs nh sl dh", nh = self.n_heads)
        v = rearrange(v, "bs sl (nh dh) -> bs nh sl dh", nh = self.n_heads)


        # calculate attention    
        kT = rearrange(k, "bs nh sl dh -> bs nh dh sl")
     
        weight = q @ kT     # (bs, nh, sl, sl)
        
        # attention for tokens preceding current token only
        # upper triangle sets to -inf while calculating softmax
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.dim_head)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v         # convert from (bs, nh, sl, dh) to (bs, sl, d_embed)
        output = rearrange(output, "bs nh sl dh -> bs sl (nh dh)")

        # self attention
        return self.out_proj(output)
