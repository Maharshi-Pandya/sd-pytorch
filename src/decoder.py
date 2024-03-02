import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from attention import SelfAttention


# Number of groups for GroupNorm along features dimension
n_groups = 32


class VAE_AttentionBlock(nn.Module):
    """
    This block is responsible for calculating the "attention"
    between the pixels of an image/latent in order to relate them
    just like in Language Modelling.
    """

    def __init__(self, channels: int):
        super(VAE_AttentionBlock, self).__init__()

        self.gn = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)     # n_heads, n_features

    def forward(self, x: torch.Tensor):
        # shape x: (n, c, h, w)
        height = x.shape[2]

        # skip connection/residual
        residue = x

        # 1. Convert the image/latent to a sequence of tokens (pixels) i.e. H * W
        # 2. Transpose last two dimensions for attention i.e. (seq_len, n_features)
        x = rearrange(x, "n c h w -> n (h w) c")

        x = self.attention(x)                                   # (n, (h w), c)

        x = rearrange(x, "n (h w) c -> n c h w", h = height)    # (n, c, h, w)
        x += residue                                            # (n, c, h, w)

        return x


class VAE_ResidualBlock(nn.Module):
    pass
