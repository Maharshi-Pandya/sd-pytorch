import torch
import torch.nn as nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


# starting in and out channels for Conv2D
# out_channels is number of "filters" applied to input signal
s_inc = 3
s_outc = 128

# kernel size 3 x 3
ks = 3


# inherits from sequential, so process input one by one
class VAE_Encoder(nn.Sequential):
    """
    This model is responsible for encoding images to create "representations"
    of those images in the latent space.
    This is kind of similar to word embeddings where similar words
    are encoded closer in the "embedding" space.
    """
    def __init__(self):
        super().__init__(
            # since padding is 1
            # (bs, s_inc, H, W) -> (bs, s_outc, H, W)
            nn.Conv2d(s_inc, s_outc, kernel_size=ks, padding=1),

            # (bs, s_outc, H, W) -> (bs, s_outc, H, W)
            VAE_ResidualBlock(s_outc, s_outc),

            # (bs, s_outc, H, W) -> (bs, s_outc, H, W)
            VAE_ResidualBlock(s_outc, s_outc),

            # since stride is 2 and no padding
            # (bs, s_outc, H, W) -> (bs, s_outc, H / 2, W / 2)
            nn.Conv2d(s_outc, s_outc, kernel_size=ks, stride=2),

            # (bs, s_outc, H, W) -> (bs, s_outc * 2, H, W)
            VAE_ResidualBlock(s_outc, s_outc * 2),

            # (bs, s_outc * 2, H, W) -> (bs, s_outc * 2, H, W)
            VAE_ResidualBlock(s_outc * 2, s_outc * 2),

            # again H and W halved
            # (bs, s_outc * 2, H, W) -> (bs, s_outc * 2, H / 4, W / 4)
            nn.Conv2d(s_outc * 2, s_outc * 2, kernel_size=ks, stride=2)
        )
