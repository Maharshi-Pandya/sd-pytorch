import torch
import torch.nn as nn

from einops import rearrange
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


# starting in and out channels for Conv2D
# out_channels is number of "filters" applied to input signal
s_inc = 3
s_outc = 128

# Size of the encoder's bottle neck
enc_bottleneck_size = 8

# kernel size 3 x 3
ks = 3


# inherits from sequential, so process input one by one
class VAE_Encoder(nn.Sequential):
    """
    This model is responsible for encoding images to create "representations"
    of those images in the latent space. It learns a "latent space" which is
    a multivariate Gaussian distribution. So it returns a Mean (mu) and a Variance (sigma)
    of the distribution.

    We learn Mean and Variance to sample from the distribution of the images.
    Also, from that Mean and Variance, we sample using a Gaussian (0, 1).
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

            # (bs, s_outc, H / 2, W / 2) -> (bs, s_outc * 2, H / 2, W / 2)
            VAE_ResidualBlock(s_outc, s_outc * 2),

            # (bs, s_outc * 2, H / 2, W / 2) -> (bs, s_outc * 2, H / 2, W / 2)
            VAE_ResidualBlock(s_outc * 2, s_outc * 2),

            # again H and W halved
            # (bs, s_outc * 2, H / 2, W / 2) -> (bs, s_outc * 2, H / 4, W / 4)
            nn.Conv2d(s_outc * 2, s_outc * 2, kernel_size=ks, stride=2),

            # (bs, s_outc * 2, H / 4, W / 4) -> (bs, s_outc * 2 * 2, H / 4, W / 4)
            VAE_ResidualBlock(s_outc * 2, s_outc * 2 * 2),

            # (bs, s_outc * 2 * 2, H / 4, W / 4) -> (bs, s_outc * 2 * 2, H / 4, W / 4)
            VAE_ResidualBlock(s_outc * 2 * 2, s_outc * 2 * 2),

            # H and W halved again
            # (bs, s_outc * 2 * 2, H / 4, W / 4) -> (bs, s_outc * 2 * 2, H / 8, W / 8)
            nn.Conv2d(s_outc * 2 * 2, s_outc * 2 * 2, kernel_size=ks, stride=2),

            # (512, 512) x 3
            VAE_ResidualBlock(s_outc * 2 * 2, s_outc * 2 * 2),
            VAE_ResidualBlock(s_outc * 2 * 2, s_outc * 2 * 2),
            VAE_ResidualBlock(s_outc * 2 * 2, s_outc * 2 * 2),

            # attention block for "relating" the embeddings
            # see attention mechanisms link in README
            VAE_AttentionBlock(s_outc * 2 * 2),

            VAE_ResidualBlock(s_outc * 2 * 2, s_outc * 2 * 2),

            # Normalize in groups
            # separate normalization happens in different groups
            nn.GroupNorm(32, s_outc * 2 * 2,),

            # No reason to choose Silu over Relu
            # Silu works better in practice
            nn.SiLU(),

            # size remains same (this is "encoding")
            nn.Conv2d(s_outc * 2 * 2, enc_bottleneck_size, kernel_size=ks, padding=1),
            # note kernel size is 1
            nn.Conv2d(enc_bottleneck_size, enc_bottleneck_size, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (bs, c, h, w)
        # noise: (bs, 8, h / 8, w / 8)

        # for all conv2d having a stride,
        # we need to apply padding to the image
        # but only on right and down (asymetrical padding)
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))      # (left, right, up, down)
            
            x = module(x)

        # the mean and log_variance tensors
        # chunk the output tensor from the last layer
        # of the sequential model
        
        mu, log_variance = rearrange(x, "n (c s) h w -> s n c h w", s = 2)
        log_variance = torch.clamp(log_variance, -30, 20)

        sigma = log_variance.exp()
        std = sigma.sqrt()

        # if we sample from z = N(0, 1) how do we sample from x = N(mean, variance)
        # Answer: reverse of normalization
        # x = mean + z * std
        # we have the noise coming from N(0, 1)
        # lastly we scale the output

        x = mu + std * noise
        x *= 0.18215    # magic number?

        return x
