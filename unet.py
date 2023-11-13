import math
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import einsum, nn

# class SelfAttention(nn.Module):
#     def __init__(self, channels, size):
#         super(SelfAttention, self).__init__()
#         self.channels = channels
#         self.size = size
#         self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x):
#         x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(
#             -1, self.channels, self.size, self.size
#         )


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
#         super().__init__()
#         self.residual = residual
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, mid_channels),
#             nn.GELU(),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.GroupNorm(1, out_channels),
#         )

#     def forward(self, x):
#         if self.residual:
#             return F.gelu(x + self.double_conv(x))
#         else:
#             return self.double_conv(x)


# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_dim, out_channels),
#         )

#     def forward(self, x, t):
#         x = self.maxpool_conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb


# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, emb_dim=256):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.conv = nn.Sequential(
#             DoubleConv(in_channels, in_channels, residual=True),
#             DoubleConv(in_channels, out_channels, in_channels // 2),
#         )

#         self.emb_layer = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(emb_dim, out_channels),
#         )

#     def forward(self, x, skip_x, t):
#         x = self.up(x)
#         x = torch.cat([skip_x, x], dim=1)
#         x = self.conv(x)
#         emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
#         return x + emb


# class UNet(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8)

#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)

#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 16)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 32)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)

#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc

#     def forward(self, x, t):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)

#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)

#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)

#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output


######################################################################################################################################


# def sinusoidal_embedding(n, d):
#     # Returns the standard positional embedding
#     embedding = torch.zeros(n, d)  # Shape: (n, d)
#     wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])  # Shape: (d,)
#     wk = wk.reshape((1, d))  # Shape: (1, d)
#     t = torch.arange(n).reshape((n, 1))  # Shape: (n, 1)
#     embedding[:, ::2] = torch.sin(t * wk[:, ::2])  # Shape: (n, d)
#     embedding[:, 1::2] = torch.cos(t * wk[:, ::2])  # Shape: (n, d)
#     return embedding


# class MyBlock(nn.Module):
#     def __init__(
#         self,
#         shape,
#         in_c,
#         out_c,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         activation=None,
#         normalize=True,
#     ):
#         super(MyBlock, self).__init__()
#         self.ln = nn.LayerNorm(shape)
#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
#         self.activation = nn.SiLU() if activation is None else activation
#         self.normalize = normalize

#     def forward(self, x):
#         out = self.ln(x) if self.normalize else x
#         out = self.conv1(out)
#         out = self.activation(out)
#         out = self.conv2(out)
#         out = self.activation(out)
#         return out


# class UNet(nn.Module):
#     def __init__(self, n_steps=1000, time_emb_dim=100):
#         super(UNet, self).__init__()

#         # Sinusoidal embedding
#         self.time_embed = nn.Embedding(
#             num_embeddings=n_steps, embedding_dim=time_emb_dim
#         )
#         self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
#         self.time_embed.requires_grad_(False)

#         # First half
#         self.te1 = self._make_te(time_emb_dim, 1)
#         self.b1 = nn.Sequential(
#             MyBlock((1, 28, 28), 1, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10),
#         )
#         self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

#         self.te2 = self._make_te(time_emb_dim, 10)
#         self.b2 = nn.Sequential(
#             MyBlock((10, 14, 14), 10, 20),
#             MyBlock((20, 14, 14), 20, 20),
#             MyBlock((20, 14, 14), 20, 20),
#         )
#         self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

#         self.te3 = self._make_te(time_emb_dim, 20)
#         self.b3 = nn.Sequential(
#             MyBlock((20, 7, 7), 20, 40),
#             MyBlock((40, 7, 7), 40, 40),
#             MyBlock((40, 7, 7), 40, 40),
#         )
#         self.down3 = nn.Sequential(
#             nn.Conv2d(40, 40, 2, 1), nn.SiLU(), nn.Conv2d(40, 40, 4, 2, 1)
#         )

#         # Bottleneck
#         self.te_mid = self._make_te(time_emb_dim, 40)
#         self.b_mid = nn.Sequential(
#             MyBlock((40, 3, 3), 40, 20),
#             MyBlock((20, 3, 3), 20, 20),
#             MyBlock((20, 3, 3), 20, 40),
#         )

#         # Second half
#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(40, 40, 4, 2, 1),
#             nn.SiLU(),
#             nn.ConvTranspose2d(40, 40, 2, 1),
#         )

#         self.te4 = self._make_te(time_emb_dim, 80)
#         self.b4 = nn.Sequential(
#             MyBlock((80, 7, 7), 80, 40),
#             MyBlock((40, 7, 7), 40, 20),
#             MyBlock((20, 7, 7), 20, 20),
#         )

#         self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
#         self.te5 = self._make_te(time_emb_dim, 40)
#         self.b5 = nn.Sequential(
#             MyBlock((40, 14, 14), 40, 20),
#             MyBlock((20, 14, 14), 20, 10),
#             MyBlock((10, 14, 14), 10, 10),
#         )

#         self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
#         self.te_out = self._make_te(time_emb_dim, 20)
#         self.b_out = nn.Sequential(
#             MyBlock((20, 28, 28), 20, 10),
#             MyBlock((10, 28, 28), 10, 10),
#             MyBlock((10, 28, 28), 10, 10, normalize=False),
#         )

#         self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

#     def forward(self, x, t):
#         # x is (N, 2, 28, 28) (image with positinoal embedding stacked on channel dimension)
#         t = self.time_embed(t)
#         n = len(x)
#         out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # N, 10, 28, 28
#         out2 = self.b2(
#             self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1)
#         )  # N, 20, 14, 14
#         out3 = self.b3(
#             self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1)
#         )  # N, 40, 7, 7

#         out_mid = self.b_mid(
#             self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1)
#         )  # N, 40, 3, 3

#         out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # N, 80, 7, 7
#         out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # N, 20, 7, 7

#         out5 = torch.cat((out2, self.up2(out4)), dim=1)  # N, 40, 14, 14
#         out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # N, 10, 14, 14

#         out = torch.cat((out1, self.up3(out5)), dim=1)  # N, 20, 28, 28
#         out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # N, 1, 28, 28

#         out = self.conv_out(out)  # N, 1, 28, 28

#         return out

#     def _make_te(self, dim_in, dim_out):
#         return nn.Sequential(
#             nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
#         )


######################################################################################################################################


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, 1, padding=0
        )  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
