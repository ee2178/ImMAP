# -*- coding: utf-8 -*-
"""
SBUnet -- the I2SB paper's regressor architecture, adapted to this repo's calling convention.

I2SB does not use a bespoke network: it takes ADM / guided-diffusion's UNet unchanged and only
decides WHAT to feed its timestep input (i2sb/network.py, i2sb/runner.py). This file is a compact
port of that UNet -- same blocks, same conditioning path -- so the baseline is architecturally the
paper's rather than "a UNet".

HOW I2SB ENCODES THE NOISE LEVEL (the part worth being faithful about)

    runner.py:86   noise_levels = torch.linspace(opt.t0, opt.T, opt.interval) * opt.interval
    network.py:47  t = self.noise_levels[steps]
    network.py:51  return self.diffusion_model(torch.cat([x, cond], 1), t)

Despite the name, `noise_levels` is NOT a noise standard deviation. With t0=1e-4, T=1, interval=1000
it is `linspace(1e-4, 1, 1000) * 1000`, i.e. approximately the STEP INDEX 0..1000 -- the scale ADM's
pretrained checkpoints were trained on. So the network is conditioned on the discrete bridge
position, not on sigma, and the schedule enters only through which step you are at.

    step -> sinusoidal timestep_embedding(model_channels) -> Linear-SiLU-Linear
         -> per-ResBlock FiLM (scale, shift) on the GroupNorm output          [use_scale_shift_norm]

This matters here because `sb.base.predict_x0` passes SIGMA, not the step. We recover the step by
inverting std_fwd through BridgeScheduleMixin -- the same lookup SBCDLNet uses -- and then follow
I2SB exactly. That inversion is why this net carries its own copy of the schedule and why
`assert_schedule_matches` is inherited: model.params (kind/tau/n_points/beta_max) MUST equal
cfg["i2sb"], or every step conditions on the wrong bridge position.

Conditioning contrasts are concatenated onto the input channel, exactly as I2SB's `cond_x1` does,
so C = target_channels + len(cond_idx) and nothing else changes between the T1-only and the
all-contrast run.

WHAT IS DELIBERATELY NOT PORTED. fp16 conversion, gradient checkpointing, class conditioning and
the AttentionPool head are all ADM machinery I2SB does not exercise for this task. The blocks that
ARE used -- ResBlock with scale-shift norm, QKV self-attention in the legacy head order, zero-init
output projections, GroupNorm(32)+SiLU throughout -- are ported faithfully.

Calling convention (matches every other regressor in the repo):

    x0_hat, _ = net(y, E=Identity(), sigma=std_fwd)      # y = cat([x_t, cond], dim=1)

`E` is accepted and ignored (pure translation has no forward operator). The second return value is
None -- there is no sparse code to hand back; it exists so `out, _ = net(...)` in predict_x0 works
for unrolled and non-unrolled regressors alike.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sb_schedule import BridgeScheduleMixin
from operators.padding import calc_pad_2d
from operators.padding import unpad


# ---------------------------------------------------------------------------
# ADM primitives (guided_diffusion/nn.py)
# ---------------------------------------------------------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings. Ported from guided_diffusion/nn.py unchanged, including
    the cos-then-sin concatenation order (a different order is a different -- untrained -- basis,
    so it is not a free choice if you ever want to warm-start from an ADM checkpoint)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def zero_module(module):
    """Zero out a module's parameters. ADM does this on every residual/attention OUTPUT
    projection so each block starts as an exact identity and the net begins training as a stack
    of skip connections."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels, groups=32):
    """GroupNorm(32) as in ADM, falling back to fewer groups for narrow layers."""
    g = math.gcd(groups, channels)
    return nn.GroupNorm(g if g > 0 else 1, channels)


class TimestepBlock(nn.Module):
    """A module whose forward takes the timestep embedding as a second argument."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that forwards `emb` only to the children that want it."""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=1)
        else:
            assert self.out_channels == channels
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    """ADM residual block. With `use_scale_shift_norm` the timestep embedding enters as FiLM on
    the second GroupNorm -- h <- norm(h) * (1 + scale) + shift -- which is the conditioning path
    I2SB's checkpoints use. Without it the embedding is simply added, the DDPM original."""

    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, use_scale_shift_norm=True, up=False, down=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1))

        self.updown = up or down
        if up:
            self.h_upd, self.x_upd = Upsample(channels, False), Upsample(channels, False)
        elif down:
            self.h_upd, self.x_upd = Downsample(channels, False), Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels,
                      2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(
            normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_conv(self.h_upd(in_rest(x)))
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    """ADM's default attention head order (use_new_attention_order=False): qkv is split
    head-first, so the layout is (B*heads, 3*ch, T)."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))          # split across q and k, as in ADM
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Spatial self-attention over the flattened feature map, with a zero-init output
    projection so the block starts as the identity."""

    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super().__init__()
        if num_head_channels > 0:
            if channels % num_head_channels != 0:
                raise ValueError(
                    f"AttentionBlock: channels={channels} is not divisible by "
                    f"num_head_channels={num_head_channels}")
            num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.attention(self.qkv(self.norm(x)))
        return (x + self.proj_out(h)).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# the net
# ---------------------------------------------------------------------------
class SBUnet(BridgeScheduleMixin, nn.Module):
    """ADM UNet conditioned on the bridge step, in this repo's regressor interface.

    Parameters
    ----------
    C                    input width = target_channels + len(cond_idx). Conditioning contrasts
                         are concatenated onto x_t, exactly as I2SB's cond_x1 does.
    out_channels         predicted channels; 1 for a single-contrast target. (ADM predicts
                         2*C with learn_sigma; I2SB regresses x0 only, so this stays 1.)
    model_channels       base width. Channel widths are model_channels * channel_mult[level].
    num_res_blocks       ResBlocks per resolution level.
    channel_mult         width multipliers, one per level; len-1 downsamples total.
    attention_resolutions
                         SPATIAL resolutions (e.g. [16, 8]) at which to insert self-attention,
                         interpreted relative to `image_size` the way ADM's create_model does
                         (ds = image_size // res). It selects LEVELS at build time, so the net
                         still runs at any input size -- a val image larger than the training
                         crop just means attention acts on a bigger map.
    image_size           only used to turn `attention_resolutions` into levels. Set it to the
                         training crop_size.
    num_head_channels    channels per attention head (ADM's preferred knob); -1 to use num_heads.
    use_scale_shift_norm FiLM conditioning (True = what I2SB's checkpoints use).
    resblock_updown      use ResBlocks for up/downsampling instead of conv/interpolate.
    kind, tau, n_points, beta_max
                         the bridge schedule. MUST match cfg["i2sb"] -- sigma is inverted through
                         it to recover the step the network is conditioned on.
    t0, T                endpoints of I2SB's `noise_levels` ramp (runner.py:86). Leave at the
                         paper values; they set the numeric scale the embedding sees.
    """

    def __init__(self, C=1, out_channels=1, model_channels=64, num_res_blocks=2,
                 channel_mult=(1, 2, 3, 4), attention_resolutions=(16,),
                 image_size=192, num_heads=4, num_head_channels=32, dropout=0.0,
                 use_scale_shift_norm=True, resblock_updown=False, conv_resample=True,
                 kind="i2sb", tau=0.19, n_points=1000, beta_max=0.3, t0=1.0e-4, T=1.0):
        super().__init__()

        self.C = int(C)                       # train_i2sb asserts this against 1 + len(cond_idx)
        self.out_channels = int(out_channels)
        self.model_channels = int(model_channels)
        channel_mult = tuple(int(m) for m in channel_mult)
        self.channel_mult = channel_mult
        # every level halves the map, so the input must be divisible by this; forward() pads
        self.size_divisor = 2 ** (len(channel_mult) - 1)

        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)
        # I2SB runner.py:86 -- linspace(t0, T, interval) * interval, i.e. ~the step index. This
        # is what goes into the sinusoidal embedding; it is NOT a noise std despite the name.
        self.register_buffer(
            "noise_levels",
            torch.linspace(float(t0), float(T), int(n_points)) * float(n_points))

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim))

        # ADM's create_model takes attention_resolutions as image resolutions and converts them
        # to downsample factors against image_size; do the same so configs read the same way.
        attn_ds = {int(image_size) // int(res) for res in attention_resolutions}
        # A resolution deeper than the network goes silently yields NO attention block: ds only
        # reaches 2**(len(channel_mult)-1). Catch the mismatch instead of training a config that
        # says "attention at 8" and has none.
        reachable = {2 ** i for i in range(len(channel_mult))}
        unreachable = sorted(d for d in attn_ds if d not in reachable)
        if unreachable:
            raise ValueError(
                f"attention_resolutions {list(attention_resolutions)} with image_size="
                f"{image_size} asks for attention at downsample factor(s) {unreachable}, but "
                f"channel_mult={channel_mult} only reaches {sorted(reachable)}. With this "
                f"image_size the usable resolutions are "
                f"{sorted(int(image_size) // d for d in reachable)}.")

        ch = model_channels
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(self.C, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout,
                                   out_channels=mult * model_channels,
                                   use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attn_ds:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout, out_channels=ch,
                             use_scale_shift_norm=use_scale_shift_norm, down=True)
                    if resblock_updown else Downsample(ch, conv_resample, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, num_heads, num_head_channels),
            ResBlock(ch, time_embed_dim, dropout, use_scale_shift_norm=use_scale_shift_norm))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout,
                                   out_channels=model_channels * mult,
                                   use_scale_shift_norm=use_scale_shift_norm)]
                ch = model_channels * mult
                if ds in attn_ds:
                    layers.append(AttentionBlock(ch, num_heads, num_head_channels))
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=ch,
                                 use_scale_shift_norm=use_scale_shift_norm, up=True)
                        if resblock_updown else Upsample(ch, conv_resample, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1)))

    # -----------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, step=None):
        """`y = cat([x_t, cond], dim=1)`, the tensor sb.base.predict_x0 builds. `E` is accepted
        for signature parity with the repo's denoisers and ignored. Returns (x0_hat, None).

        Pass `step` to condition on the bridge index directly; otherwise `sigma` (the schedule's
        std_fwd) is inverted through this net's own schedule tables."""
        if y.shape[1] != self.C:
            raise ValueError(f"SBUnet built for C={self.C} input channels, got {y.shape[1]}")

        if step is None:
            if sigma is None:
                raise ValueError(
                    "SBUnet needs the bridge position: pass `sigma` (the schedule's std_fwd, as "
                    "sb.base.predict_x0 does) or `step`.")
            step = self._step_from_sigma(sigma)
        step = torch.as_tensor(step, device=y.device).reshape(-1).long()
        if step.numel() == 1:
            step = step.expand(y.shape[0])
        elif step.numel() != y.shape[0]:
            raise ValueError(f"got {step.numel()} bridge positions for a batch of {y.shape[0]}")

        t = self.noise_levels[step.clamp(0, self.noise_levels.shape[0] - 1)]
        emb = self.time_embed(timestep_embedding(t, self.model_channels))

        # The encoder halves the map once per level; BraTS is 240 native (divisible) but a crop
        # need not be, and the skip concatenations require exact shape agreement.
        pad = calc_pad_2d(*y.shape[2:], self.size_divisor)
        h = F.pad(y, pad, mode="reflect")

        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb)
        return unpad(self.out(h), pad), None
