# -*- coding: utf-8 -*-
"""
Configurable 2D U-Net for CCL pretraining and contrast synthesis.

This is a faithful, config-driven port of the TF reference `modelObj`
(Umapathy et al., Med. Phys. 2024 — utils/model_utils.py in the
multi-contrast-contrastive-learning / CCL-Synthetis repos). Every knob that
differs between the reference and a "plain" U-Net is exposed in the config:

  * norm            : "group" (reference, num_groups = ch // group_size) | "batch" | ...
  * act             : "relu" (pretrain) | "tanh" (synthesis) | ...   (see models/norm.py)
  * convs_per_block : convs per encoder/decoder block (reference = 2)
  * num_pool_layers : downsampling depth (reference encoder = 5 pools, chans 16..128)
  * up_conv         : reference decoder upsamples then applies a learned conv->norm->act
                      BEFORE the skip concat; set False for a parameter-free upsample.
  * head            : output head for synthesis. "conv" = reference
                      (Conv3x3->norm->act->Conv1x1, no bias); "simple" = a single 1x1 conv.

Encoder: ConvBlock(in->16) at full res, then `num_pool_layers` MaxPool-then-ConvBlock
stages with channels min(chans*2**i, max_chans). Decoder mirrors back to `chans`,
upsampling via F.interpolate to the skip's size (so non-power-of-2 inputs are fine)
and concatenating the skip. The decoder returns FULL resolution, so
downsample_factor = 1 and CCL runs in unfold mode (loss: partial_decoder=False,
patch_size=P).

Transfer: self.backbone (Unet2D) is what you keep; the projection head is discarded.
For synthesis, instantiate Unet2D with out_chans set, load the pretrained backbone
(training/synthesis.py does a shape-tolerant partial load), and train the new head.
Keep `norm`, `convs_per_block`, `num_pool_layers`, `up_conv`, `group_size` IDENTICAL
between the CCL and synthesis configs so the backbone keys line up; `act` may differ
(activations are parameter-free).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import build_norm, build_activation


class ConvBlock(nn.Module):
    """n_convs x (Conv3x3 padded -> Norm -> Act). Reference uses n_convs=2."""
    def __init__(self, in_ch, out_ch, n_convs=2, norm="group", act="relu", group_size=4):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(n_convs):
            layers += [nn.Conv2d(c, out_ch, 3, padding=1, bias=False),
                       build_norm(norm, out_ch, group_size),
                       build_activation(act)]
            c = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample -> (optional conv->norm->act) -> concat skip -> conv block.

    With up_conv=True this matches the reference decoder, which learns a conv on
    the upsampled feature map before the skip concat; with up_conv=False the
    upsample is parameter-free (the previous ImMAP behavior)."""
    def __init__(self, in_ch, skip_ch, out_ch, n_convs=2, norm="group", act="relu",
                 group_size=4, up_mode="nearest", up_conv=True):
        super().__init__()
        self.up_mode = up_mode
        if up_conv:
            # reference learns a conv on the upsampled map before the skip concat
            # (it uses a 2x2 'same' conv; 3x3 is the warning-free, strict-superset
            # equivalent and avoids a zero-padded input copy each forward)
            self.pre = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                build_norm(norm, out_ch, group_size),
                build_activation(act),
            )
            concat_ch = out_ch + skip_ch
        else:
            self.pre = None
            concat_ch = in_ch + skip_ch
        self.conv = ConvBlock(concat_ch, out_ch, n_convs, norm, act, group_size)

    def forward(self, x, skip):
        if self.up_mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=skip.shape[-2:], mode=self.up_mode, align_corners=False)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode=self.up_mode)
        if self.pre is not None:
            x = self.pre(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Head(nn.Module):
    """Reference synthesis head: Conv3x3 -> norm -> act -> Conv1x1 (linear, no bias)."""
    def __init__(self, in_ch, out_ch, norm="group", act="relu", group_size=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            build_norm(norm, in_ch, group_size),
            build_activation(act),
        )
        self.out = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.out(self.body(x))


class Unet2D(nn.Module):
    """Configurable U-Net backbone (see module docstring)."""
    def __init__(self, in_chans, chans=16, num_pool_layers=5, max_chans=128,
                 out_chans=None, convs_per_block=2, norm="group", act="relu",
                 group_size=4, up_mode="nearest", up_conv=True, head="conv"):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        chs = [min(chans * (2 ** i), max_chans) for i in range(num_pool_layers + 1)]  # 16,32,64,128,128,...

        self.inc = ConvBlock(in_chans, chs[0], convs_per_block, norm, act, group_size)
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList(
            [ConvBlock(chs[i - 1], chs[i], convs_per_block, norm, act, group_size)
             for i in range(1, num_pool_layers + 1)])

        ups = []
        for j in range(num_pool_layers):
            t = num_pool_layers - 1 - j                 # encoder level being matched
            ups.append(Up(in_ch=chs[t + 1], skip_ch=chs[t], out_ch=chs[t],
                          n_convs=convs_per_block, norm=norm, act=act,
                          group_size=group_size, up_mode=up_mode, up_conv=up_conv))
        self.ups = nn.ModuleList(ups)

        self.out_channels = chs[0]
        if out_chans is None:
            self.final = None
        elif head == "conv":
            self.final = Head(chs[0], out_chans, norm, act, group_size)
        else:
            self.final = nn.Conv2d(chs[0], out_chans, 1)

    def forward(self, x):
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(self.pool(feats[-1])))
        cur = feats[self.num_pool_layers]               # bottleneck
        for j, up in enumerate(self.ups):
            t = self.num_pool_layers - 1 - j
            cur = up(cur, feats[t])
        if self.final is not None:
            cur = self.final(cur)
        return cur


class ProjectionHead(nn.Module):
    """1x1 Conv -> norm -> act -> 1x1 Conv, to the contrastive embedding space."""
    def __init__(self, in_ch, proj_dim, norm="group", act="relu", group_size=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            build_norm(norm, in_ch, group_size),
            build_activation(act),
            nn.Conv2d(in_ch, proj_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class CCLNet(nn.Module):
    """Configurable U-Net backbone (+ optional projection head) for constrained
    contrastive pretraining. proj_dim=None runs CCL directly on the backbone's
    decoder features (the reference setup; add_PH=False), matching the channel
    count carried into synthesis exactly."""
    def __init__(self, in_chans=3, proj_dim=None, chans=16, num_pool_layers=5,
                 max_chans=128, convs_per_block=2, norm="group", act="relu",
                 group_size=4, up_mode="nearest", up_conv=True):
        super().__init__()
        self.backbone = Unet2D(in_chans=in_chans, chans=chans, num_pool_layers=num_pool_layers,
                               max_chans=max_chans, out_chans=None,
                               convs_per_block=convs_per_block, norm=norm, act=act,
                               group_size=group_size, up_mode=up_mode, up_conv=up_conv)
        self.proj = (ProjectionHead(self.backbone.out_channels, proj_dim, norm, act, group_size)
                     if proj_dim else None)
        self._ds = 1                                    # full-resolution decoder

    @property
    def downsample_factor(self):
        return self._ds

    def forward(self, x):
        f = self.backbone(x)
        return self.proj(f) if self.proj is not None else f

    def backbone_state_dict(self):
        """U-Net weights to carry into downstream finetuning (projection head dropped)."""
        return self.backbone.state_dict()


if __name__ == "__main__":
    # Smoke test: CCL backbone -> synthesis transfer with the reference setup.
    net = CCLNet(in_chans=3, proj_dim=None, chans=16, num_pool_layers=5,
                 norm="group", act="relu", convs_per_block=2, up_conv=True)
    x = torch.randn(2, 3, 240, 240)
    y = net(x)
    print("downsample_factor:", net.downsample_factor, "| CCL feat out:", tuple(y.shape))
    y.sum().backward()

    synth = Unet2D(in_chans=3, out_chans=1, chans=16, num_pool_layers=5,
                   norm="group", act="tanh", convs_per_block=2, up_conv=True, head="conv")
    bb = net.backbone_state_dict()
    own = synth.state_dict()
    loaded = sum(1 for k, v in bb.items() if k in own and own[k].shape == v.shape)
    print(f"backbone keys transferable to synthesis: {loaded}/{len(bb)}")
    print("synth out:", tuple(synth(x).shape))
