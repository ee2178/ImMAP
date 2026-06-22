# -*- coding: utf-8 -*-
"""
CCL pretraining model following the customized 2D U-Net in the reference figure:

  * Conv block      : (Conv 3x3, padded) -> BatchNorm -> ReLU
  * Encoder         : 16 -> 32 -> 64 -> 128 -> 128, downsampled by 2D MaxPool
                      (channels capped at max_chans=128, so the bottleneck is 128)
  * Decoder         : 2D Upsampling -> concat skip -> Conv block, mirroring back to 16
  * Output feature  : full-resolution map at `chans` (16) channels

The decoder returns full resolution, so downsample_factor = 1 and CCL runs in unfold
mode (loss: partial_decoder=False, patch_size=P). forward(x) -> (B, proj_dim, H, W).

Transfer: self.backbone (Unet2D) is what you keep; the projection head is discarded.
For downstream tasks (e.g. contrast synthesis) instantiate Unet2D with out_chans set,
load the pretrained backbone weights, and train the new final 1x1 conv.

Self-contained: no dependency on your fastmri Unet (this is a different architecture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """n_convs x (Conv3x3 padded -> BatchNorm -> ReLU). Figure shows a single C-B-R;
    set convs_per_block=2 for the common stronger variant."""
    def __init__(self, in_ch, out_ch, n_convs=1):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(n_convs):
            layers += [nn.Conv2d(c, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace=True)]
            c = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """2D upsampling -> concat skip -> conv block."""
    def __init__(self, in_ch, skip_ch, out_ch, n_convs=1, up_mode="nearest"):
        super().__init__()
        self.up_mode = up_mode
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, n_convs)

    def forward(self, x, skip):
        if self.up_mode in ("bilinear", "bicubic"):
            x = F.interpolate(x, size=skip.shape[-2:], mode=self.up_mode, align_corners=False)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode=self.up_mode)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Unet2D(nn.Module):
    """Customized U-Net from the reference figure (BatchNorm/ReLU, upsample+concat)."""
    def __init__(self, in_chans, chans=16, num_pool_layers=4, max_chans=128,
                 out_chans=None, convs_per_block=1, up_mode="nearest"):
        super().__init__()
        self.num_pool_layers = num_pool_layers
        chs = [min(chans * (2 ** i), max_chans) for i in range(num_pool_layers + 1)]  # 16,32,64,128,128

        self.inc = ConvBlock(in_chans, chs[0], convs_per_block)
        self.pool = nn.MaxPool2d(2)
        self.downs = nn.ModuleList(
            [ConvBlock(chs[i - 1], chs[i], convs_per_block) for i in range(1, num_pool_layers + 1)])

        ups = []
        for j in range(num_pool_layers):
            t = num_pool_layers - 1 - j                 # encoder level being matched
            ups.append(Up(in_ch=chs[t + 1], skip_ch=chs[t], out_ch=chs[t],
                          n_convs=convs_per_block, up_mode=up_mode))
        self.ups = nn.ModuleList(ups)

        self.out_channels = chs[0]
        self.final = nn.Conv2d(chs[0], out_chans, 1) if out_chans is not None else None

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
    """1x1 Conv -> BN -> ReLU -> 1x1 Conv, to the contrastive embedding space."""
    def __init__(self, in_ch, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, proj_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class CCLNet(nn.Module):
    """Customized U-Net backbone + projection head for constrained contrastive pretraining."""
    def __init__(self, in_chans=4, proj_dim=64, chans=16, num_pool_layers=4,
                 max_chans=128, convs_per_block=1, up_mode="nearest"):
        super().__init__()
        self.backbone = Unet2D(in_chans=in_chans, chans=chans, num_pool_layers=num_pool_layers,
                               max_chans=max_chans, out_chans=None,
                               convs_per_block=convs_per_block, up_mode=up_mode)
        self.proj = ProjectionHead(self.backbone.out_channels, proj_dim)
        self._ds = 1                                    # full-resolution decoder

    @property
    def downsample_factor(self):
        return self._ds

    def forward(self, x):
        return self.proj(self.backbone(x))

    def backbone_state_dict(self):
        """U-Net weights to carry into downstream finetuning (projection head dropped)."""
        return self.backbone.state_dict()


if __name__ == "__main__":
    net = CCLNet(in_chans=4, proj_dim=64, chans=16, num_pool_layers=4)
    x = torch.randn(2, 4, 240, 240)
    y = net(x)
    print("downsample_factor:", net.downsample_factor, "| out:", tuple(y.shape))
    y.sum().backward()
    print("backward OK")
