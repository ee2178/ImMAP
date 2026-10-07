# -*- coding: utf-8 -*-
"""
Networks for the Unpaired Neural Schrodinger Bridge (sb/unsb.py): a time-conditioned PatchGAN, the
tap that reads G's encoder features, and the PatchNCE loss on them. Ported from the UNSB / CUT code
(NLayerDiscriminator_ncsn, PatchSampleF, PatchNCELoss).

    PatchDiscriminator   the piece that matters most. As D it scores an image (+ optional
                         conditioning) as real/fake per patch; as the entropy critic E it takes TWO
                         (x_t, x_hat) pairs stacked on channels (`x2`). `build_critics` sizes both
                         from ONE conditioning toggle.
    PatchEncoder         a small own-weights feature extractor for PatchNCE: needs nothing from G,
                         so it works with ANY regressor (UNet, SBCDLNet, SBGuidedGroupCDL, ...).
    FeatureTap           the alternative extractor: reads named modules of G with forward hooks.
                         Only for nets that HAVE encoder-like modules (SBUnet's input_blocks); the
                         unrolled CDL nets do not.
    PatchNCE             MLP heads + the contrastive loss; takes either extractor.

Both extractors share one protocol -- `ext(x, cond, sigma) -> [feature maps]` and
`ext.channels(x, cond, sigma) -> [C per map]` -- which is all sb.unsb.make_nce_fn needs.

TIME. D and E are conditioned on the bridge STEP INDEX (the same quantity SBUnet embeds), through
a sinusoidal embedding added as a per-channel bias in every block. "cond" in the original name
`basic_cond` means time, NOT image conditioning; conditioning on T2/FLAIR/a prior study is done
by widening `in_channels` and concatenating them in sb.unsb.d_loss.

RECEPTIVE FIELD. Three antialiased stride-2 stages and 4x4 convs: ~80 px per output logit by a
hand count, at 1/8 resolution. D therefore judges local appearance, not whole-lesion placement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sb_unet import timestep_embedding
from operators import Identity


def _init_normal(m, std=0.02):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class BlurPool(nn.Module):
    """Antialiased stride-2 downsampling: a fixed [1 2 1] x [1 2 1] / 16 blur, per channel."""

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        k = k[:, None] * k[None, :]
        self.channels = int(channels)
        self.register_buffer("kernel", (k / k.sum())[None, None].repeat(self.channels, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernel,
                        stride=2, groups=self.channels)


class _Block(nn.Module):
    """conv4 -> + time bias -> [instance norm] -> LeakyReLU -> [blur-downsample]."""

    def __init__(self, cin, cout, emb_dim, norm, down, spectral_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=4, stride=1, padding=1)
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        self.dense = nn.Linear(emb_dim, cout)
        self.norm = nn.InstanceNorm2d(cout, affine=False) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, True)
        self.down = BlurPool(cout) if down else nn.Identity()

    def forward(self, x, emb):
        h = self.conv(x) + self.dense(emb)[..., None, None]
        return self.down(self.act(self.norm(h)))


class PatchDiscriminator(nn.Module):
    """Time-conditioned PatchGAN. Output: a (B, 1, h, w) map of patch logits.

    in_channels     channels of ONE input. As D: target_channels (+ n_cond if conditional). As E:
                    2 * target_channels per pair, and E is called with x2 so the net sees TWICE this.
    spectral_norm   Lipschitz-bound every conv (a standard GAN stabiliser; off by default, as in CUT).
    The default instance norm and the 0.02-normal init follow CUT. Build through `build_critics`
    so D's width and the conditioning toggle cannot disagree."""

    def __init__(self, in_channels, ndf=64, n_layers=3, pair=False, spectral_norm=False):
        super().__init__()
        self.ndf = int(ndf)
        self.pair = bool(pair)
        self.in_channels = int(in_channels)
        emb = 4 * self.ndf
        self.t_embed = nn.Sequential(
            nn.Linear(self.ndf, emb), nn.LeakyReLU(0.2, True), nn.Linear(emb, emb))
        cin = int(in_channels) * (2 if pair else 1)
        chans = [self.ndf * min(2 ** i, 8) for i in range(n_layers + 1)]    # ndf, 2, 4, 8 ndf
        sn = bool(spectral_norm)
        blocks = [_Block(cin, chans[0], emb, norm=False, down=True, spectral_norm=sn)]
        blocks += [_Block(chans[i - 1], chans[i], emb, norm=True, down=True, spectral_norm=sn)
                   for i in range(1, n_layers)]
        blocks += [_Block(chans[n_layers - 1], chans[n_layers], emb, norm=True, down=False,
                          spectral_norm=sn)]
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(chans[n_layers], 1, kernel_size=4, stride=1, padding=1)
        if sn:
            self.final = nn.utils.spectral_norm(self.final)
        self.apply(_init_normal)             # (a no-op on spectral-norm convs' normalised weight)

    def forward(self, x, step, x2=None):
        if (x2 is not None) != self.pair:
            raise ValueError("a pair critic needs x2; a plain discriminator must not get one")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"this net was built for {self.in_channels} input channel(s) per image, got "
                f"{x.shape[1]}. Size D from the conditioning toggle with build_critics("
                f"target_channels, n_cond, cond_d=...) and pass the SAME cond_d to d_loss/g_loss.")
        step = torch.as_tensor(step, device=x.device).reshape(-1).long()
        if step.numel() == 1:
            step = step.expand(x.shape[0])
        emb = self.t_embed(timestep_embedding(step, self.ndf))
        h = x if x2 is None else torch.cat([x, x2], dim=1)
        for block in self.blocks:
            h = block(h, emb)
        return self.final(h)


def build_critics(target_channels=1, n_cond=0, cond_d=True, entropy=True, ndf=64, n_layers=3,
                  spectral_norm=False):
    """(D, E) sized from ONE conditioning toggle.

    cond_d=True   D scores (image, cond) pairs: input width target_channels + n_cond. This is the
                  only way the discriminator can enforce that the output is consistent with
                  T2 / FLAIR / the prior study. Pass the same `cond_d` to d_loss and g_loss.
    cond_d=False  D scores the image alone: it learns only what a target-contrast image looks
                  like, so G is free to ignore its conditioning channels.
    entropy=False returns E = None (UNSB without the entropy term; also lifts the batch >= 2 need).
    """
    D = PatchDiscriminator(target_channels + (n_cond if cond_d else 0), ndf=ndf,
                           n_layers=n_layers, spectral_norm=spectral_norm)
    E = PatchDiscriminator(2 * target_channels, ndf=ndf, n_layers=n_layers,
                           pair=True) if entropy else None
    return D, E


class PatchEncoder(nn.Module):
    """A small conv pyramid with its OWN weights, used as the PatchNCE feature extractor when G has
    no encoder to tap (the unrolled SBCDLNet / SBGuidedGroupCDL) or when you want PatchNCE to be
    independent of G's architecture. Takes an image, ignores `cond` and `sigma`.

    It is trained by the NCE loss alone, so put its parameters (and PatchNCE's) in G's optimizer.
    Reflect padding, so the zero-padding border cannot leak absolute position -- a learned encoder
    would otherwise match query to key by WHERE a patch is, not WHAT it contains.

    Level l has ch * 2**l channels at 1/2**l resolution (level 0 is full resolution)."""

    def __init__(self, in_channels=1, ch=32, n_levels=3):
        super().__init__()
        self.widths = [int(ch) * 2 ** l for l in range(int(n_levels))]
        blocks, c = [], int(in_channels)
        for l, w in enumerate(self.widths):
            blocks.append(nn.Sequential(
                nn.Conv2d(c, w, 3, stride=1 if l == 0 else 2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(w, w, 3, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2, True)))
            c = w
        self.blocks = nn.ModuleList(blocks)
        self.apply(_init_normal)

    def forward(self, x, cond=None, sigma=None):
        feats, h = [], x
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return feats

    def channels(self, x=None, cond=None, sigma=None):
        return list(self.widths)


class _Stop(Exception):
    """Raised by FeatureTap once every requested layer has fired: skips the rest of the net."""


class FeatureTap:
    """Read the outputs of named submodules of `net` during ONE forward call.

    Hooks are registered once but only record while a tap call is running, so G's ordinary
    training and sampling passes are untouched (no retained activations, no early exit). With
    `stop_early` the forward is abandoned as soon as the last requested layer has fired, which
    skips the decoder of a UNet entirely. A module called several times keeps its LAST output.

    `layers` are names from `net.named_modules()`, e.g. SBUnet: "input_blocks.3", "input_blocks.6".
    The tap calls the net as predict_x0 does -- net(cat[x, cond], E=Identity(), sigma=sigma) -- so
    it supports plain regressors, not guided or learned-DC ones."""

    def __init__(self, net, layers, stop_early=True):
        mods = dict(net.named_modules())
        missing = [name for name in layers if name not in mods]
        if missing:
            raise ValueError(f"no such module(s) {missing}; try list(net.named_modules())")
        self.net, self.layers, self.stop_early = net, list(layers), bool(stop_early)
        self._on, self._buf = False, {}
        self._handles = [mods[name].register_forward_hook(self._hook(name)) for name in layers]

    def _hook(self, name):
        def hook(module, args, out):
            if not self._on:
                return
            self._buf[name] = out[0] if isinstance(out, (tuple, list)) else out
            if self.stop_early and len(self._buf) == len(self.layers):
                raise _Stop
        return hook

    def __call__(self, x, cond, sigma):
        net_in = x if (cond is None or cond.shape[1] == 0) else torch.cat([x, cond], dim=1)
        self._buf, self._on = {}, True
        try:
            self.net(net_in, E=Identity(), sigma=sigma)
        except _Stop:
            pass
        finally:
            self._on = False
        return [self._buf[name] for name in self.layers]

    @torch.no_grad()
    def channels(self, x, cond, sigma):
        """Channel count of each tapped layer, from one probe call -- what PatchNCE is built with
        (its MLPs are created up front, not lazily, so they exist when the optimizer is built)."""
        return [f.shape[1] for f in self(x, cond, sigma)]

    def remove(self):
        for h in self._handles:
            h.remove()


class PatchNCE(nn.Module):
    """CUT's PatchNCE. At P sampled locations per layer, the OUTPUT's feature (query) must match
    the SOURCE's feature at the same location (positive) better than the source's features at the
    other P-1 sampled locations of the same image (negatives): an InfoNCE over locations.

    Features go through a per-layer MLP (Linear-ReLU-Linear) and are L2-normalised. Keys are
    detached, so only the output side is pulled. The same locations are used for query and key.

    Compares LEARNED features, not intensities: contrast may change freely, anatomy may not move.
    `mask` (B, 1, H, W), if given, biases the sampled locations into the brain -- uniform sampling
    over skull-stripped data draws many near-identical air patches, which are false negatives."""

    def __init__(self, channels, nc=256, num_patches=256, temperature=0.07):
        super().__init__()
        self.num_patches, self.temperature = int(num_patches), float(temperature)
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(c, nc), nn.ReLU(), nn.Linear(nc, nc)) for c in channels])
        self.apply(_init_normal)

    @staticmethod
    def _ids(B, H, W, P, mask, device):
        score = torch.rand(B, H * W, device=device)
        if mask is not None:                              # in-mask locations always outrank the rest
            score = score + F.interpolate(mask.float(), size=(H, W), mode="nearest").flatten(1)
        return score.topk(P, dim=1).indices               # (B, P)

    @staticmethod
    def _gather(feat, ids):
        flat = feat.flatten(2).transpose(1, 2)            # (B, HW, C)
        return flat.gather(1, ids[..., None].expand(-1, -1, flat.shape[-1]))

    def _info_nce(self, q, k):
        B, P, _ = q.shape
        l_pos = (q * k).sum(-1, keepdim=True)             # (B, P, 1)
        l_neg = torch.bmm(q, k.transpose(1, 2))           # (B, P, P)
        l_neg = l_neg.masked_fill(torch.eye(P, dtype=torch.bool, device=q.device)[None], -10.0)
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature
        target = torch.zeros(B * P, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits.reshape(B * P, -1), target)

    def forward(self, feats_q, feats_k, mask=None):
        total = 0.0
        for mlp, fq, fk in zip(self.mlps, feats_q, feats_k):
            B, _, H, W = fk.shape
            ids = self._ids(B, H, W, min(self.num_patches, H * W), mask, fk.device)
            k = F.normalize(mlp(self._gather(fk, ids)), dim=-1).detach()
            q = F.normalize(mlp(self._gather(fq, ids)), dim=-1)
            total = total + self._info_nce(q, k)
        return total / len(self.mlps)
