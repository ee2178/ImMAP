"""
ContrastTransfer -- a deliberately TINY net that maps the guide contrast onto the target's, so a
same-contrast guided network (LGGS) can use a different-contrast guide.

LGGS pools pixels by their similarity in the GUIDE (models/guided_prox.py): it assumes the guide
looks like the target. A PD guide for a PDFS target breaks that where the contrasts disagree --
fat is bright in PD and suppressed in PDFS -- so pixels that are alike in PD get pooled although
they differ in PDFS. This net is pretrained on its own (training/contrast_transfer.py: guide in,
fully-sampled target out) and then frozen inside LGGS (LGGSNet(guide_transfer=...)), which sees
T(guide) instead of the guide.

Input: the guide exactly as a guided recon net gets it -- the side scan coil-combined under the
operator's maps (training/recon.py::_guide), complex (B, 1, H, W) or (B, G, 1, H, W).

    arch "pixel"   per-pixel MLP (1x1 convs): a pure intensity map, no spatial context
         "patch"   one kernel x kernel conv, then 1x1 layers: an MLP on a small neighbourhood
         "unet"    a very small UNet (`levels` resolutions, `width` channels at the top)
    mode "mag"     magnitude in, magnitude out, the GUIDE's phase kept:
                       T(g) = relu(|g| + f(|g|)) * g / |g|
                   LGGS's similarity is phase-invariant (pidistance), so the magnitude is what
                   matters; the residual form starts the net at the identity
         "complex" [Re, Im] in and out, T(g) = g + f(g): also learns a phase map

Every arch is residual around the identity (f's last layer starts at zero), so an untrained
transfer IS the plain guide.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCHS = ("pixel", "patch", "unet")
MODES = ("mag", "complex")


def _mlp(c_in, c_out, width, depth, kernel):
    """`depth` conv layers; the first `kernel` x `kernel`, the rest 1x1, ReLU between."""
    layers, c = [], c_in
    for i in range(depth - 1):
        k = kernel if i == 0 else 1
        layers += [nn.Conv2d(c, width, k, padding=k // 2, padding_mode="reflect"), nn.ReLU()]
        c = width
    k = kernel if depth == 1 else 1
    layers.append(nn.Conv2d(c, c_out, k, padding=k // 2, padding_mode="reflect"))
    return nn.Sequential(*layers)


class _TinyUNet(nn.Module):
    def __init__(self, c_in, c_out, width=8, levels=2):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.levels = int(levels)

        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, padding_mode="reflect"),
                                 nn.ReLU(),
                                 nn.Conv2d(b, b, 3, padding=1, padding_mode="reflect"),
                                 nn.ReLU())
        ws = [int(width) * 2 ** i for i in range(self.levels)]
        self.down = nn.ModuleList([block(c_in if i == 0 else ws[i - 1], ws[i])
                                   for i in range(self.levels)])
        self.up = nn.ModuleList([block(ws[i] + ws[i + 1], ws[i])
                                 for i in range(self.levels - 1)])
        self.head = nn.Conv2d(ws[0], c_out, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** (self.levels - 1)
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for i in reversed(range(self.levels - 1)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.up[i](torch.cat([skips[i], x], dim=1))
        return self.head(x)[..., :H, :W]


class ContrastTransfer(nn.Module):
    # what training/i2sb_recon.py::_measure / training/recon.py::_guide read off a net: it takes
    # the image-domain guide, on the measured grid, combined with the operator's maps
    guided = True
    guide_domain = "image"
    pad_stride = 1
    uses_operator_smaps = True

    def __init__(self, arch="unet", mode="mag", width=8, depth=3, kernel=3, levels=2):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"arch {arch!r} not in {ARCHS}")
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        self.arch, self.mode = arch, mode
        c = 1 if mode == "mag" else 2
        if arch == "pixel":
            self.f = _mlp(c, c, int(width), int(depth), 1)
        elif arch == "patch":
            self.f = _mlp(c, c, int(width), int(depth), int(kernel))
        else:
            self.f = _TinyUNet(c, c, int(width), int(levels))
        last = [m for m in self.f.modules() if isinstance(m, nn.Conv2d)][-1]
        nn.init.zeros_(last.weight)                 # starts at the identity
        nn.init.zeros_(last.bias)

    def forward(self, guide):
        if not torch.is_complex(guide):
            raise TypeError(f"ContrastTransfer takes the complex guide image, got {guide.dtype}")
        shape = guide.shape
        g = guide.reshape(-1, 1, *shape[-2:])        # (B[, G], 1, H, W) -> (N, 1, H, W)
        if self.mode == "mag":
            mag = g.abs()
            m = F.relu(mag + self.f(mag))
            phase = torch.where(mag > 0, g / mag.clamp_min(1e-12), torch.ones_like(g))
            out = m * phase
        else:
            p = torch.cat([g.real, g.imag], dim=1)
            d = self.f(p)
            out = g + torch.complex(d[:, :1], d[:, 1:2])
        return out.reshape(shape)

    def extra_repr(self):
        n = sum(p.numel() for p in self.parameters())
        return f"arch={self.arch}, mode={self.mode}, params={n}"


def load_frozen_transfer(spec):
    """{"params": {...ContrastTransfer kwargs}, "ckpt": path} -> the pretrained net, frozen, in
    eval mode. The checkpoint's model_state_dict (net_bestval.ckpt of a contrast_transfer run)."""
    if not isinstance(spec, dict) or "ckpt" not in spec:
        raise ValueError("guide_transfer must be {'params': {...}, 'ckpt': <path>}, got "
                         f"{spec!r}")
    net = ContrastTransfer(**(spec.get("params") or {}))
    sd = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model_state_dict"])
    for p in net.parameters():
        p.requires_grad_(False)
    return net.eval()
