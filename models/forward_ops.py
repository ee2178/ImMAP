"""
Learned forward operators for bridge data consistency.

For the T1 -> CT1 bridge the unknown is x = CT1 and the measurement is y = T1, so the operator
maps CT1 -> T1 (it REMOVES enhancement) and the data term is

    f(x) = 1/2 || y - E(x) ||^2,        grad f(x) = -J_E(x)^T (y - E(x)).

J_E(x)^T r is one vector-Jacobian product -- a forward plus a backward through E -- so no
hand-written adjoint is needed; `data_grad` does it with autograd. For the linear kinds the VJP
is the transposed convolution, same cost as a forward.

Every kind is RESIDUAL with a zero-initialised last layer, so E starts as the identity (the
trivial baseline T1 ~ CT1) and training only ever has to learn the correction.

Kinds, lightest first (receptive field in brackets):
    affine   1x1 conv, 1 -> 1, no activation        [1]           per-pixel gain + offset
    linear   one k x k conv, 1 -> 1, no activation  [k]           linear, VJP = conv_transpose
    mlp      1x1 convs, depth layers of width        [1]           per-pixel nonlinear intensity map
    conv     k x k convs, depth layers of width      [depth*(k-1)+1]  local nonlinear map
    unet     small UNet: `levels` 2x downsamplings,   [grows ~2^levels]  multi-scale nonlinear map
             `convs` 3x3 convs per stage, channels width * min(2^level, max_mult)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


KINDS = ("affine", "linear", "mlp", "conv", "unet")


def _act(act):
    # smooth on purpose: training THROUGH data_grad differentiates the VJP again, and ReLU's
    # second derivative is zero almost everywhere
    return nn.SiLU() if act == "silu" else nn.GELU()


def _stage(c_in, c_out, convs, act):
    layers = []
    for i in range(convs):
        layers += [nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1,
                             padding_mode="replicate"), _act(act)]
    return nn.Sequential(*layers)


class SmallUNet(nn.Module):
    """1 -> 1 channel UNet: no normalisation (E must stay a fixed pointwise-in-batch map), strided
    2x2 convs down, transposed 2x2 convs up, concatenating skips, zero-initialised 1x1 readout.
    Any H, W: the input is replicate-padded to a multiple of 2^levels and the output cropped."""

    def __init__(self, width=16, levels=3, convs=2, max_mult=8, act="silu"):
        super().__init__()
        if levels < 1 or convs < 1:
            raise ValueError("unet needs levels >= 1 and convs >= 1")
        self.levels, self.convs = int(levels), int(convs)
        ch = [width * min(2 ** l, max_mult) for l in range(levels + 1)]
        self.inc = _stage(1, ch[0], convs, act)
        self.down = nn.ModuleList(nn.Conv2d(ch[l], ch[l + 1], 2, stride=2) for l in range(levels))
        self.enc = nn.ModuleList(_stage(ch[l + 1], ch[l + 1], convs, act) for l in range(levels))
        self.up = nn.ModuleList(nn.ConvTranspose2d(ch[l + 1], ch[l], 2, stride=2)
                                for l in range(levels))
        self.dec = nn.ModuleList(_stage(2 * ch[l], ch[l], convs, act) for l in range(levels))
        self.out = nn.Conv2d(ch[0], 1, 1)

    @property
    def receptive_field(self):
        # 3x3 convs add 2 * scale each, a 2x2 stride-2 down adds 1 * scale (its input scale);
        # transposed 2x2 stride-2 ups add nothing. Encoder and decoder both count.
        rf, n, L = 1, self.convs, self.levels
        rf += 2 * n                                         # inc, scale 1
        for l in range(L):
            rf += 2 ** l                                    # down from scale 2^l
            rf += 2 * n * 2 ** (l + 1)                      # enc stage at scale 2^(l+1)
            rf += 2 * n * 2 ** l                            # dec stage at scale 2^l
        return rf

    def forward(self, x):
        H, W = x.shape[-2:]
        m = 2 ** self.levels
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        h = self.inc(x)
        skips = [h]
        for down, enc in zip(self.down, self.enc):
            h = enc(down(h))
            skips.append(h)
        for l in reversed(range(self.levels)):
            h = self.dec[l](torch.cat([self.up[l](h), skips[l]], dim=1))
        return self.out(h)[..., :H, :W]


class ForwardOp(nn.Module):
    def __init__(self, kind="conv", width=16, depth=3, kernel=3, act="silu", residual=True,
                 levels=3, convs=2, max_mult=8):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.residual = kind, bool(residual)

        if kind == "unet":
            if not self.residual:
                raise ValueError("residual=False is only supported for affine/linear")
            self.net = SmallUNet(width=width, levels=levels, convs=convs, max_mult=max_mult, act=act)
            nn.init.zeros_(self.net.out.weight)
            nn.init.zeros_(self.net.out.bias)
            return

        if kind == "affine":
            layers = [nn.Conv2d(1, 1, 1)]
        elif kind == "linear":
            layers = [nn.Conv2d(1, 1, kernel, padding=kernel // 2, padding_mode="replicate")]
        else:
            if depth < 2:
                raise ValueError(f"{kind} needs depth >= 2 (depth 1 is 'affine'/'linear')")
            k = 1 if kind == "mlp" else kernel
            chans = [1] + [width] * (depth - 1) + [1]
            layers = []
            for i in range(depth):
                layers.append(nn.Conv2d(chans[i], chans[i + 1], k, padding=k // 2,
                                        padding_mode="replicate"))
                if i < depth - 1:
                    layers.append(_act(act))
        self.net = nn.Sequential(*layers)

        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if not self.residual:          # without the skip, start at identity through the weights
            with torch.no_grad():
                if kind in ("affine", "linear"):
                    last.weight[0, 0, last.kernel_size[0] // 2, last.kernel_size[1] // 2] = 1.0
                else:
                    raise ValueError("residual=False is only supported for affine/linear")

    @property
    def is_linear(self):
        return self.kind in ("affine", "linear")

    @property
    def receptive_field(self):
        if self.kind == "unet":
            return self.net.receptive_field
        return 1 + sum(m.kernel_size[0] - 1 for m in self.net if isinstance(m, nn.Conv2d))

    def forward(self, x):
        out = self.net(x)
        return x + out if self.residual else out

    def data_grad(self, x, y, create_graph=False):
        """J_E(x)^T (E(x) - y): the gradient of 1/2 ||y - E(x)||^2 w.r.t. x.

        create_graph=False  sampling-time correction; x is treated as a leaf.
        create_graph=True   inside an unrolled net that is being trained -- keeps x's graph so the
                            loss can differentiate through this step (second order in E).
        """
        with torch.enable_grad():
            xin = x if (create_graph and x.requires_grad) else x.detach().requires_grad_(True)
            r = self(xin) - y
            (g,) = torch.autograd.grad(r, xin, grad_outputs=r, create_graph=create_graph)
        return g


def load_forward_op(path, map_location="cpu", freeze=True):
    """Rebuild an E saved by scripts/fit_forward_ladder.py."""
    ckpt = torch.load(path, map_location=map_location)
    spec = {k: v for k, v in ckpt["spec"].items() if k != "name"}
    E = ForwardOp(**spec)
    E.load_state_dict(ckpt["state_dict"])
    if freeze:
        E.requires_grad_(False).eval()
    return E
