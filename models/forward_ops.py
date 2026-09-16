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
"""

import torch
import torch.nn as nn


KINDS = ("affine", "linear", "mlp", "conv")


class ForwardOp(nn.Module):
    def __init__(self, kind="conv", width=16, depth=3, kernel=3, act="silu", residual=True):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind, self.residual = kind, bool(residual)

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
                    # smooth on purpose: training THROUGH data_grad differentiates the VJP again,
                    # and ReLU's second derivative is zero almost everywhere
                    layers.append(nn.SiLU() if act == "silu" else nn.GELU())
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
