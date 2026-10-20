"""
RAM (Reconstruct Anything Model) as an ImMAP model.

The network is `_upstream/ram.py`, vendored from github.com/matthieutrs/ram
with one import moved (see that file's header). Nothing here changes the
architecture; this is the calling-convention adapter, and every difference
between the two codebases is handled in one place so the baseline stays honest.

WHAT HAS TO BE TRANSLATED
-------------------------
                    ImMAP                          RAM / deepinv
  measurement       y (B, C, H, W) complex         real tensor; here (B, 2, C, H, W)
  operator          E = Mask @ FFT2D @ Sense       a deepinv LinearPhysics: A, A_adjoint
  image             (B, 1, H, W) complex           (B, 2, H, W) real, [re; im] -- the
                                                   2-channel head RAM uses for complex
  noise level       sigma (B, 1, 1, 1): std of     physics.noise_model.sigma: std of
                    the COMPLEX noise, E|z|^2 = 1  EACH REAL COMPONENT

`_ImMAPPhysics` exposes `E` to RAM as a deepinv `LinearPhysics` and nothing
more: RAM only ever calls `A`, `A_adjoint` and (through its own multiscale /
padding wrappers) deepinv's generic `LinearPhysics.prox_l2`, the conjugate-
gradient solve it uses to realign its input. Those run unmodified from the
installed deepinv, so the data-consistency path is deepinv's code, not a
re-implementation of it.

THE NOISE LEVEL. ImMAP's noise is complex with E|z|^2 = 1 (operators/noise.py),
so each real component has std sigma / sqrt(2). deepinv's GaussianNoise draws
each real entry with std sigma, so the level RAM is conditioned on -- and the
one its realignment SNR divides by -- is sigma / sqrt(2). That is a fixed
factor, which a model trained from scratch would absorb; it is converted anyway
so the conditioning means what it meant in the paper, and so pretrained weights
see the levels they were trained on.

Noise is still added the ImMAP way (coil-image domain, before the mask), so
unsampled k-space bins are exactly zero rather than pure noise. RAM never sees
how y was generated, only A and sigma, so nothing in the network depends on it.

THE ONE REPRODUCED METHOD. `RAM.forward` is re-stated in `_ram_forward` below
because it turns sigma into a one-element tensor with Python's `max`, which
works only for a scalar; ImMAP passes one level per batch element. The change is
`max(sigma, 1e-3)` -> `sigma.clamp_min(1e-3)` over a (B,) tensor. With B = 1
(every generated config) the two are identical -- tests/test_ram.py checks the
outputs against upstream's `forward` bit for bit.

GRID SIZE. `pad_stride = 1`: RAM pads its own input to a multiple of 8 (its
`Pad` physics, top-left), so no image-domain embedding is needed here and
`training/common.py::embed_for_net` leaves the operator alone -- as for VarNet.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from models.ram._upstream.ram import RAM as _UpstreamRAM, Pad
except ModuleNotFoundError as e:                       # pragma: no cover
    if e.name != "deepinv":
        raise
    raise ModuleNotFoundError(
        "the RAM baseline needs deepinv (tested with 0.4.2). Install it on a "
        "LOGIN node -- see the setup note at the top of torch/exp13_ram.sbatch, "
        "which checks it will not replace your CUDA torch.", name="deepinv") from e


def _physics_base():
    # deepinv is imported lazily so `import models` keeps working in an env
    # without it; only building a RAM needs it.
    from deepinv.physics import GaussianNoise, LinearPhysics
    return GaussianNoise, LinearPhysics


def _make_physics(E, sigma):
    """ImMAP's `E` as a deepinv LinearPhysics in deepinv's real layout.

    `sigma` is (B,), the per-REAL-component std -- see the module docstring.
    """
    GaussianNoise, LinearPhysics = _physics_base()

    class _PinnedGaussianNoise(GaussianNoise):
        """A GaussianNoise that no parent `.to()` can move.

        deepinv 0.4's `LinearPhysics.__init__` ends in `self.to(device)` with
        `device` defaulting to "cpu", and that recurses into the noise model.
        RAM's `Pad` and `MultiScalePhysics` (upstream ram.py) wrap this physics
        WITHOUT a device and share its noise model, so on a GPU they moved sigma
        to the CPU mid-forward and `realign_input` failed mixing cuda and cpu
        (seen 2026-10-20, RAM_knee_R12). The noise model is built per forward,
        on sigma's device, and has no reason to move -- so it doesn't. Upstream
        was written against deepinv 0.3, where the physics did not move itself.
        """

        def _apply(self, fn, recurse=True):
            return self

    class _ImMAPPhysics(LinearPhysics):
        def __init__(self, E, sigma):
            # device= so this wrapper's own `self.to(device)` is a no-op rather
            # than a move to deepinv's default "cpu".
            super().__init__(noise_model=_PinnedGaussianNoise(sigma=sigma),
                             device=sigma.device)
            self.E = E

        def A(self, x, **kwargs):
            # (B, 2, H, W) real -> E -> (B, C, H, W) complex -> (B, 2, C, H, W)
            y = self.E.forward(torch.complex(x[:, 0:1], x[:, 1:2]))
            return torch.stack((y.real, y.imag), dim=1)

        def A_adjoint(self, y, **kwargs):
            # (B, 2, C, H, W) -> complex (B, C, H, W) -> E^H -> (B, 2, H, W)
            x = self.E.adjoint(torch.complex(y[:, 0], y[:, 1]))
            return torch.cat((x.real, x.imag), dim=1)

    return _ImMAPPhysics(E, sigma)


def _ram_forward(net, y, physics, sigma):
    """`RAM.forward` (upstream ram.py, lines 294-318), per-sample sigma.

    Upstream, with the physics' noise model a GaussianNoise and no gain:

        sigma = physics.noise_model.sigma
        sigma = torch.tensor([max(sigma, 1e-3)], ...)
        gamma = 5e-3                      # noise model has no `gain`
        gamma = torch.tensor([max(gamma, 1e-3)], ...)

    Here sigma is a (B,) tensor, so `max` becomes `clamp_min`; gamma is the
    same constant, broadcast to (B,). Everything else is line for line,
    including the adjoint computed twice (once only to read off the padding).
    """
    x_temp = physics.A_adjoint(y)
    pad = (-x_temp.size(-2) % 8, -x_temp.size(-1) % 8)
    physics = Pad(physics, pad)

    x_in = physics.A_adjoint(y)

    sigma = sigma.to(x_in.device).clamp_min(1e-3)
    gamma = torch.full_like(sigma, max(5e-3, 1e-3))

    out = net.forward_unet(x_in, sigma=sigma, gamma=gamma, physics=physics, y=y)

    out = physics.remove_pad(out)

    return out


# ---------------------------------------------------------------------------
# DRUNet initialisation -- how the paper starts training ("We use the
# pretrained weights of the DRUNet denoiser to initialize our model", Terris et
# al. 2025, Sec. 4). RAM is DRUNet's U-Net with a measurement-conditioning
# branch added to every residual block, so the TRUNK maps one to one:
#
#   RAM                                  DRUNet (deepinv)
#   m_down{k}.enc[i].conv1 / conv2       m_down{k}[i].res[0] / res[2]
#   pool{k}                              m_down{k}[nb]          (strided conv)
#   m_body.enc[i].conv1 / conv2          m_body[i].res[0] / res[2]
#   up{k}                                m_up{k}[0]             (transposed conv)
#   m_up{k}.enc[i].conv1 / conv2         m_up{k}[1 + i].res[0] / res[2]
#
# 62 convolutions, ~97% of RAM's parameters. NOT copied, because DRUNet has no
# counterpart: the complex (2-channel) head and tail -- DRUNet's are 1- or
# 3-channel, and the paper does not say how its 2-channel head was started --
# and every MeasCondBlock / gain (RAM's additions). Those keep upstream's own
# initialisation. The paper does not publish its init code; this is the
# mapping the shared module structure fixes.
# ---------------------------------------------------------------------------
DRUNET_INITS = {"drunet_gray": 1, "drunet_color": 3}


def drunet_trunk_pairs(ram, drunet, nb=4):
    """[(ram_conv, drunet_conv)] for every trunk convolution, in order."""
    pairs = []
    for k in (1, 2, 3):
        down, up = getattr(drunet, f"m_down{k}"), getattr(drunet, f"m_up{k}")
        r_down, r_up = getattr(ram, f"m_down{k}"), getattr(ram, f"m_up{k}")
        for i in range(nb):
            pairs += [(r_down.enc[i].conv1, down[i].res[0]),
                      (r_down.enc[i].conv2, down[i].res[2])]
        pairs.append((getattr(ram, f"pool{k}"), down[nb]))
        pairs.append((getattr(ram, f"up{k}"), up[0]))
        for i in range(nb):
            pairs += [(r_up.enc[i].conv1, up[1 + i].res[0]),
                      (r_up.enc[i].conv2, up[1 + i].res[2])]
    for i in range(nb):
        pairs += [(ram.m_body.enc[i].conv1, drunet.m_body[i].res[0]),
                  (ram.m_body.enc[i].conv2, drunet.m_body[i].res[2])]
    return pairs


@torch.no_grad()
def init_trunk_from_drunet(ram, drunet, nb=4):
    """Copy DRUNet's trunk into an upstream RAM; returns the number of tensors copied.

    Strict both ways: every pair must match in shape, and every DRUNet
    parameter outside its head and tail must be used exactly once -- so a
    deepinv release that reshapes DRUNet fails here, at construction, rather
    than leaving part of the trunk silently random.
    """
    used, n = set(), 0
    for dst, src in drunet_trunk_pairs(ram, drunet, nb):
        for name, p in src.named_parameters(recurse=False):
            q = getattr(dst, name)
            if q is None or q.shape != p.shape:
                raise ValueError(f"DRUNet -> RAM shape mismatch at {type(dst).__name__}."
                                 f"{name}: {None if q is None else tuple(q.shape)} "
                                 f"vs {tuple(p.shape)}")
            q.copy_(p)
            used.add(id(p))
            n += 1
        if dst.bias is not None and src.bias is None:
            dst.bias.zero_()
    left = [name for name, p in drunet.named_parameters()
            if id(p) not in used and not name.startswith(("m_head", "m_tail"))]
    if left:
        raise ValueError(f"DRUNet parameters with no place in RAM's trunk: {left[:6]}"
                         f"{' ...' if len(left) > 6 else ''}")
    return n


def load_drunet(kind, weights=None):
    """deepinv's pretrained DRUNet: `kind` is "drunet_gray" or "drunet_color".

    `weights`, if given, is a path to that DRUNet's state dict; otherwise
    deepinv's own download, which reads ~/.cache/torch/hub/checkpoints first --
    so fetch it once on a LOGIN node (compute nodes have no route out):
        python -c "from deepinv.models import DRUNet; DRUNet(1, 1, pretrained='download')"
    (3, 3 for drunet_color).
    """
    if kind not in DRUNET_INITS:
        raise ValueError(f"init_from={kind!r}; known: {sorted(DRUNET_INITS)}")
    from deepinv.models import DRUNet
    c = DRUNET_INITS[kind]
    return DRUNet(in_channels=c, out_channels=c, pretrained=weights or "download")


class RAM(nn.Module):
    """RAM behind ImMAP's `(y, E, sigma) -> (x_hat, extra)`.

    Parameters
    ----------
    in_channels : list of int
        RAM's input heads. `[2]` (the default) builds only the complex head,
        which is all MRI uses -- the right choice from scratch, since unused
        heads would sit in the optimizer receiving no gradient and inflate the
        parameter count. Upstream's default `[1, 2, 3]` is required to load
        the published weights.
    pretrained : False, True or str
        False: random init (from scratch). True: upstream's own download from
        Hugging Face (`mterris/ram`) -- a login node only, compute nodes have
        no route out. A path: load that state dict (the same file, fetched
        beforehand), e.g. `ram.pth.tar`.
    init_from : None, "drunet_gray" or "drunet_color"
        Start the TRUNK from deepinv's pretrained DRUNet, as the paper does
        (see `init_trunk_from_drunet`); everything else keeps upstream's init.
        Training is otherwise from scratch. Exclusive with `pretrained`.
    init_weights : str, optional
        Path to that DRUNet's state dict, instead of deepinv's cached download.

    The DRUNet load happens on EVERY construction, a resumed or evaluated run
    included (its checkpoint then overwrites the trunk) -- so the cached
    weights must exist wherever a `ramdrunet` run is built.
    """

    # Read by training/common.py::embed_for_net; 1 = no embedding (RAM pads).
    pad_stride = 1
    # A complex image, the same quantity the unrolled nets produce.
    returns_magnitude = False
    # RAM uses the operator's coil maps through A / A_adjoint.
    uses_operator_smaps = True

    def __init__(self, in_channels=(2,), pretrained=False, init_from=None,
                 init_weights=None):
        super().__init__()
        if init_from is not None and pretrained:
            raise ValueError("init_from (DRUNet trunk) and pretrained (the published "
                             "RAM) are alternatives; give one.")
        in_channels = [int(c) for c in in_channels]
        if 2 not in in_channels:
            raise ValueError(
                f"in_channels={in_channels} has no 2-channel head, which is the "
                f"one RAM uses for complex images; MRI cannot run without it.")
        if pretrained and in_channels != [1, 2, 3]:
            raise ValueError(
                "the published weights carry heads for 1, 2 and 3 channels; "
                "load them with in_channels=[1, 2, 3].")
        self.in_channels = in_channels
        self.net = _UpstreamRAM(in_channels=in_channels,
                                pretrained=(pretrained is True))
        if isinstance(pretrained, str):
            state = torch.load(pretrained, map_location="cpu")
            self.net.load_state_dict(state.get("model_state_dict", state)
                                     if isinstance(state, dict) else state)
        self.pretrained = pretrained
        self.init_from = init_from
        if init_from is not None:
            n = init_trunk_from_drunet(self.net, load_drunet(init_from, init_weights))
            print(f"[RAM] trunk initialised from {init_from}: {n} tensors")

    def forward(self, y, E=None, sigma=None, **_):
        if E is None:
            raise ValueError("RAM needs the encoding operator E: it reconstructs "
                             "through A and A^H, not from an adjoint image.")
        if sigma is None:
            raise ValueError("RAM is noise-conditioned; pass the noise level sigma.")
        if not torch.is_complex(y):
            raise ValueError(f"expected complex k-space, got {y.dtype}.")

        B = y.shape[0]
        # ImMAP's complex std -> deepinv's per-real-component std.
        s = torch.as_tensor(sigma, device=y.device, dtype=torch.float32)
        s = s.reshape(-1).expand(B) if s.numel() == 1 else s.reshape(B)
        s = s / math.sqrt(2.0)

        physics = _make_physics(E, s)
        y_r = torch.stack((y.real, y.imag), dim=1).contiguous()   # (B, 2, C, H, W)
        out = _ram_forward(self.net, y_r, physics, s)
        return torch.complex(out[:, 0:1], out[:, 1:2]), {}

    def project(self):
        """No constraint set. Present so the training loop's hasattr holds."""
        return self

    def extra_repr(self):
        return (f"in_channels={self.in_channels}, pretrained={self.pretrained!r}, "
                f"init_from={self.init_from!r}")
