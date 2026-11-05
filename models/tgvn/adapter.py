"""
TGVN (trust-guided variational network) as an ImMAP model, and its I2SB bridge regressor.

The networks are `_upstream.py`, vendored unmodified apart from its imports. This file is the
calling-convention adapter, the same job models/e2evarnet/adapter.py does for E2E-VarNet, plus
two optional extensions that leave the upstream modules untouched:

                    ImMAP                              TGVN upstream
  measurement       y (B, C, H, W) complex             (B, C, H, W, 2) real-view
  operator          E = Mask @ FFT2D @ Sense           mask alone, broadcastable bool
  side information  guide = SIDE k-space (B, C, H, W)  second_kspace (B, C, H, W, 2)
  coil maps         ignored, unless use_smaps          estimated by its own SensitivityModel
  noise level       sigma, used only if cond_sigma     not an input
  output            (B, 1, H, W): |x|, or x if complex complex_abs of the iterate

`guide_domain = "kspace"`: the recon loops hand this model the side k-space itself (zero-filled
when the guide is accelerated), not a side image -- TGVN reduces it with the MAIN maps inside
every cascade (`TGVN_1S`), which is part of the published method.

`guided=False` builds the paper's matched no-side-information baseline: upstream `VarNetImage`,
an image-domain E2E-VarNet with the same NormUnet and SensitivityModel.

EXTENSIONS (all off by default, so the default is upstream TGVN):

  cond_sigma   CONDITIONING PLANES. Upstream TGVN takes no noise level. With cond_sigma=True
               each cascade's regularizer Phi is a CondNormUnet: the same NormUnet with constant
               input planes appended after its normalization. A noise-level plane is divided by
               the image scale the NormUnet whitens by, sqrt(tr Cov), so the U-Net sees sigma in
               its own normalized units. H (the side-to-main map) is not conditioned.
  use_smaps    KNOWN MAPS: the cascades run on the operator's coil maps instead of the
               SensitivityModel's (deleted, so it adds no idle parameters). Needed for a complex
               output that shares a phase reference with anything outside the net.
  output       "magnitude" (upstream) or "complex" (requires use_smaps).

BridgeTGVN is the I2SB regressor: known maps, complex output, the cascades started at the bridge
state x_t instead of E^H y, and Phi conditioned on the bridge position (and on sigma_n).

Upstream defaults (`scripts/main_fastmri.py`) are num_cascades=12, chans=18; the paper's knee runs
use 14 cascades and 21 channels.
"""

import torch
import torch.nn as nn

from models.sb_schedule import BridgeScheduleMixin
from models.tgvn._upstream import (NormUnet, TGVN_1S, VarNetImage, complex_abs,
                                   sens_reduce)
from operators.accessors import get_mask, get_sensitivity_map


class CondNormUnet(NormUnet):
    """Upstream NormUnet with `n_cond` constant input planes, set per forward via `.cond`.

    `.cond` is (B, n_cond). Channels listed in `noise_channels` are noise stds (complex, total)
    and are divided by sqrt(tr Cov) of the input -- the scale the Cholesky whitening removes --
    before they become planes; the rest (e.g. a bridge position in [-1, 1]) go in as given."""

    def __init__(self, chans, num_pools, n_cond, noise_channels=()):
        super().__init__(chans, num_pools, in_chans=2 + int(n_cond), out_chans=2)
        self.n_cond = int(n_cond)
        self.noise_channels = tuple(int(i) for i in noise_channels)
        self.cond = None

    def forward(self, x):
        if self.cond is None or self.cond.shape[-1] != self.n_cond:
            raise RuntimeError(f"CondNormUnet needs .cond of shape (B, {self.n_cond}) set "
                               f"before each forward")
        x = self.complex_to_chan_dim(x)
        x, mean, L = self.norm(x)
        c = self.cond.to(x.dtype)
        if self.noise_channels:
            # Built OUT OF PLACE: scale depends on the input (through L), so autograd saves the
            # numerator of each division; writing the result back into `c` would overwrite it
            # (the "modified by an inplace operation" failure, seen 2026-11-05 on the cluster).
            scale = L.pow(2).sum(dim=(1, 2)).sqrt().clamp_min(1e-12)     # sqrt(tr(L L^T))
            c = torch.stack([c[:, i] / scale if i in self.noise_channels else c[:, i]
                             for i in range(self.n_cond)], dim=1)
        x, pad_sizes = self.pad(x)
        planes = c[:, :, None, None].expand(-1, -1, *x.shape[-2:])
        x = self.unet(torch.cat([x, planes], dim=1))
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, L)
        return self.chan_complex_to_last_dim(x)


class TGVN(nn.Module):
    """TGVN_1S (or, with guided=False, VarNetImage) behind `(y, E, sigma, guide)`."""

    pad_stride = 1                 # works at the measured size; NormUnet pads to 16 internally
    guide_domain = "kspace"        # the loop passes the side k-space, not a side image

    def __init__(self, guided=True, num_cascades=12, sens_chans=8, sens_pools=4, chans=18,
                 pools=4, cond_sigma=False, use_smaps=False, output="magnitude"):
        super().__init__()
        if output not in ("magnitude", "complex"):
            raise ValueError(f"output must be 'magnitude' or 'complex', got {output!r}")
        if output == "complex" and not use_smaps:
            raise ValueError("output='complex' needs use_smaps=True: the estimated maps set a "
                             "phase reference nothing outside the net shares")
        self.guided = bool(guided)
        self.cond_sigma = bool(cond_sigma)
        self.use_smaps = bool(use_smaps)
        self.output = output
        self.uses_operator_smaps = self.use_smaps       # read by the loops (online ESPIRiT)
        self.returns_magnitude = output == "magnitude"
        kws = dict(num_cascades=num_cascades, sens_chans=sens_chans, sens_pools=sens_pools,
                   chans=chans, pools=pools)
        self.net = TGVN_1S(**kws) if self.guided else VarNetImage(**kws)
        n_cond, noise_ch = self._cond_layout()
        if n_cond:
            for cascade in self.net.cascades:
                cascade.model = CondNormUnet(chans, pools, n_cond, noise_ch)
        if self.use_smaps:
            self.net.sens_net = None

    # -- conditioning ---------------------------------------------------------
    def _cond_layout(self):
        """(n_cond, noise channel indices) of the Phi planes."""
        return (1, (0,)) if self.cond_sigma else (0, ())

    def _set_cond(self, cond):
        for cascade in self.net.cascades:
            if isinstance(cascade.model, CondNormUnet):
                cascade.model.cond = cond

    @staticmethod
    def _per_sample(v, batch, device, name):
        if v is None:
            raise ValueError(f"this TGVN is conditioned on {name} but got None")
        v = torch.as_tensor(v, device=device, dtype=torch.float32).reshape(-1)
        return v.expand(batch) if v.numel() == 1 else v

    # -- conversions ----------------------------------------------------------
    @staticmethod
    def _mask(E, batch, device):
        """ImMAP mask (vertical phase-encode lines) -> the (B, 1, 1, W, 1) bool upstream uses."""
        m = get_mask(E)
        while m.dim() < 4:
            m = m.unsqueeze(0)
        col = m[..., :1, :]
        if not torch.equal(m.to(torch.bool), col.to(torch.bool).expand_as(m)):
            raise ValueError("TGVN expects a mask constant along the readout axis (vertical "
                             "phase-encode lines), as physics/mask.py::make_acc_mask produces.")
        return col.reshape(-1, 1, 1, col.shape[-1], 1)[:1].expand(
            batch, 1, 1, col.shape[-1], 1).to(device=device, dtype=torch.bool)

    @staticmethod
    def _real_view(k):
        if not torch.is_complex(k):
            raise ValueError(f"expected complex data, got {k.dtype}")
        return torch.view_as_real(k.resolve_conj().contiguous())

    def _side(self, guide, y):
        if not self.guided:
            if guide is not None:
                raise ValueError("TGVN(guided=False) is the no-side-information baseline but "
                                 "was handed a guide; use guided=True or drop the guide")
            return None
        if guide is None:
            raise ValueError("TGVN(guided=True) needs the side k-space as `guide`")
        if guide.dim() == 5:                   # (B, G, C, H, W): upstream takes ONE side scan
            if guide.shape[1] != 1:
                raise ValueError(f"TGVN takes one side scan, got {guide.shape[1]}")
            guide = guide[:, 0]
        if guide.shape != y.shape:
            raise ValueError(f"side k-space {tuple(guide.shape)} must match the main "
                             f"k-space {tuple(y.shape)}: TGVN reduces it with the main maps")
        return self._real_view(guide)

    def _sens(self, E, y):
        sm = get_sensitivity_map(E)
        if sm.shape[:2] != y.shape[:2] or sm.shape[-2:] != y.shape[-2:]:
            raise ValueError(f"coil maps {tuple(sm.shape)} do not match the k-space "
                             f"{tuple(y.shape)}; E must carry the maps for THIS batch")
        return self._real_view(sm)

    # -- the cascades on given maps (upstream forward minus its map estimate) ---------------
    def _cascades(self, image, ks, mask, side, sens):
        for cascade in self.net.cascades:
            if self.guided:
                image = cascade(image, ks, mask, side, sens, self.net.delta)
            else:
                image = cascade(image, ks, mask, sens)
        return image

    def _emit(self, image):
        """(B, 1, H, W, 2) iterate -> |x| (B, 1, H, W) or complex x (B, 1, H, W)."""
        if self.output == "complex":
            return torch.view_as_complex(image.contiguous())
        return complex_abs(image)

    def forward(self, y, E=None, sigma=None, guide=None, **_):
        if E is None:
            raise ValueError("TGVN needs the encoding operator to recover the sampling mask")
        B = y.shape[0]
        mask = self._mask(E, B, y.device)
        ks = self._real_view(y)
        side = self._side(guide, y)
        if self.cond_sigma:
            self._set_cond(self._per_sample(sigma, B, y.device, "sigma")[:, None])
        if not self.use_smaps:                 # upstream forward, untouched
            out = self.net(ks, mask) if side is None else self.net(ks, mask, side)
            return _check_out(out), {}
        sens = self._sens(E, y)
        image = self._cascades(sens_reduce(ks, sens), ks, mask, side, sens)
        return _check_out(self._emit(image)), {}

    def project(self):
        return self

    def extra_repr(self):
        return (f"guided={self.guided}, cascades={len(self.net.cascades)}, "
                f"cond_sigma={self.cond_sigma}, use_smaps={self.use_smaps}, "
                f"output={self.output}")


def _check_out(x):
    """Upstream returns complex_abs of the (B, 1, H, W, 2) iterate: already (B, 1, H, W)."""
    if x.dim() != 4 or x.shape[1] != 1:
        raise RuntimeError(f"unexpected TGVN output shape {tuple(x.shape)}")
    return x


class BridgeTGVN(BridgeScheduleMixin, TGVN):
    """TGVN as an I2SB x0-regressor (task "i2sb_recon"; contract in sb/i2sb_recon.py):

        x0_hat, _ = net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=side_kspace)

    The cascades start at the bridge state x_t (upstream starts at E^H y, which IS x1, the
    t = 1 end) and keep TGVN's data consistency on y and trust guidance on the side scan. Phi is
    conditioned on the bridge position -- log sigma_eff(t) mapped to [-1, 1], the SB nets'
    `s_log` -- and, with cond_sigma, on the measurement noise sigma_n.

    Known maps and complex output are forced: the bridge state is a complex image, and it must
    share its phase reference with the operator (training/i2sb_recon.py builds x0 under the
    same online maps, x0_source "operator"). Upstream TGVN estimates its maps; this one cannot.

    The schedule params MUST match cfg["i2sb"] (assert_schedule_matches runs at startup)."""

    def __init__(self, guided=True, num_cascades=12, sens_chans=8, sens_pools=4, chans=18,
                 pools=4, cond_sigma=True, kind="brownian", tau=0.1, n_points=1000,
                 beta_max=0.3):
        super().__init__(guided=guided, num_cascades=num_cascades, sens_chans=sens_chans,
                         sens_pools=sens_pools, chans=chans, pools=pools, cond_sigma=cond_sigma,
                         use_smaps=True, output="complex")
        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)

    def _cond_layout(self):
        # plane 0: bridge position; plane 1: sigma_n (a noise std, rescaled) when cond_sigma
        return (2, (1,)) if self.cond_sigma else (1, ())

    def forward(self, x_t, sigma=None, x1=None, y=None, E=None, sigma_n=None, guide=None,
                step=None):
        if y is None or E is None:
            raise ValueError("BridgeTGVN needs the measurement y and its operator E")
        if not torch.is_complex(x_t) or x_t.shape[1] != 1:
            raise ValueError(f"expected a complex (B, 1, H, W) bridge state, got "
                             f"{tuple(x_t.shape)} {x_t.dtype}")
        B = y.shape[0]
        if step is None:
            if sigma is None:
                raise ValueError("BridgeTGVN needs the bridge position: sigma (std_fwd) or step")
            step = self._step_from_sigma(sigma)
        step = torch.as_tensor(step, device=self.std_fwd.device).reshape(-1).long()
        step = step.expand(B) if step.numel() == 1 else step
        s_log = (torch.log(self.sigma_eff_tab[step].clamp_min(1e-12)) - self.s_mid) / self.s_half
        cond = [s_log.to(y.device)]
        if self.cond_sigma:
            cond.append(self._per_sample(sigma_n, B, y.device, "sigma_n"))
        self._set_cond(torch.stack(cond, dim=1))

        mask = self._mask(E, B, y.device)
        ks = self._real_view(y)
        side = self._side(guide, y)
        image = self._cascades(self._real_view(x_t), ks, mask, side, self._sens(E, y))
        return _check_out(self._emit(image)), None
