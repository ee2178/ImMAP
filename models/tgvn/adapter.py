"""
TGVN (trust-guided variational network) as an ImMAP model.

The networks are `_upstream.py`, vendored unmodified apart from its imports. This file is only the
calling-convention adapter, the same job models/e2evarnet/adapter.py does for E2E-VarNet:

                    ImMAP                              TGVN upstream
  measurement       y (B, C, H, W) complex             (B, C, H, W, 2) real-view
  operator          E = Mask @ FFT2D @ Sense           mask alone, broadcastable bool
  side information  guide = SIDE k-space (B, C, H, W)  second_kspace (B, C, H, W, 2)
  coil maps         ignored (uses_operator_smaps)      estimated by its own SensitivityModel
  noise level       sigma                              not an input
  output            (B, 1, H, W) magnitude             (B, 1, H, W) |x| (coil-combined, real)

`guide_domain = "kspace"` tells the recon loop to hand this model the side k-space itself, not a
coil-combined side image: TGVN reduces it with the MAIN scan's learned maps inside every
cascade (`TGVN_1S`), and that is part of the published method.

`guided=False` builds the paper's matched no-side-information baseline instead: upstream
`VarNetImage`, an image-domain E2E-VarNet with the same NormUnet (Cholesky-whitened complex
normalization) and SensitivityModel as TGVN, so the TGVN/VarNet pair differs only in the trust
guidance (the paper's Q1). It is NOT models/e2evarnet's fastMRI k-space VarNet.

Upstream defaults (`scripts/main_fastmri.py`) are num_cascades=12, chans=18; the paper's knee runs
use 14 cascades and 21 channels.
"""

import torch
import torch.nn as nn

from models.tgvn._upstream import TGVN_1S, VarNetImage
from operators.accessors import get_mask


class TGVN(nn.Module):
    """TGVN_1S (or, with guided=False, VarNetImage) behind `(y, E, sigma, guide)`."""

    pad_stride = 1                 # works at the measured size; NormUnet pads to 16 internally
    returns_magnitude = True
    uses_operator_smaps = False    # estimates its own maps; the loop skips online ESPIRiT
    guide_domain = "kspace"        # the loop passes the side k-space, not a side image

    def __init__(self, guided=True, num_cascades=12, sens_chans=8, sens_pools=4, chans=18,
                 pools=4):
        super().__init__()
        self.guided = bool(guided)
        kws = dict(num_cascades=num_cascades, sens_chans=sens_chans, sens_pools=sens_pools,
                   chans=chans, pools=pools)
        self.net = TGVN_1S(**kws) if self.guided else VarNetImage(**kws)

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
            raise ValueError(f"expected complex k-space, got {k.dtype}")
        return torch.view_as_real(k.resolve_conj().contiguous())

    def forward(self, y, E=None, sigma=None, guide=None, **_):
        if E is None:
            raise ValueError("TGVN needs the encoding operator to recover the sampling mask")
        mask = self._mask(E, y.shape[0], y.device)
        ks = self._real_view(y)
        if not self.guided:
            if guide is not None:
                raise ValueError("TGVN(guided=False) is the no-side-information baseline but "
                                 "was handed a guide; use guided=True or drop the guide")
            return self._out(self.net(ks, mask)), {}
        if guide is None:
            raise ValueError("TGVN(guided=True) needs the side k-space as `guide`")
        if guide.dim() == 5:                   # (B, G, C, H, W): upstream takes ONE side scan
            if guide.shape[1] != 1:
                raise ValueError(f"TGVN takes one side scan, got {guide.shape[1]}")
            guide = guide[:, 0]
        if guide.shape != y.shape:
            raise ValueError(f"side k-space {tuple(guide.shape)} must match the main "
                             f"k-space {tuple(y.shape)}: TGVN reduces it with the main maps")
        return self._out(self.net(ks, mask, self._real_view(guide))), {}

    @staticmethod
    def _out(x):
        """Upstream returns complex_abs of the (B, 1, H, W, 2) iterate: already (B, 1, H, W)."""
        if x.dim() != 4 or x.shape[1] != 1:
            raise RuntimeError(f"unexpected TGVN output shape {tuple(x.shape)}")
        return x

    def project(self):
        return self

    def extra_repr(self):
        return f"guided={self.guided}, cascades={len(self.net.cascades)}"
