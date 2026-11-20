"""
SBReconCDL -- SBCDLNet (models/sb_cdlnet.py) as an I2SB RECONSTRUCTION regressor, task "i2sb_recon".

SBCDLNet is the bridge's own unrolled sparse coder: one shared code z explains the target through
D_D and the bridge start through D_P, and its target fidelity uses the debiased residual
x_t - mu_1 x_1. It is called the I2SB way, `net(cat([x_t, cond]), sigma=std_fwd)`, with the start
x1 in the conditioning stack. This adapter gives it the recon loops' interface,

    x0_hat, extra = net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n)

with complex images in and out (complex filters: `complex=True`), and x1 as the single conditioning
channel (C = 2, prior_idx = 0). It reads only x_t and x1 -- never y or E -- so it is
`image_domain_only` and trains cropped to batch (training/i2sb_recon.py "CROP TO BATCH").

Every other SBCDLNet parameter passes through (K, M, P, s, t0, deg_eta, deg_tau, init, s_mode, ...);
the schedule parameters must match cfg["i2sb"], as for every SB* net.
"""

import torch
import torch.nn as nn

from models.sb_cdlnet import SBCDLNet


class SBReconCDL(nn.Module):
    """SBCDLNet on complex [x_t, x1] -> complex x0_hat."""

    pad_stride = 1                 # SBCDLNet pads to its stride itself; the bridge stays on its grid
    guide_domain = "image"
    image_domain_only = True       # reads x_t and x1 only (see the module docstring)
    cond_x1 = True                 # x1 is its prior-fidelity input, always
    guided = False

    def __init__(self, **cdl_kws):
        super().__init__()
        for k in ("C", "prior_idx", "complex"):
            if k in cdl_kws:
                raise ValueError(f"SBReconCDL sets SBCDLNet's {k} itself (complex [x_t, x1])")
        self.cdl = SBCDLNet(C=2, prior_idx=0, complex=True, **cdl_kws)

    def assert_schedule_matches(self, sched, atol=1e-6):
        self.cdl.assert_schedule_matches(sched, atol=atol)

    def project(self):
        """SBCDLNet's constraints (unit filters, nonnegative thresholds), after every step."""
        self.cdl.project()

    def forward(self, x_t, sigma=None, x1=None, y=None, E=None, sigma_n=None, step=None,
                guide=None):
        if not torch.is_complex(x_t):
            raise TypeError(f"SBReconCDL expects a complex bridge state, got {x_t.dtype}")
        if x_t.shape[1] != 1:
            raise ValueError(f"expected a single-channel image (B, 1, H, W), got {tuple(x_t.shape)}")
        if x1 is None or x1.shape != x_t.shape:
            raise ValueError("SBReconCDL needs x1, the bridge start, shaped like x_t")
        if guide is not None:
            raise ValueError("SBReconCDL is unguided: the bridge start x1 is its only side input")
        out, _ = self.cdl(torch.cat([x_t, x1.to(x_t.dtype)], dim=1), sigma=sigma, step=step)
        return out, None
