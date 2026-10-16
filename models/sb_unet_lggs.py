"""
SBUnetLGGS -- a bridge UNet whose CT1 estimate is refined by K guided primal-dual sweeps.

At bridge step t the UNet sees x_t and the current study's contrasts -- exactly the plain
`i2sb_unet_all` SBUnet -- and returns a CT1 estimate u. LGGS then treats u as the MEASUREMENT
(E = Identity) and the prior study's CT1 as the guide:

    u      = U_theta(x_t, cond; sigma_t)
    x0_hat = LGGS_K(u; guides = prior CT1 planes, sigma_t)
           ~ argmin_x  1/2 ||x - u||^2  +  lambda R(A x; A w)

so x0_hat is (approximately) the guided group-sparse prox of the UNet output. One sweep k,
with x^0 = u and z^0 = clip_0(A_0 u; A_0 w):

    x^{k+1} = x^k - tau_k(sigma) [ (x^k - u) + B_k z^k ]
    xb      = x^{k+1} + theta_k(sigma) (x^{k+1} - x^k)
    z^{k+1} = clip_k( z^k + A_k xb ; A_k w )                 models/guided_lpds.py

The prior study reaches the estimate ONLY through the clip's adjacency: it decides which
coefficients of u survive, never adds its own intensities. (`unet_guides` can also feed some
guide planes to the UNet as input channels, the `pair_w2` way; the default [] keeps the UNet
blind to the prior study.) With `guided=False` the same sweeps
run self-only (the unguided-sweeps control): any gain of the guided arm over it is the prior
study acting through the prior, not extra refinement capacity.

`last_u` is the UNet estimate of the last forward, kept with its graph so the trainer can add
`u_weight * loss(u, x0)` (training/i2sb.py) -- the term that keeps u a CT1 estimate rather than
whatever pre-image LGGS maps best onto CT1.

WARM START. `load_state_dict` also accepts a plain SBUnet state dict (no `unet.` / `lggs.`
prefixes) and loads it into the UNet alone, leaving LGGS at its init. So `paths.init_ckpt` --
which train.py pops before saving the run's config and never re-applies on a requeue -- can
point straight at a trained SBUnet checkpoint. LGGS starts near the identity (lam0 small: the
dual is clipped to ~lam0, so B z ~ 0 and x^K ~ u), so the warm-started net starts near the UNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.guided_lpds import LGGSNet
from models.sb_unet import SBUnet


class SBUnetLGGS(nn.Module):
    """SBUnet -> LGGSNet(E = Identity), trained end to end as one bridge regressor.

    C               net input width the trainer checks: target_channels + len(cond_idx). The
                    guide planes arrive SEPARATELY (guide_as_cond: false).
    n_guide_planes  how many planes the loader delivers (guide_window and guide_idx decide it).
    lggs_guides     indices into those planes that LGGS uses as guides; with guide_idx [2] and
                    guide_window 2 the planes are CT1 at z-2..z+2, so [0, 1, 2, 3, 4].
    unet_guides     indices of planes ALSO appended to the UNet input, in that order (default []:
                    the UNet sees only x_t and cond, and a plain SBUnet checkpoint of C =
                    target_channels + len(cond_idx) loads as-is).
    guided          False = the same sweeps with no guide (self-only prox): the control arm.
    unet, lggs      the SBUnet / LGGSNet params; unet.C is filled in as C + len(unet_guides).
    """

    def __init__(self, C, n_guide_planes, lggs_guides, unet, lggs, unet_guides=(),
                 guided=True):
        super().__init__()
        self.C = int(C)
        self.n_guide_planes = int(n_guide_planes)
        self.lggs_guides = [int(i) for i in lggs_guides]
        self.unet_guides = [int(i) for i in unet_guides]
        for name, idx, need in (("lggs_guides", self.lggs_guides, True),
                                ("unet_guides", self.unet_guides, False)):
            if any(not 0 <= i < self.n_guide_planes for i in idx) or (need and not idx):
                raise ValueError(f"{name} {idx} must be a{' non-empty' if need else ''} subset "
                                 f"of range({self.n_guide_planes})")
        self.guided = bool(guided)

        unet = dict(unet)
        unet_C = self.C + len(self.unet_guides)
        if "C" in unet and int(unet["C"]) != unet_C:
            raise ValueError(f"unet.C={unet['C']} but C + len(unet_guides) = {unet_C}; "
                             f"leave unet.C out")
        unet["C"] = unet_C
        self.unet = SBUnet(**unet)

        lggs = dict(lggs)
        if lggs.get("preproc", "image") == "kspace":
            raise ValueError("the LGGS stage acts on an image estimate: preproc 'image' or "
                             "'identity', not 'kspace'")
        lggs.setdefault("preproc", "image")
        lggs.setdefault("C", 1)
        self.lggs = LGGSNet(**lggs)
        self.attn_backend = self.lggs.attn_backend
        self.last_u = None

    # -- forward ------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, guide=None):
        """`y = cat([x_t, cond])` (predict_x0 builds it), `guide` (B, G, 1, H, W). -> (x0_hat, state).
        `E` is accepted for signature parity and ignored: the LGGS stage's operator is the
        identity on the UNet estimate."""
        if guide is None or guide.dim() != 5 or guide.shape[1] != self.n_guide_planes:
            got = None if guide is None else tuple(guide.shape)
            raise ValueError(f"SBUnetLGGS needs the loader's {self.n_guide_planes} guide planes "
                             f"as (B, G, 1, H, W) with guide_as_cond: false; got {got}")
        unet_in = y
        if self.unet_guides:
            unet_in = torch.cat([y, guide[:, self.unet_guides].flatten(1, 2)], dim=1)
        u, _ = self.unet(unet_in, sigma=sigma)
        u = u[:, :1]
        self.last_u = u
        g = guide[:, self.lggs_guides] if self.guided else None
        x, state = self.lggs(u, guide=g, sigma=sigma)
        return x, state

    # -- hooks the trainer looks for ------------------------------------------
    @torch.no_grad()
    def project(self):
        self.lggs.project()

    def compile_flex(self):
        self.lggs.compile_flex()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """A full SBUnetLGGS state dict, or a plain SBUnet one (the warm start: UNet only)."""
        if not any(k.startswith(("unet.", "lggs.")) for k in state_dict):
            print(f"[SBUnetLGGS] plain SBUnet state dict ({len(state_dict)} tensors): loading it "
                  f"into the UNet; LGGS keeps its initialisation")
            return self.unet.load_state_dict(state_dict, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def extra_repr(self):
        return (f"C={self.C}, n_guide_planes={self.n_guide_planes}, "
                f"lggs_guides={self.lggs_guides}, unet_guides={self.unet_guides}, "
                f"guided={self.guided}")
