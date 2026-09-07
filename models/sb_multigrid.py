# -*- coding: utf-8 -*-
"""
SBMGCDLNet -- SBCDLNet's two-fidelity Schrodinger-bridge problem, solved by multigrid V-cycles.

The motivation is the observation that the ADM UNet beats the flat unrolled nets on this task and
the main structural difference is MULTIRESOLUTION. T1 -> T1ce is jointly a denoising problem and a
tumour-detection problem: the first is local, the second is not. Enhancing tumour is a
spatially coherent region tens of pixels across, so deciding *where* it is wants a receptive field
much wider than a stack of 7x7 convolutions at one scale provides, and is cheaper to decide on a
coarse grid. A V-cycle gives both -- the coarse levels see the whole lesion at once, and the FAS
correction carries that decision back to the fine grid.

HOW THIS REUSES THE V-CYCLE UNCHANGED
-------------------------------------
SBCDLNet's layer has TWO fidelity terms and two dictionary pairs, which looks like it needs a
multigrid built for two-term objectives. It does not. Stack the pairs into one
(1 + n_cond)-channel dictionary and put the bridge weight in the MEASUREMENT OPERATOR:

    E = ChannelGain(diag(mu_0, 1, ..., 1))    (operators/gain.py)
    y = [r ; c]      r = x_t - mu_1 x_1 - mu_0 dc   (debiased target residual)
                     c = cond - mean(cond)          (DC-removed conditioning stack)

Then the ordinary LISTA step `A(E^H E B z - E^H y)` equals

    mu_0 A_D(mu_0 B_D z - r) + A_P(B_P z - c)

term for term -- i.e. exactly SBCDLNet's step. See operators/gain.py for the derivation, and
tests for the numerical check. Every piece of models/multigrid.py -- the V-cycle, the FAS
correction pi, the Galerkin coarse operator, the transfer pair -- then applies with no changes,
because from its point of view this is a plain CDL problem with an unusual E.

WHAT THIS COSTS RELATIVE TO SBCDLNet
------------------------------------
ONE STEP, NOT TWO. SBCDLNet gives its two fidelities separate learned steps (eta_k, nu_k), which
it describes as a block-preconditioned step rather than a literal proximal-gradient step on J_k.
Folding both into one operator forces a single step, so this net is the *more* faithful
proximal-gradient iteration and the *less* flexible one. The learned per-channel threshold
absorbs part of the difference, as it does in CDLNet.

The bridge conditioning that remains is therefore: mu_0 and mu_1 STRUCTURALLY, through the
operator and the debiased residual (unchanged from SBCDLNet), plus the prox threshold's learned
polynomial in sigma_eff. `degrees` must be >= 1 or that polynomial is a constant and the only
step-awareness left is the structural part.

WHAT STAYS THE SAME
-------------------
The debiased residual, the analytic DC (mean of x_1, available at train and inference), the
sigma -> bridge-position lookup and the schedule tables all come from BridgeScheduleMixin, shared
verbatim with SBCDLNet and SBGroupCDL. `assert_schedule_matches` still guards model.params
against cfg["i2sb"], and train_i2sb still calls it at startup.

Calling convention matches the rest of the repo's regressors, so sb.base.predict_x0 works
unchanged:

    x0_hat, z = net(y, E=Identity(), sigma=std_fwd)

with `y = cat([x_t, cond])`. The `E` argument is accepted and IGNORED -- this net builds its own
operator from the bridge step, and pure translation has no measurement operator anyway.

REQUIRES a conditioning channel: the bridge prior x_1 must be in `cond` (set the loader's
`cond_idx` to include it and point `prior_idx` at its position), so C = 1 + len(cond_idx) >= 2.
"""

import torch
import torch.nn as nn

from models.multigrid import MGCDLNet
from models.sb_schedule import BridgeScheduleMixin
from operators.gain import bridge_gain
from operators.padding import unpad


class SBMGCDLNet(BridgeScheduleMixin, nn.Module):
    """Two-fidelity Schrodinger-bridge regression by multigrid V-cycles.

    Parameters
    ----------
    K            `[K_outer, [i0, i1, ...]]` -- K_outer V-cycles, `i_l` smoothing iterations at
                 level l (finest first; every non-coarsest entry must be even). A bare int gives
                 a flat CDLNet from the same blocks, which is the no-multigrid ablation.
    M, P, s      atoms, filter side, stride -- as in CDLNet. `s` is the LATENT stride; the input
                 is padded to a multiple of `s * 2**(levels-1)` so every level halves exactly.
    C            input width, = 1 + len(cond_idx). Must be >= 2 (the prior lives in cond).
    prior_idx    which CONDITIONING channel is the bridge prior x_1 (0-based within cond). With
                 cond_idx=[0,1,3] = [FLAIR,T1,T2] and x1_idx=1 (T1), this is 1.
    widen        subband multiplier per coarse level (M * widen**l). 1 keeps the parameter count
                 close to a flat CDLNet, which is what isolates multigrid from capacity.
    degrees      degree of the threshold polynomial in sigma_eff. MUST be >= 1 or the prox is a
                 constant and the net loses its learned step-awareness (see above).
    tau0, alpha0, eta0, Mh, W, dK, ...
                 passed through to MGCDLNet; `W > 1` with `Mh` set gives the nonlocal group prox.
    kind, tau, n_points, beta_max
                 the bridge schedule -- MUST match cfg["i2sb"].
    """

    def __init__(self, K=(3, (8, 8, 4)), M=169, C=2, P=7, s=2, prior_idx=0,
                 widen=1, degrees=1, tau0=1e-3, alpha0=0.1, eta0=0.1, eta_degrees=0,
                 alpha_conv=True, Mh=None, W=1, dK=1, sim_fun="distance", nheads=1,
                 rho0=1.0, gamma0=0.8, init_strategy="spectral_norm",
                 subgrad_mode="rigorous", attn_backend="gather", flex_block_size=128,
                 julia_compat=False, transfer_filter=None, learn_transfer=False,
                 is_complex=False, resize_noise=False,
                 kind="i2sb", tau=0.19, n_points=1000, beta_max=0.3, init=True):
        super().__init__()

        if C < 2:
            raise ValueError(
                f"SBMGCDLNet needs the bridge prior x_1 as a conditioning channel, so "
                f"C = 1 + len(cond_idx) >= 2; got C={C}. Set the loader's cond_idx to include "
                f"the contrast used as x1_idx.")
        self.C = int(C)
        self.n_cond = self.C - 1
        if not 0 <= prior_idx < self.n_cond:
            raise ValueError(
                f"prior_idx={prior_idx} out of range for {self.n_cond} conditioning channel(s). "
                f"It indexes cond (0-based), not the stored contrasts.")
        self.prior_idx = int(prior_idx)
        if int(degrees) < 1:
            raise ValueError(
                f"degrees={degrees} makes the prox threshold a CONSTANT in sigma_eff, so the "
                f"only bridge-awareness left is the structural mu_0/mu_1 weighting. Use "
                f"degrees >= 1 (1 reproduces CDLNet's adaptive threshold, 2 spans the "
                f"sigma^2 law).")

        # preproc="identity": bridge_inputs already removes the DC analytically and pads, so the
        # image-mode mean subtraction would double-count it and the padding would be redundant.
        self.mg = MGCDLNet(
            K=K, M=M, C=self.C, P=P, s=s, Mh=Mh, W=W, widen=widen, tau0=tau0,
            degrees=int(degrees), eta0=eta0, eta_degrees=eta_degrees, alpha0=alpha0,
            alpha_conv=alpha_conv, is_complex=is_complex, dual=False, dK=dK,
            sim_fun=sim_fun, nheads=nheads, rho0=rho0, gamma0=gamma0,
            init_strategy=init_strategy, subgrad_mode=subgrad_mode,
            attn_backend=attn_backend, flex_block_size=flex_block_size,
            preproc="identity", resize_noise=resize_noise, julia_compat=julia_compat,
            transfer_filter=transfer_filter, learn_transfer=learn_transfer)

        self.M, self.P = int(M), int(P)
        # BridgeScheduleMixin.bridge_inputs pads to a multiple of `self.s`. The V-cycle needs
        # every level to halve exactly on BOTH the image and the latent grid, so the padding
        # stride is the whole pad_stride, not just the dictionary stride.
        self.s = self.mg.pad_stride
        self.stride = int(s)                          # the dictionary stride, for reference
        self.levels = self.mg.levels
        # train.py compiles the fused FlexAttention kernel via this attribute; without it a
        # group-prox variant silently runs the uncompiled kernel.
        self.attn_backend = attn_backend

        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)
        if not init:
            pass          # MGCDLNet spectral-initialises at construction; nothing to undo

    # `visualization/filters.py` renders net.A / net.B. Expose the finest level's pair so the
    # filter logging shows the dictionary that actually reads out.
    @property
    def A(self):
        from models.multigrid import first_layer
        return nn.ModuleList([first_layer(self.mg.lista.first).analysis])

    @property
    def B(self):
        from models.multigrid import first_layer
        return nn.ModuleList([first_layer(self.mg.lista.first).synthesis])

    @property
    def D(self):
        return self.mg.D

    # -----------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, step=None):
        """`y = cat([x_t, cond], dim=1)`, the tensor sb.base.predict_x0 builds. `E` is accepted
        for signature parity and IGNORED -- the bridge operator is built here. Returns
        (x0_hat, z)."""
        r, c, dc, pad, mu0, s_log, s_hat = self.bridge_inputs(y, sigma=sigma, step=step)

        # The two fidelities as ONE measurement problem (see operators/gain.py):
        #   observation  [r ; c]        operator  diag(mu_0, 1, ..., 1)
        Eg = bridge_gain(mu0, self.n_cond)
        obs = torch.cat([r, c], dim=1)

        # sigma_eff as a (B,) vector, not (B,1,1,1): restrict_noise only resamples a 4-D sigma,
        # and a per-sample scalar has no map to resample. It still gets the per-level noise_scale
        # factor, which is right -- sigma_eff IS the effective denoising level of the bridge step,
        # so coarse-grid averaging genuinely lowers it.
        x, z = self.mg(obs, E=Eg, sigma=s_hat.reshape(-1))

        # channel 0 of the read-out is B_D z, the target-domain synthesis
        x0_hat = unpad(x[:, :1], pad) + dc
        return x0_hat, z

    # -----------------------------------------------------------------
    def compile_flex(self):
        self.mg.compile_flex()
        return self

    @torch.no_grad()
    def project(self):
        """Constraint projection, recursing over every submodule that defines `project_` --
        the same contract MGCDLNet uses, so a new constrained block is covered automatically."""
        for m in self.modules():
            if hasattr(m, "project_"):
                m.project_()

    @torch.no_grad()
    def param_logs(self, probes=(0.0, 0.5, 1.0)):
        """The threshold the FINEST level will actually use at a few bridge positions, plus the
        schedule's own range. The raw polynomial coefficients say little on their own; what is
        worth watching is whether the threshold collapses or stops varying with the step."""
        from models.multigrid import first_layer
        out = {}
        prox = getattr(first_layer(self.mg.lista.first), "prox", None)
        thr = getattr(prox, "tau", None) if prox is not None else None
        n = self.std_fwd.shape[0]
        for t in probes:
            k = min(max(int(round(t * (n - 1))), 0), n - 1)
            s_hat = (self.sigma_eff_tab[k] / self.sigma_ref.clamp_min(1e-12)).view(1)
            if thr is not None:
                try:
                    out[f"tau.t{t:.2f}.mean"] = float(thr(s_hat).mean())
                except Exception:
                    pass
        out["sigma_eff.min"] = float(self.sigma_eff_tab.min())
        out["sigma_eff.max"] = float(self.sigma_eff_tab.max())
        return out

    def extra_repr(self):
        return (f"C={self.C} (1 + {self.n_cond} cond), prior_idx={self.prior_idx}, "
                f"levels={self.levels}, pad_stride={self.s}")
