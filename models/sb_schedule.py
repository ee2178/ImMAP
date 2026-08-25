# -*- coding: utf-8 -*-
"""
Shared Schrodinger-bridge scaffolding for the SB* unrolled nets (SBCDLNet, SBGroupCDL).

Everything here is prox-agnostic: the schedule tables, the sigma -> bridge-position lookup, the
debiased two-fidelity inputs, and the derived-parameter logging. What a concrete SB net adds is
the PRIOR -- and therefore the prox that ends each layer:

    SBCDLNet     l1 over the code         -> soft threshold          (models/sb_cdlnet.py)
    SBGroupCDL   nonlocal group sparsity  -> group threshold, Eq. 11 (models/sb_groupcdl.py)

The gradient step itself is identical in both, because it comes from the DATA terms, not the
prior. See models/sb_cdlnet.py's module docstring for the full derivation of the two-fidelity
scaffold; the short version is that at bridge step k the net sees x_t and must return x_0, with
x_1 known exactly, so

    r = x_t - mu_1 x_1  ~  mu_0 D_D z + sigma_sb eps        (target fidelity, debiased)
    c = cond            ~  D_P z                            (prior-contrast fidelity)

and mu_0 stays on the OPERATOR side so nothing is ever divided by it.

WHY THIS IS A MIXIN AND NOT A BASE CLASS. SBGroupCDL wants GroupCDL's attention/prox machinery
AND this; Python resolves "class SBGroupCDL(BridgeScheduleMixin, GroupCDL)" cleanly because this
class defines no __init__ of its own -- the subclass calls _init_bridge_tables() explicitly after
its own super().__init__() has run. Registering the buffers from a method rather than an __init__
also keeps the state_dict keys identical to the pre-refactor SBCDLNet.
"""

import torch
import torch.nn.functional as F

from operators.padding import calc_pad_2d
from sb.base import build_schedule, bridge_coeffs


def horner(coef, x):
    """Evaluate sum_d coef[d] * x**d by Horner's rule.

    `coef` is indexed on its FIRST axis by degree; any trailing axes broadcast against `x`. This
    serves both the scalar per-layer steps (coef (D+1,) -> scalar) and the per-channel thresholds
    (coef (D+1, M, 1, 1) -> (B, M, 1, 1))."""
    y = coef[-1]
    for d in range(coef.shape[0] - 2, -1, -1):
        y = y * x + coef[d]
    return y


class BridgeScheduleMixin:
    """Schedule tables + the bridge-position lookup, shared by every SB* net.

    A subclass must call `_init_bridge_tables(...)` once in its __init__, and must set `self.s`
    (the dictionary stride), `self.C`, `self.n_cond` and `self.prior_idx` before `bridge_inputs`
    is used.
    """

    # -----------------------------------------------------------------
    # tables
    # -----------------------------------------------------------------
    def _init_bridge_tables(self, kind="brownian", tau=0.19, n_points=1000, beta_max=0.3):
        """Register the schedule as buffers: saved with the checkpoint, moved by .to().

        Buffers rather than plain tensors matters at INFERENCE: `load_state_dict` then restores
        the schedule the run was actually trained with, even if model.params drifted afterwards.
        It does NOT make the net safe against a mismatch with cfg["i2sb"] -- the sampler steps on
        its own schedule object -- which is what `assert_schedule_matches` is for.
        """
        sched = build_schedule(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)
        mu0, mu1, std_sb = bridge_coeffs(sched)
        self.register_buffer("std_fwd", sched.std_fwd.clone())
        self.register_buffer("mu0_tab", mu0.clone())
        self.register_buffer("mu1_tab", mu1.clone())
        self.register_buffer("sigma_eff_tab", (std_sb / mu0).clone())
        # sigma_eff spans ~3 decades (0.01 -> 14), so neither polynomial gets it raw:
        #   steps     log(sigma_eff), then affinely mapped to [-1, 1] over the schedule's own
        #             range. Without the remap log spans [-4.6, 2.65], so s^2 reaches ~21 and the
        #             degree-2 coefficient sees a ~21x larger effective learning rate than the
        #             constant -- harmless at degree <= 1, bad conditioning at the degree 2+ this
        #             is meant to support. On [-1, 1] every monomial is bounded by 1.
        #   threshold sigma_eff / max(sigma_eff) in [0, 1]; it carries noise UNITS, so it must
        #             scale with sigma_eff rather than its log.
        self.register_buffer("sigma_ref", self.sigma_eff_tab.max().clone())
        _s = torch.log(self.sigma_eff_tab.clamp_min(1e-12))
        self.register_buffer("s_mid", 0.5 * (_s.max() + _s.min()))
        self.register_buffer("s_half", (0.5 * (_s.max() - _s.min())).clamp_min(1e-12))

    # -----------------------------------------------------------------
    # bridge coefficient lookup
    # -----------------------------------------------------------------
    def _step_from_sigma(self, sigma):
        """Invert std_fwd -> step index. std_fwd is strictly increasing, and predict_x0 passes
        values taken FROM this table, so this is exact; the neighbour comparison only guards
        against float round-trips."""
        s = torch.as_tensor(sigma, device=self.std_fwd.device,
                            dtype=self.std_fwd.dtype).reshape(-1).contiguous()
        n = self.std_fwd.shape[0]
        hi = torch.searchsorted(self.std_fwd, s).clamp(0, n - 1)
        lo = (hi - 1).clamp(0, n - 1)
        take_lo = (self.std_fwd[lo] - s).abs() < (self.std_fwd[hi] - s).abs()
        return torch.where(take_lo, lo, hi)

    def bridge_coefficients(self, sigma=None, step=None, batch=None):
        """(mu_0, mu_1, sigma_eff) at this bridge position, each shaped (B, 1, 1, 1).

        Give `step` directly, or `sigma` (the schedule's std_fwd, what predict_x0 passes) to
        recover it by table lookup. A single value broadcasts over the batch."""
        if step is None:
            if sigma is None:
                raise ValueError(
                    f"{type(self).__name__} needs the bridge position: pass `sigma` (= the "
                    f"schedule's std_fwd, as sb.base.predict_x0 does) or `step`.")
            step = self._step_from_sigma(sigma)
        step = torch.as_tensor(step, device=self.std_fwd.device).reshape(-1).long()
        if batch is not None:
            if step.numel() == 1:
                step = step.expand(batch)
            elif step.numel() != batch:
                raise ValueError(f"got {step.numel()} bridge positions for a batch of {batch}")
        v = lambda tab: tab[step].view(-1, 1, 1, 1)
        return v(self.mu0_tab), v(self.mu1_tab), v(self.sigma_eff_tab)

    def assert_schedule_matches(self, sched, atol=1e-6):
        """Fail loudly if this net's schedule differs from the one training/sampling is using.

        The bridge coefficients are recovered by inverting std_fwd, so a mismatch between
        model.params (kind/tau/n_points/beta_max) and cfg["i2sb"] would silently look up the
        WRONG mu_0/mu_1 at every step -- a bug that trains happily and produces nonsense. Call
        this once at startup with the schedule the training loop or the sampler built."""
        ref = sched.std_fwd.to(self.std_fwd.device)
        if ref.shape != self.std_fwd.shape:
            raise ValueError(
                f"schedule length mismatch: model was built with n_points={self.std_fwd.shape[0]}, "
                f"caller uses {ref.shape[0]}. Keep model.params and cfg['i2sb'] in sync.")
        d = float((ref - self.std_fwd).abs().max())
        if d > atol:
            raise ValueError(
                f"schedule mismatch: max |std_fwd difference| = {d:.3e} > {atol:g}. The model's "
                f"kind/tau/beta_max must equal cfg['i2sb']'s, or the bridge coefficients "
                f"recovered from sigma will be wrong.")

    # -----------------------------------------------------------------
    # the shared two-fidelity inputs
    # -----------------------------------------------------------------
    def bridge_inputs(self, y, sigma=None, step=None):
        """Split `y = cat([x_t, cond])` into everything a layer needs.

        Returns (r, c, dc, pad, mu0, s_log, s_hat):
            r      debiased target residual, padded       (B, 1, H', W')
            c      DC-removed conditioning stack, padded  (B, n_cond, H', W')
            dc     the scalar DC to re-add at readout     (B, 1, 1, 1)
            pad    the padding applied, for `unpad` at readout
            mu0    target-fidelity trust weight           (B, 1, 1, 1)
            s_log  log(sigma_eff) mapped to [-1, 1]    -- argument of the STEP polynomials
            s_hat  sigma_eff / max(sigma_eff) in [0, 1] -- argument of the THRESHOLD polynomial
        """
        x_t, cond = y[:, :1], y[:, 1:]
        if cond.shape[1] != self.n_cond:
            raise ValueError(
                f"expected {self.n_cond} conditioning channel(s) (C={self.C}), got {cond.shape[1]}")
        x1 = cond[:, self.prior_idx:self.prior_idx + 1]

        mu0, mu1, sig_eff = self.bridge_coefficients(sigma=sigma, step=step, batch=y.shape[0])

        # ---- DC: one term for both inputs, carried analytically ----
        # The dictionaries are (near) zero-mean, so the DC has to be handled outside them. Using
        # mean(x_1) -- available at train AND inference -- keeps the two fidelities consistent:
        # r - mu_0*dc ~ mu_0 * B_D z  and  x_1 - dc ~ B_P z  are then solved by the SAME code.
        # (Same trick as decode_dc="x1_mean" in sb/latent_i2sb.py.)
        dc = x1.mean(dim=(1, 2, 3), keepdim=True)
        r = x_t - mu1 * x1 - mu0 * dc                       # debiased bridge residual
        c = cond - cond.mean(dim=(2, 3), keepdim=True)      # per-channel DC removal

        pad = calc_pad_2d(*r.shape[2:], self.s)             # shared, to a multiple of the stride
        r = F.pad(r, pad, mode="reflect")
        c = F.pad(c, pad, mode="reflect")

        s_log = (torch.log(sig_eff.clamp_min(1e-12)) - self.s_mid) / self.s_half   # -> [-1, 1]
        s_hat = sig_eff / self.sigma_ref.clamp_min(1e-12)                          # -> [0, 1]
        return r, c, dc, pad, mu0, s_log, s_hat

    # -----------------------------------------------------------------
    # logging hook (visualization/params.py picks `param_logs` up automatically)
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _sb_param_logs(self, curves, probes=(0.0, 0.5, 1.0)):
        """Summarize the DERIVED per-layer quantities at a few bridge positions.

        A generic parameter walker can only see the raw polynomial coefficients, which say
        nothing on their own -- eta is sigmoid(poly(log sigma_eff)), so the coefficient values
        are not interpretable and the same eta can come from many of them. What you want to watch
        is the derived curve: is the step pinned at 0 or 1, is the threshold collapsing, do later
        unrolled layers behave differently from earlier ones.

        `curves` maps a name to fn(s_log, s_hat) -> a length-K list of tensors. `probes` are
        positions along the bridge (0 = target end, 1 = prior end).
        """
        out = {}
        n = self.std_fwd.shape[0]
        for t in probes:
            k = min(max(int(round(t * (n - 1))), 0), n - 1)
            sig = self.sigma_eff_tab[k]
            s_log = ((sig.clamp_min(1e-12).log() - self.s_mid) / self.s_half).view(1, 1, 1, 1)
            s_hat = (sig / self.sigma_ref.clamp_min(1e-12)).view(1, 1, 1, 1)
            tag = f"t{t:.2f}"
            for name, fn in curves.items():
                v = torch.stack([x.mean() for x in fn(s_log, s_hat)])
                out[f"{name}.{tag}.mean"] = float(v.mean())
                out[f"{name}.{tag}.first"] = float(v[0])       # depth dependence, cheaply
                out[f"{name}.{tag}.last"] = float(v[-1])
        out["sigma_eff.min"] = float(self.sigma_eff_tab.min())
        out["sigma_eff.max"] = float(self.sigma_eff_tab.max())
        return out
