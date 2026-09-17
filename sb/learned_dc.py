"""
sb/learned_dc.py -- learned data consistency for I2SB regressors.

Lets an ordinary unrolled net (CDLNet, ...) act as the I2SB x0-regressor with a LEARNED forward
operator E: CT1 -> T1 (models/forward_ops.py) supplying data consistency, instead of an SB* net's
learned prior fidelity. Configured under cfg["i2sb"]:

    "learned_dc": {
        "ckpt":    "trained_nets/.../E_unet_w16_l3_xc.pt",   # ladder checkpoint, or a train.py
                                                             #   net.ckpt (config.json beside it)
        "sigma_E": null                                      # E's residual std; null = the val rmse
                                                             #   stored in a ladder checkpoint
    }

With it, the loader's `cond` goes to E (its side information, e.g. [T2, FLAIR]) and NOT into the
net's input: predict_x0 hands the net x_t alone plus a per-batch BridgeDCOperator
(operators/learned.py) built here from the bridge coefficients at the current step, x1 (= T1, the
bridge prior AND E's measurement) and cond. The net's C is therefore 1.
"""

import os

import torch
import yaml

from models.forward_ops import ForwardOp, load_forward_op
from operators.learned import LearnedOperator, BridgeDCOperator
from sb.base import bridge_coeffs


def _load_E(ckpt, device):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    val_rmse = None
    if isinstance(blob, dict) and "spec" in blob:                  # scripts/fit_forward_ladder.py
        E = load_forward_op(ckpt, map_location="cpu", freeze=True)
        val_rmse = (blob.get("val") or {}).get("rmse")
    elif isinstance(blob, dict) and "model_state_dict" in blob:    # train.py task forward_op
        cfg_path = os.path.join(os.path.dirname(ckpt), "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{ckpt} is a train.py checkpoint but {cfg_path} is missing; "
                                    f"it holds the ForwardOp's model.params")
        with open(cfg_path) as f:
            mcfg = yaml.safe_load(f)["model"]
        if mcfg["type"] != "ForwardOp":
            raise ValueError(f"{ckpt}: model.type is {mcfg['type']!r}, not ForwardOp")
        E = ForwardOp(**mcfg["params"])
        E.load_state_dict(blob["model_state_dict"])
        E.requires_grad_(False).eval()
    else:
        raise ValueError(f"{ckpt}: not a ForwardOp checkpoint")
    return E.to(device), val_rmse


class LearnedBridgeDC:
    """Frozen E + the schedule's bridge coefficients; `bind` makes the per-batch operator."""

    def __init__(self, sched, device, ckpt, sigma_E=None, feed_cond=False):
        self.E, val_rmse = _load_E(ckpt, device)
        if sigma_E is None:
            if val_rmse is None:
                raise ValueError(f"learned_dc.sigma_E is null but {ckpt} stores no val rmse; set "
                                 f"sigma_E to E's validation residual std")
            sigma_E = float(val_rmse)
        if not sigma_E > 0:
            raise ValueError(f"learned_dc.sigma_E must be > 0, got {sigma_E}")
        self.sigma_E = float(sigma_E)
        self.feed_cond = bool(feed_cond)        # also concatenate cond into the net input
        self.std_fwd = sched.std_fwd.to(device)
        mu0, mu1, std_sb = bridge_coeffs(sched)
        self.mu0, self.mu1, self.std_sb = mu0.to(device), mu1.to(device), std_sb.to(device)
        print(f"[learned_dc] E from {ckpt}: inputs="
              f"{'x' if not self.E.cond_channels else ('x+c' if self.E.use_x else 'c')}, "
              f"cond_channels={self.E.cond_channels}, sigma_E={self.sigma_E:.4g}")
        if not self.E.use_x:
            raise ValueError("learned_dc: E ignores x (use_x=False); its data gradient is zero")

    @property
    def cond_channels(self):
        return self.E.cond_channels

    def step_from_sigma(self, sigma):
        """std_fwd -> step index (std_fwd is strictly increasing; callers pass table values)."""
        s = torch.as_tensor(sigma, device=self.std_fwd.device,
                            dtype=self.std_fwd.dtype).reshape(-1).contiguous()
        n = self.std_fwd.shape[0]
        hi = torch.searchsorted(self.std_fwd, s).clamp(0, n - 1)
        lo = (hi - 1).clamp(0, n - 1)
        take_lo = (self.std_fwd[lo] - s).abs() < (self.std_fwd[hi] - s).abs()
        return torch.where(take_lo, lo, hi)

    def bind(self, sigma, x1, cond):
        """The BridgeDCOperator for this batch at bridge noise `sigma` (std_fwd, (B,1,1,1) or (B,))."""
        if x1 is None:
            raise ValueError("learned_dc needs x1 (T1): pass x1= to predict_x0")
        n_c = 0 if cond is None else cond.shape[1]
        if n_c != self.E.cond_channels:
            raise ValueError(f"learned_dc: E expects {self.E.cond_channels} cond channel(s) but "
                             f"the batch has {n_c}; set data.*.cond_idx to E's side information "
                             f"(e.g. [3, 0] for T2, FLAIR)")
        step = self.step_from_sigma(sigma)
        if step.numel() == 1 and x1.shape[0] > 1:
            step = step.expand(x1.shape[0])
        v = lambda tab: tab[step].view(-1, 1, 1, 1)
        E = LearnedOperator(self.E, cond)
        return BridgeDCOperator(E, x1=x1, t1=x1, mu0=v(self.mu0), mu1=v(self.mu1),
                                std_sb=v(self.std_sb), sigma_E=self.sigma_E)
