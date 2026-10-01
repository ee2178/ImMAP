"""
CycleSynth -- a synthesis net trained jointly with a backward map, through a cycle loss.

    G : input stack (FLAIR, T1, T2 [, prior-study guides]) -> CT1      the synthesis net
    F : (CT1, side information)                             -> T1       the backward map

trained on

    L = loss(G(X), CT1)  +  cycle_weight * loss(F(G(X), side), T1)

(training/synthesis.py, `cycle_weight`). The second term asks that the synthesised CT1 still
EXPLAIN this session's T1: whatever G invents has to survive being mapped back. It is the
training-time counterpart of learned data consistency -- the same CT1 -> T1 direction as
models/forward_ops.ForwardOp, but learned jointly with G instead of frozen in front of it.

`forward(X)` is G alone, so everything that treats this as an ordinary synthesis net -- the
validation loop, metrics, notebooks, eval -- sees one input stack in and one CT1 out.

THE SIDE INFORMATION MUST NOT CONTAIN T1. F's inputs are the CT1 estimate plus `side_idx`
channels of X (default FLAIR, T2). If T1 were among them, F could return it and the cycle term
would be zero whatever G did; `__init__` refuses that configuration.

Two backward maps are supported, by spec:
    {"type": "ForwardOp", ...}   a residual operator CT1 + net(CT1, side), zero-initialised so it
                                 starts as the identity (T1 ~ CT1). kind="unet" with levels=1 is
                                 the shallow backward path.
    {"type": "Unet2D", ...}      any plain image-to-image net, fed cat([CT1, side]). Built with the
                                 same params as G (bar in_chans) it is "the same model in both
                                 directions".
"""

import torch
import torch.nn as nn


def _build(spec):
    from models import build_model          # local: models/__init__ imports this module
    if not isinstance(spec, dict) or "type" not in spec:
        raise ValueError(f"a CycleSynth net spec is {{'type': ..., 'params': {{...}}}}, got {spec!r}")
    return build_model({"model": {"type": spec["type"], "params": dict(spec.get("params", {}))}})


class CycleSynth(nn.Module):
    def __init__(self, forward, backward, src_idx=1, side_idx=(0, 2)):
        super().__init__()
        self.src_idx = int(src_idx)
        self.side_idx = [int(i) for i in side_idx]
        if self.src_idx in self.side_idx:
            raise ValueError(f"side_idx {self.side_idx} contains src_idx {self.src_idx}: F would "
                             f"be handed the T1 it is asked to predict, and the cycle term would "
                             f"be satisfied by copying it")
        self.G = _build(forward)
        self.F = _build(backward)
        self.backward_type = backward["type"]
        n_side = len(self.side_idx)
        if self.backward_type == "ForwardOp":
            if getattr(self.F, "cond_channels", None) != n_side:
                raise ValueError(f"backward ForwardOp has cond_channels="
                                 f"{getattr(self.F, 'cond_channels', None)} but side_idx gives "
                                 f"{n_side} channel(s)")

    def forward(self, X):
        """G alone: the CT1 estimate. Unrolled-style tuple outputs are reduced to the image."""
        out = self.G(X)
        return out[0] if isinstance(out, (tuple, list)) else out

    def t1(self, X):
        """This session's T1 -- the cycle target -- as (B, 1, H, W)."""
        return X[:, self.src_idx:self.src_idx + 1]

    def back(self, ct1, X):
        """F(CT1, side) -> T1 estimate."""
        side = X[:, self.side_idx]
        if self.backward_type == "ForwardOp":
            out = self.F(ct1, side)
        else:
            out = self.F(torch.cat([ct1, side], dim=1))
        return out[0] if isinstance(out, (tuple, list)) else out

    def extra_repr(self):
        return (f"src_idx={self.src_idx}, side_idx={self.side_idx}, "
                f"backward={self.backward_type}")
