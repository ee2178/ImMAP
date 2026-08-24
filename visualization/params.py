# -*- coding: utf-8 -*-
"""
Scalar logging for a network's learnable parameters -- thresholds, step sizes, extrapolation
weights, anything.

`visualization/filters.py` renders what the DICTIONARIES look like; this renders what the
SCALARS are doing. For an unrolled net those scalars carry most of the interpretable behaviour:
CDLNet's per-layer soft-threshold `t`, SBCDLNet's step sizes, IPALMNet's `eta` / `beta` / `theta`.
Watching them over training answers questions a loss curve cannot -- whether thresholds are
collapsing to zero, whether a learned step is pinned at its bound, whether later unrolled layers
are doing anything at all.

Two layers, both optional:

    get_param_logs(net)   generic. Summary scalars (mean/std/min/max, and grad norm when a
                          backward has run) for every trainable tensor, plus a per-LAYER
                          breakdown when a tensor's leading axis indexes unrolled iterations.
                          Works on any nn.Module and knows nothing about this repo.

    net.param_logs()      optional hook. A model returning {name: float} here gets it merged in
                          under the same prefix. Use it for quantities that are not parameters
                          but functions of them -- e.g. a learned step evaluated at several
                          noise levels, which no generic walker could reconstruct.

INVARIANT, same as visualization/filters.py: this must never raise. Callers wrap it loosely or
not at all, and a logging helper must not be able to kill a training run. Anything unexpected is
skipped, and a total failure returns {}.
"""

import torch


def _finite(x):
    """float(x) if it is a usable number, else None -- keeps NaN/inf out of the logs."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def _summary(t):
    """mean / std / min / max of one tensor, as plain floats."""
    t = t.detach()
    if t.is_complex():
        t = t.abs()
    t = t.float().reshape(-1)
    if t.numel() == 0:
        return {}
    out = {"mean": _finite(t.mean()), "min": _finite(t.min()), "max": _finite(t.max())}
    if t.numel() > 1:
        out["std"] = _finite(t.std())
    return {k: v for k, v in out.items() if v is not None}


def _layer_axis(p, K):
    """True when p's leading axis indexes unrolled layers, so a per-layer split is meaningful.

    The heuristic is deliberately narrow: dim0 must equal the model's own K. Parameters that
    live in a ModuleList (CDLNet's A/B) already carry the layer index in their NAME and are not
    matched here -- only stacked tensors like CDLNet's t (K, 2, M, 1, 1) are."""
    return K is not None and p.ndim >= 1 and p.shape[0] == K


def get_param_logs(net, prefix="params", with_grad=True, per_layer=True,
                   max_layers=64, max_scalars=4000):
    """-> {f"{prefix}/...": float} ready to hand straight to wandb.log.

    Parameters
    ----------
    with_grad   also log ||grad|| per tensor. Only populated after backward() and before
                zero_grad(), so call this right after the optimizer step (or during validation,
                where it is simply absent).
    per_layer   additionally emit one scalar per unrolled layer for stacked parameters -- this
                is what turns "threshold mean = 0.02" into a curve over depth.
    max_layers  skip the per-layer split for very deep stacks.
    max_scalars hard cap on how many series this can create, so a large model cannot silently
                flood the run with thousands of wandb panels.
    """
    logs = {}
    try:
        K = getattr(net, "K", None)
        if not isinstance(K, int):
            K = None

        for name, p in net.named_parameters():
            if not p.requires_grad or len(logs) >= max_scalars:
                continue
            base = f"{prefix}/{name}"
            for stat, val in _summary(p).items():
                logs[f"{base}.{stat}"] = val

            if with_grad and p.grad is not None:
                g = _finite(p.grad.detach().float().norm())
                if g is not None:
                    logs[f"{base}.grad_norm"] = g

            if per_layer and _layer_axis(p, K) and K <= max_layers:
                # collapse everything after the layer axis; if axis 1 is small it usually
                # separates DIFFERENT quantities (CDLNet's t is [constant, sigma-slope]), so
                # keep it split rather than averaging two unrelated numbers together
                split1 = p.ndim >= 2 and p.shape[1] <= 4
                for k in range(K):
                    if len(logs) >= max_scalars:
                        break
                    if split1:
                        for j in range(p.shape[1]):
                            v = _finite(p[k, j].detach().float().mean())
                            if v is not None:
                                logs[f"{base}.layer{k:02d}.c{j}"] = v
                    else:
                        v = _finite(p[k].detach().float().mean())
                        if v is not None:
                            logs[f"{base}.layer{k:02d}"] = v

        # model-supplied derived quantities (see the module docstring)
        hook = getattr(net, "param_logs", None)
        if callable(hook):
            for k, v in (hook() or {}).items():
                v = _finite(v)
                if v is not None and len(logs) < max_scalars:
                    logs[f"{prefix}/{k}"] = v
    except Exception:
        return logs        # partial logs beat killing the run

    return logs
