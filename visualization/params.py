# -*- coding: utf-8 -*-
"""
Logging for a network's small learnable parameters -- thresholds, step sizes, extrapolation
weights, anything whose LITERAL VALUE is interpretable.

`visualization/filters.py` renders what the DICTIONARIES look like; this renders what the
SCALARS are doing. For an unrolled net those scalars carry most of the interpretable behaviour:
LPDS's per-layer `tau` / `theta`, the V-cycle's coarse-correction `alpha`, CDLNet's soft
threshold. Watching them over training answers questions a loss curve cannot -- whether
thresholds are collapsing to zero, whether a learned step is pinned at its bound, whether later
unrolled layers are doing anything at all.

Literal values, not summaries
-----------------------------
This used to emit mean / std / min / max / grad_norm for EVERY trainable tensor, plus per-layer
means for stacked ones. On MGLPDSNet R8 that was 3648 wandb series per call, and -- because
every one of them ended in a `float()` -- 3648 separate GPU->CPU synchronisations, each one
stalling the pipeline behind it.

It also answered the wrong question. A mean over an 8281-element conv dictionary is not
interpretable; the number that matters is `tau[0]`, and a summary of a two-element tensor is
strictly worse than the two elements.

So: tensors with at most `max_elems` entries are logged ELEMENT BY ELEMENT, at their literal
values, in ONE transfer per tensor. Everything larger is skipped entirely rather than
summarised -- use `visualization/filters.py` to look at dictionaries. Gradient NORMS are still
emitted for every tensor, large ones included: one scalar apiece, and the cheapest check that a
parameter is being learned at all.

Everything is gathered and transferred in two batched `.tolist()` calls, so the cost is TWO host
synchronisations for the whole model rather than one per scalar. On MGLPDSNet R8 that is 1080
scalars (348 literal values + 732 grad norms) and 2 syncs, against 3648 scalars and 3648 syncs.

    get_param_logs(net)   generic; knows nothing about this repo.

    net.param_logs()      optional hook. A model returning {name: float} here gets it merged in
                          under the same prefix. Use it for quantities that are not parameters
                          but functions of them -- e.g. a learned step evaluated at several
                          noise levels, which no generic walker could reconstruct.
                          (models/sb_cdlnet.py and models/sb_groupcdl.py define one.)

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


def _values(p):
    """Every element of `p`, as plain floats, in ONE host transfer.

    `.tolist()` costs a single synchronisation for the whole tensor; calling `float()` per
    element would cost one each, which is the entire reason this function exists.
    """
    t = p.detach()
    if t.is_complex():
        t = t.abs()
    flat = t.float().reshape(-1).tolist()
    return [_finite(v) for v in flat]


def get_param_logs(net, prefix="params", max_elems=64, max_scalars=4000,
                   with_grad=True):
    """-> {f"{prefix}/...": float} ready to hand straight to wandb.log.

    Two kinds of series, and no value summaries anywhere:

      VALUES     the literal value of every element of every trainable tensor with at most
                 `max_elems` entries. A single-element tensor logs as `{prefix}/{name}`; a
                 larger one as `{prefix}/{name}.{i}` over its flattened index -- for LPDS's
                 `tau` of shape (2, 1) that is `.0` (the constant) and `.1` (the sigma-slope),
                 which is the split worth watching separately. Tensors above `max_elems` get no
                 value series at all; use `visualization/filters.py` to look at dictionaries.

      GRADIENTS  `{prefix}/{name}.grad_norm`, for EVERY trainable tensor including the ones too
                 large to log literally. One scalar each, and the cheapest way to answer "is
                 this actually being learned, or sitting at its initialisation?" -- which is the
                 whole reason it is on by default. It is the one summary kept, because a
                 gradient has no small literal form worth logging.

    Both are gathered into a single tensor and transferred once, so the whole call costs TWO
    host synchronisations regardless of model size -- not one per scalar, which is what made
    the previous version expensive on GPU.

    Parameters
    ----------
    max_elems    value-logging cutoff. 64 covers the per-layer scalars (LPDS's `tau` / `theta`
                 are 2 elements, the V-cycle's `alpha` is 1) without touching dictionaries.
                 Note a PER-CHANNEL threshold has one element per subband, so at small M it
                 falls under this cutoff and logs M series per layer; `max_scalars` is what
                 stops that from flooding a run. Every such tensor still gets its grad_norm
                 either way.
    max_scalars  hard cap on how many series this can create. Values are collected before
                 gradients, so if the cap binds it is the grad_norms that are lost -- raise it
                 rather than guessing from a truncated panel list.
    with_grad    log the gradient norms described above. They read `p.grad`, which is only
                 populated between backward() and zero_grad(). Every loop in training/ zero_grads
                 at the TOP of the step, so at validation time the last training step's
                 gradients are still in place. A loop that zero_grads at the END of a step would
                 log nothing here -- the key is simply absent, never wrong.
    """
    logs = {}
    try:
        val_names, val_chunks, val_sizes = [], [], []
        grad_names, grad_vals = [], []

        for name, p in net.named_parameters():
            if not p.requires_grad:
                continue
            base = f"{prefix}/{name}"
            if p.numel() <= max_elems:
                t = p.detach()
                t = t.abs() if t.is_complex() else t
                val_chunks.append(t.float().reshape(-1))
                val_names.append(base)
                val_sizes.append(t.numel())
            if with_grad and p.grad is not None:
                g = p.grad.detach()
                grad_vals.append((g.abs() if g.is_complex() else g).float().norm())
                grad_names.append(f"{base}.grad_norm")

        # ONE transfer for every value, and one for every gradient norm.
        if val_chunks:
            flat = torch.cat(val_chunks).tolist()
            at = 0
            for base, n in zip(val_names, val_sizes):
                for i in range(n):
                    v = _finite(flat[at + i])
                    if v is None or len(logs) >= max_scalars:
                        continue
                    logs[base if n == 1 else f"{base}.{i}"] = v
                at += n

        if grad_vals:
            for nm, gv in zip(grad_names, torch.stack(grad_vals).tolist()):
                gv = _finite(gv)
                if gv is not None and len(logs) < max_scalars:
                    logs[nm] = gv

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
