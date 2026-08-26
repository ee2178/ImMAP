# -*- coding: utf-8 -*-
"""
Stacking initialisation: pretrain ONE outer V-cycle, then seed all `K_outer` of them.

Why this is cheap
-----------------
`MGLPDSNet(K=[K_outer, iters])` is `K_outer` untied V-cycles threaded in sequence. Depth costs
super-linearly: measured fwd+bwd at M=64, 64x64, iters=[4,4,6],

    K_outer     1        2        3        6
    ms      126.9    289.6    434.5   1022.4
    vs K=6  0.124x   0.283x   0.425x    1.00x

so a single V-cycle trains at ~1/8 the cost of the full stack -- better than the naive 1/6. A
100k-step pretrain of K=1 costs about 12k K=6-steps.

Why it needs care
-----------------
`_OuterStack` already deepcopies one prototype, so the outer V-cycles are ALREADY identical at
init; this only changes what the prototype IS. The catch is that a K=1 network learns a
*terminal* map -- y~ to x_hat in one shot -- and composing six terminal maps overshoots badly.
Measured init loss on a toy problem, all seeded from the same pretrained K=1:

    naive replication            2.223e-01     (75x WORSE than fresh random)
    step size x0.01 in slots 1+  1.893e-03     (1.6x better than fresh)
    fresh random                 2.970e-03

The fix is `x_new = x - tau * residual`: driving the primal STEP SIZE toward zero makes a
V-cycle near-identity, so the tail starts as a no-op refinement of the pretrained map instead of
as another terminal map. This is ReZero / Fixup applied to the outer stack.

The two `tau`s -- read this before touching anything
----------------------------------------------------
Two DIFFERENT parameters are both spelled `tau.weight`, and conflating them is the easiest
mistake here (an earlier draft of this made it):

    <layer>.tau.weight                 shape (deg+1, 1)   PRIMAL STEP SIZE
    <layer>.prox.prox.tau.weight       shape (deg+1, M)   PROX THRESHOLD, per subband

`is_step_size` / `is_threshold` below are the disambiguation, and they are what every caller
should use rather than an `endswith` test.

Do NOT zero the thresholds. They are what make the dictionary act as a prior at all -- with no
shrinkage the coefficients pass through untouched and the reconstruction goes linear. Zeroing
them on the pretrained K=1 network measured 2659x worse than leaving them alone, i.e. it throws
away essentially everything the pretrain bought. The dictionary and its thresholds are jointly
calibrated; transfer both.
"""

from __future__ import annotations

import torch


def is_step_size(name):
    """The LPDS primal step size in `x_new = x - tau * residual`."""
    return name.endswith("tau.weight") and ".prox." not in name


def is_threshold(name):
    """The per-subband prox threshold. Transfer it; never scale it to zero."""
    return name.endswith("tau.weight") and ".prox." in name


def outer_slot(name, stack_attr="net"):
    """The outer V-cycle index a parameter belongs to, or None if outside the stack."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == stack_attr and parts[1] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def stack_state_dict(src_state, dst_state, K_outer, step_scale=0.01,
                     thresh_scale=1.0, slot=0, stack_attr="net", strict=True):
    """Replicate outer slot `slot` of `src_state` into all `K_outer` slots of `dst_state`.

    Returns a NEW state dict; neither input is mutated.

    Parameters
    ----------
    step_scale    multiplies the primal step size in slots OTHER than the first. 0.01 is the
                  recommended value: it makes the tail near-identity while leaving the gradient
                  path alive. Exactly 0.0 also reproduces the pretrained map, but measured only
                  70 of 610 tail parameters receiving gradients (against 606 at 0.01) -- the
                  tail starts dead and has to wait for the step size to crawl off zero. Pass
                  1.0 for naive replication, which is the arm this exists to be compared with.
    thresh_scale  multiplies the prox thresholds in the tail. Leave at 1.0. Exposed only so an
                  ablation can demonstrate why (see the module docstring).
    slot          which outer slot of the source to use as the prototype.
    strict        raise if a source tensor has no shape-matching destination.
    """
    prefix = f"{stack_attr}.layers.{slot}."
    proto = {k[len(prefix):]: v for k, v in src_state.items() if k.startswith(prefix)}
    if not proto:
        raise ValueError(
            f"no parameters under {prefix!r} in the source checkpoint. Keys look like "
            f"{sorted(src_state)[:3]}. Was this saved from an MGLPDSNet with an outer stack?")

    out = dict(dst_state)
    copied = skipped = 0
    for j in range(K_outer):
        for k, v in proto.items():
            full = f"{stack_attr}.layers.{j}.{k}"
            if full not in out:
                if strict:
                    raise KeyError(f"{full} missing from the destination model.")
                skipped += 1
                continue
            if out[full].shape != v.shape:
                if strict:
                    raise ValueError(
                        f"{full}: source {tuple(v.shape)} vs destination "
                        f"{tuple(out[full].shape)}. The two models must agree on M / P / s / "
                        f"iters -- only K_outer may differ.")
                skipped += 1
                continue
            w = v.clone()
            if j != 0:                       # the tail, i.e. everything after the first slot
                if step_scale != 1.0 and is_step_size(full):
                    w = w * step_scale
                if thresh_scale != 1.0 and is_threshold(full):
                    w = w * thresh_scale
            out[full] = w
            copied += 1

    # anything outside the stack (read-out dictionary, etc.) transfers verbatim where it fits
    for k, v in src_state.items():
        if outer_slot(k, stack_attr) is None and k in out and out[k].shape == v.shape:
            out[k] = v.clone()
            copied += 1

    return out, {"copied": copied, "skipped": skipped, "proto_tensors": len(proto)}


def describe(state, stack_attr="net"):
    """Per-slot mean |step size| and mean |threshold| -- a one-glance check of a transfer."""
    steps, threshs = {}, {}
    for k, v in state.items():
        j = outer_slot(k, stack_attr)
        if j is None:
            continue
        if is_step_size(k):
            steps.setdefault(j, []).append(float(v.detach().abs().mean()))
        elif is_threshold(k):
            threshs.setdefault(j, []).append(float(v.detach().abs().mean()))
    fmt = lambda d: [round(sum(v) / len(v), 6) for _, v in sorted(d.items())]
    return {"step_size_per_slot": fmt(steps), "threshold_per_slot": fmt(threshs)}
