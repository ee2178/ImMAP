"""K-space losses for reconstruction training (training/recon.py, `training.kspace_loss`).

Why: an image-domain loss against a reference made from the SAME scan rewards whatever the
reference carries, its noise included. On the lines the network is given, the reference's
noise is the input's noise; a loss that only compares k-space the network did NOT see cannot
reward passing measurement noise (correlated or not) through to the output.

    "ssdu"       SSDU (Yaman et al., MRM 2020). Each step the acquired lines Omega are split at
                 random into input lines Theta and held-out loss lines Lambda (a fraction
                 `ssdu_rho` of the non-ACS acquired lines; the ACS block always stays in
                 Theta). The net sees y_Theta only; the loss compares Lambda F S x_hat with
                 y_Lambda. Needs no fully-sampled data. At validation / inference the net gets
                 all of Omega, as in SSDU.
    "unsampled"  Supervised, but in k-space and only on what was NOT measured: the input is
                 y_Omega as usual, the loss compares (1 - Omega) F S x_hat with the measured
                 k-space there. Uses the fully-sampled scan, still never scores a line the net
                 was given.

Both use SSDU's normalized l1 + l2 over the loss lines, per sample, on multicoil k-space
through the operator's coil maps:

    ||m (k_hat - k)||_2 / ||m k||_2  +  ||m (k_hat - k)||_1 / ||m k||_1
"""

from __future__ import annotations

import torch

KSPACE_LOSSES = ("ssdu", "unsampled")


def sampled_lines(mask):
    """(W,) bool: the phase-encode lines (columns) a line mask samples. Refuses a mask that is
    not constant along the readout (dim -2) -- the split below moves whole lines."""
    m = mask.reshape(-1, *mask.shape[-2:])[0] != 0
    if not bool((m == m[:1]).all()):
        raise ValueError("SSDU split needs a line mask (constant along the readout, dim -2)")
    return m[0]


def center_run(lines):
    """(W,) bool: the contiguous run of sampled lines through the centre column (the ACS)."""
    W = lines.shape[0]
    out = torch.zeros_like(lines)
    c = W // 2
    if not bool(lines[c]):
        return out
    lo, hi = c, c
    while lo > 0 and bool(lines[lo - 1]):
        lo -= 1
    while hi < W - 1 and bool(lines[hi + 1]):
        hi += 1
    out[lo:hi + 1] = True
    return out


def ssdu_split(mask, batch, rho=0.4, generator=None):
    """(theta, lam): (B, 1, 1, W) float masks, a disjoint split of the sampled lines, per sample.
    The ACS run stays in theta; `rho` of the other sampled lines go to lam."""
    lines = sampled_lines(mask)
    acs = center_run(lines)
    cand = (lines & ~acs).nonzero().flatten()
    n = int(round(float(rho) * cand.numel()))
    if not 0 < n < cand.numel():
        raise ValueError(f"ssdu_rho={rho} puts {n} of {cand.numel()} non-ACS lines in the loss")
    W = lines.shape[0]
    lam = torch.zeros(batch, W, dtype=torch.bool, device=mask.device)
    for b in range(batch):
        pick = torch.randperm(cand.numel(), generator=generator, device="cpu")[:n]
        lam[b, cand[pick.to(cand.device)]] = True
    theta = lines.unsqueeze(0) & ~lam
    f = lambda t: t.to(torch.float32).view(batch, 1, 1, W)     # noqa: E731
    return f(theta), f(lam)


def kspace_nl1nl2(pred, target, m, eps=1e-12):
    """SSDU's normalized l1 + l2 over the masked k-space, averaged over the batch."""
    d = (m * (pred - target)).flatten(1)
    t = (m * target).flatten(1)
    l2 = d.abs().pow(2).sum(1).sqrt() / (t.abs().pow(2).sum(1).sqrt() + eps)
    l1 = d.abs().sum(1) / (t.abs().sum(1) + eps)
    return (l1 + l2).mean()
