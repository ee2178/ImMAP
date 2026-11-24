"""K-space losses for reconstruction training (training/recon.py, `training.kspace_loss`).

Why: an image-domain loss against a reference made from the SAME scan rewards whatever the
reference carries, its noise included. On the lines the network is given, the reference's
noise is the input's noise; a loss that only compares k-space the network did NOT see cannot
reward passing measurement noise (correlated or not) through to the output.

    "ssdu"       SSDU (Yaman et al., MRM 2020). Each step the acquired k-space Omega is split at
                 random into input points Theta and held-out loss points Lambda. The net sees
                 y_Theta only; the loss compares Lambda F S x_hat with y_Lambda. Needs no
                 fully-sampled data. At validation / inference the net gets all of Omega.

                 The split (`ssdu_split_mode`):
                   "points"  (default, as SSDU's reference code byaman14/SSDU and Sljiva) over
                             individual 2D k-space samples: |Lambda| = round(rho |Omega|),
                             drawn without replacement with Gaussian (`ssdu_selection`,
                             default; std (N-1)/4 per axis around the k-space centre) or
                             uniform weights. A centre block of `ssdu_center_scale`/R of each
                             dimension (0.32/R, the ACS rule) always stays in Theta. Every
                             acquired line keeps part of its samples in Theta, so the input
                             keeps the test-time line pattern, only sparser.
                   "lines"   (the 2026-11-23 runs) whole phase-encode lines: `rho` of the
                             non-ACS acquired lines go to Lambda, the whole ACS run stays in
                             Theta. The training input is then an irregular line pattern the
                             net never sees at test.
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
SSDU_SPLIT_MODES = ("points", "lines")
SSDU_SELECTIONS = ("gaussian", "uniform")
GAUSS_STD_SCALE = 4.0          # Gaussian selection: std = (N - 1) / 4 per axis (reference code)


def center_block(H, W, R, scale=0.32, device=None):
    """(H, W) bool: the centred block of round(scale / R * N) samples along each axis (at least
    one), centred on the k-space centre N // 2 (where the centred FFT puts DC)."""
    out = torch.zeros(H, W, dtype=torch.bool, device=device)
    h = max(1, int(round(scale / float(R) * H)))
    w = max(1, int(round(scale / float(R) * W)))
    r0, c0 = H // 2 - h // 2, W // 2 - w // 2
    out[r0:r0 + h, c0:c0 + w] = True
    return out


def ssdu_split_points(mask, batch, rho=0.2, R=8, shape=None, selection="gaussian",
                      center_scale=0.32, generator=None):
    """(theta, lam): (B, 1, H, W) float masks, a disjoint split of the acquired 2D k-space
    points, drawn per sample. lam holds round(rho * |Omega|) points, drawn without replacement
    from the acquired points outside the centre block (center_block, which stays in theta),
    with Gaussian weights around the k-space centre or uniformly. `shape` = (H, W) of the
    k-space when the mask is stored with a singleton readout axis."""
    if selection not in SSDU_SELECTIONS:
        raise ValueError(f"ssdu_selection must be one of {SSDU_SELECTIONS}, got {selection!r}")
    m = mask.reshape(-1, *mask.shape[-2:])[0] != 0
    H, W = shape if shape is not None else m.shape
    m = m.expand(H, W).cpu()
    keep = center_block(H, W, R, center_scale) & m
    cand = (m & ~keep).flatten().nonzero().flatten()
    n = int(round(float(rho) * int(m.sum())))
    if not 0 < n < cand.numel():
        raise ValueError(f"ssdu_rho={rho} puts {n} points in the loss, but only {cand.numel()} "
                         f"acquired points lie outside the centre block")
    if selection == "gaussian":
        gy = torch.exp(-0.5 * ((torch.arange(H) - H // 2) / ((H - 1) / GAUSS_STD_SCALE)) ** 2)
        gx = torch.exp(-0.5 * ((torch.arange(W) - W // 2) / ((W - 1) / GAUSS_STD_SCALE)) ** 2)
        w = (gy[:, None] * gx[None, :]).flatten()[cand].double()
    else:
        w = torch.ones(cand.numel(), dtype=torch.float64)
    lam = torch.zeros(batch, H * W, dtype=torch.bool)
    for b in range(batch):
        # sequential draws without replacement in proportion to w: the reference code's
        # rejection sampling of rounded Gaussian draws (reject if not acquired, in the centre
        # block, or already taken) in one call
        lam[b, cand[torch.multinomial(w, n, replacement=False, generator=generator)]] = True
    lam = lam.view(batch, 1, H, W).to(mask.device)
    theta = m.to(mask.device).view(1, 1, H, W) & ~lam
    return theta.to(torch.float32), lam.to(torch.float32)


def ssdu_split(mask, batch, rho=0.2, R=8, shape=None, mode="points", selection="gaussian",
               center_scale=0.32, generator=None):
    """The SSDU split of the acquired k-space: `mode` "points" (ssdu_split_points) or "lines"
    (ssdu_split_lines). -> (theta, lam) float masks that broadcast against (B, C, H, W)."""
    if mode == "points":
        return ssdu_split_points(mask, batch, rho, R=R, shape=shape, selection=selection,
                                 center_scale=center_scale, generator=generator)
    if mode == "lines":
        return ssdu_split_lines(mask, batch, rho, generator=generator)
    raise ValueError(f"ssdu_split_mode must be one of {SSDU_SPLIT_MODES}, got {mode!r}")


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


def ssdu_split_lines(mask, batch, rho=0.4, generator=None):
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
