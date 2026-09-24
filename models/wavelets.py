"""
Dual-tree complex wavelet (DT-CWT) initialisation for strided conv cascades.

Builds the weights that make a 3-level cascade of stride-2 convolutions compute
Kingsbury's 2D DT-CWT, with every band carried down to the deepest level so the
whole transform is ONE analysis operator (`models/wavelet_lpds.py`).

Layout
------
There are four trees (lineages), indexed t = 2 pH + pW by their level-1
sampling parity (pH, pW) -- aa, ab, ba, bb.  Each tree is one conv group and
never sees another tree's channels.  Within a group:

    level 1   1 -> 4     [LL, B1, B2, B3] of the image at parity (pH, pW)
    level l   n -> 4n    channel 0 (LL) is split into [LL, B1, B2, B3];
                         every other channel is carried by a one-hot stride-2
                         kernel, i.e. pixel-unshuffle (4 phases, phase-minor)

so the group widths are 4 -> 16 -> 64, the net's 16 -> 64 -> 256, and channel
0 of every group is the coarse LL^3.  Bands: B1 = lo(H) x hi(W),
B2 = hi(H) x lo(W), B3 = hi(H) x hi(W).

Filters (from the `dtcwt` package, hard-coded)
----------------------------------------------
Level 1 is LeGall 5/3, the same filter for every tree, displaced by the tree's
parity -- exactly what `dtcwt` does (an undecimated level 1 whose four
polyphase components ARE the trees).  Levels 2-3 use the `qshift_06` q-shift
filters: lineage 0 of an axis uses tree-a filters and lineage 1 tree-b filters
(tree b = time reverse of tree a), both at the same tap placement.

`qshift_06` has 6 nonzero taps spread over 8 samples, so it does NOT fit a 7x7
kernel.  With P = 7 the one +-0.0352 end tap per filter is dropped (energy
0.9988, worst even-shift leakage 0.027): the q-shift bank is then only nearly
orthogonal.  P >= 11 keeps the filters exact; `tests/test_wavelet_lpds.py`
uses that to check the cascade against `dtcwt` itself.

Recombination Q
---------------
The oriented complex channels are a fixed unitary across the four trees,
applied per channel at depth 3 (`dtcwt_Q`).  Rows 0 and 3 are `dtcwt`'s two
analytic outputs of each band (its `q2c`, up to 1/sqrt(2)); rows 1 and 2 are
their anti-analytic partners, which a complex image needs because its spectrum
is not conjugate-symmetric.  A q-shift HIGHPASS lands on the opposite parity to
its input (`coldfilt` interleaving), so for a band born at level >= 2 the tree
labels are flipped along every axis on which it is highpass.  LL^3 is left in
the tree basis.
"""

from __future__ import annotations

import numpy as np
import torch

# LeGall 5/3 analysis filters (`dtcwt.coeffs.biort('legall')`: h0o, h1o).
LEGALL_H0 = np.array([-0.125, 0.25, 0.75, 0.25, -0.125])
LEGALL_H1 = np.array([-0.25, 0.5, -0.25])

# qshift_06 tree-a analysis filters (`dtcwt.coeffs.qshift('qshift_06')`).
QSHIFT06_H0A = np.array([0.0351638366, 0.0, -0.0883294245, 0.2338903206,
                         0.7602723691, 0.5875182977, 0.0, -0.1143018371,
                         0.0, 0.0])
QSHIFT06_H1A = np.array([0.0, 0.0, -0.1143018371, 0.0, 0.5875182977,
                         -0.7602723691, 0.2338903206, 0.0883294245, 0.0,
                         -0.0351638366])

# Is band b highpass along (H, W)?  Band 0 is LL.
HIGHPASS = {0: (False, False), 1: (False, True), 2: (True, False), 3: (True, True)}
NTREES = 4


def _tree_parity(t):
    return t // 2, t % 2


def _place(h, center, P):
    """A length-P kernel holding the odd-length centred filter `h` at `center`."""
    k, r = np.zeros(P), len(h) // 2
    k[center - r:center + r + 1] = h
    return k


def _level1_taps(hi, parity, P):
    return _place(LEGALL_H1 if hi else LEGALL_H0, P // 2 + parity, P)


def _qshift_taps(hi, lineage, P):
    """Cross-correlation taps g[d], d = -P//2..P//2, of a stride-2 q-shift step.

    In lineage coordinates `dtcwt.coldfilt` computes out[j] = sum_d g[d]
    x[2j + d] with g[d] = f[d + 4], f the 10-tap tree filter (tree a for
    lineage 0, tree b = reverse(tree a) for lineage 1).  Taps outside the
    window are dropped.
    """
    f = QSHIFT06_H1A if hi else QSHIFT06_H0A
    if lineage == 1:
        f = f[::-1]
    k = np.zeros(P)
    for i in range(P):
        j = i - P // 2 + 4
        if 0 <= j < len(f):
            k[i] = f[j]
    return k


def _band(taps, t, b, P):
    pH, pW = _tree_parity(t)
    hH, hW = HIGHPASS[b]
    return np.outer(taps(hH, pH, P), taps(hW, pW, P))


def dtcwt_weights(P=7, levels=3):
    """Analysis weights of the carried 2D DT-CWT cascade, one per level.

    Returns `(weights, tags)`.  `weights[l]` has the shape of a stride-2
    `Conv2d` with `groups = 1` at level 1 and `groups = 4` after:
    (16, 1, P, P), (64, 4, P, P), (256, 16, P, P) for three levels.  `tags[c]`
    is the `(level, band)` of in-group channel c at the deepest level.
    """
    if P < 7 or P % 2 == 0:
        raise ValueError("P must be odd and >= 7; got %r" % (P,))
    W1 = np.zeros((4 * NTREES, 1, P, P))
    for t in range(NTREES):
        for b in range(4):
            W1[4 * t + b, 0] = _band(_level1_taps, t, b, P)
    weights, tags = [W1], [(1, b) for b in range(4)]

    for level in range(2, levels + 1):
        n = len(tags)
        W = np.zeros((NTREES * 4 * n, n, P, P))
        for t in range(NTREES):
            o = t * 4 * n
            for b in range(4):                               # split LL
                W[o + b, 0] = _band(_qshift_taps, t, b, P)
            for c in range(1, n):                            # carry the rest
                for p in range(4):
                    W[o + 4 * c + p, c, P // 2 + p // 2, P // 2 + p % 2] = 1.0
        weights.append(W)
        tags = [(level, b) for b in range(4)] + \
               [tag for tag in tags[1:] for _ in range(4)]

    return [torch.from_numpy(w).float() for w in weights], tags


# Rows: analytic (0, 3) and anti-analytic (1, 2) combinations of the trees
# [aa, ab, ba, bb].  Row 0 is dtcwt's p - q, row 3 its p + q (`q2c`), each / sqrt 2.
_Q0 = 0.5 * np.array([[1, 1j, 1j, -1],
                      [1, -1j, 1j, 1],
                      [1, -1j, -1j, -1],
                      [1, 1j, -1j, 1]])


def dtcwt_Q(tags):
    """(n, 4, 4) complex unitary per in-group channel, mixing across trees.

    `out[i] = sum_t Q[c, i, t] tree_t[c]`.  Identity for LL; for a band born at
    level >= 2 the tree labels are flipped along its highpass axes.
    """
    Q = np.zeros((len(tags), NTREES, NTREES), dtype=np.complex64)
    for c, (level, b) in enumerate(tags):
        if b == 0:
            Q[c] = np.eye(NTREES)
            continue
        hH, hW = HIGHPASS[b] if level >= 2 else (False, False)
        for t in range(NTREES):
            pH, pW = _tree_parity(t)
            label = 2 * (pH ^ hH) + (pW ^ hW)
            Q[c, :, t] = _Q0[:, label]
    return torch.from_numpy(Q)
