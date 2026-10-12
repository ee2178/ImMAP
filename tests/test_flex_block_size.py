"""
`flex_block_size`: how much of FlexAttention's score work the window uses.

Run with `python -m tests.test_flex_block_size`.

FlexAttention is block-sparse. It computes the full B x B score tile for every
(query block, key block) pair holding at least one allowed pair and masks the
rest. The group prox's window is 2-D and the sequence is row-major
(`s = r*W + c`), so one query block reaches `win` separate row-bands of keys:
at the default B=128 most of every tile is masked.

    knee 320x184 latent, window 15 (225 keys per query)
        B=128   ~3000 scores per query     7% used
        B= 32   ~1070                     21%
        B= 16   ~ 600                     38%

The block size changes NO output -- the mask is the same -- so it is a pure
speed knob. What is checked here, on CPU:

1. the tile counter below agrees with torch's own `BlockMask`, so the table is
   what the kernel is actually scheduled to compute;
2. the group prox gives the same output at every block size, and the same as
   the gather backend (the reference), i.e. the knob is exact and the
   `kernel_options` plumbing in models/circulant_flex.py is accepted;
3. `_kernel_options` leaves the default (128) call untouched and, below it,
   hands the compiled kernel tiles that divide the block -- without which
   torch refuses to compile ("Q and KV block size must be divisible by
   BLOCK_M and BLOCK_N") on cards whose default tile is larger.

NOT checked here, because it needs a GPU: that the compiled kernel is actually
FASTER at a smaller block. Fewer scores is not automatically less time -- small
tiles cost more launches and worse occupancy. torch/time_mggroup.sbatch
measures it; do not change the generator's default on this file's say-so.
"""

from __future__ import annotations

import numpy as np
import torch

from models.circulant_flex import _kernel_options, build_block_mask
from models.prox import GroupThreshold

FAIL = []
WIN = 15
# Attention applies per level per forward at K=[6,[4,4,6]], MEASURED on
# mggroup169v6 (hooks on GroupThreshold.gamma_of): 36 at every level, 108 in
# all. Not 24/24/36 -- each non-coarsest level's coarse-correction step (dF)
# runs the prox twice more per V-cycle. Only 48 of the 108 build an adjacency.
LAYERS = (36, 36, 36)


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def tiles(H, W, B, win=WIN):
    """Non-empty (query block, key block) tiles for an H x W latent.

    Same relation as `circular_window_mask_mod`: circular Chebyshev distance
    <= (win - 1) // 2 on both axes, sequence index `r * W + c`.
    """
    S, p = H * W, (win - 1) // 2
    B = min(B, S)
    d = np.arange(-p, p + 1)
    total = 0
    for qb in range(-(-S // B)):
        s = np.arange(qb * B, min((qb + 1) * B, S))
        r, c = s // W, s % W
        k = ((r[:, None] + d[None, :]) % H)[:, :, None] * W \
            + ((c[:, None] + d[None, :]) % W)[:, None, :]
        total += np.unique(k // B).size
    return total, B


def test_counter_matches_torch():
    for H, W in ((80, 46), (80, 40), (48, 47)):
        for B in (128, 32, 16):
            bm = build_block_mask(H, W, WIN, torch.device("cpu"), circular=True,
                                  BLOCK_SIZE=B, compile=False)
            n = int(bm.kv_num_blocks.sum())
            if bm.full_kv_num_blocks is not None:
                n += int(bm.full_kv_num_blocks.sum())
            check(f"{H}x{W} B={B}: tile count == torch BlockMask",
                  n == tiles(H, W, B)[0], f"{n} tiles")


def test_block_size_is_exact():
    torch.manual_seed(0)
    M, Mh, H, W, win = 12, 6, 12, 10, 5
    z = torch.randn(1, M, H, W, dtype=torch.complex64)

    def run(backend, B=128):
        torch.manual_seed(1)
        g = GroupThreshold(M, Mh=Mh, window=win, tau0=0.05, sim_fun="distance",
                           attn_backend=backend, flex_block_size=B)
        with torch.no_grad():
            return g(z, 0.01, {})[0]

    ref = run("gather")
    base = run("flex", 128)
    check("flex (B=128) == gather reference",
          float((base - ref).abs().max() / ref.abs().max()) < 1e-5)
    for B in (64, 32, 16):
        out = run("flex", B)
        err = float((out - base).abs().max() / base.abs().max())
        check(f"flex B={B} == flex B=128 (the knob changes no output)",
              err < 1e-6, f"rel {err:.1e}")


def test_kernel_options():
    def bm(B):
        return build_block_mask(40, 40, WIN, torch.device("cpu"), circular=True,
                                BLOCK_SIZE=B, compile=False)
    check("B=128: no kernel_options (the existing call, unchanged)",
          _kernel_options(bm(128)) is None)
    for B in (64, 32, 16):
        ko = _kernel_options(bm(B))
        check(f"B={B}: tiles divide the sparse block",
              ko is not None and B % ko["BLOCK_M"] == 0 and B % ko["BLOCK_N"] == 0,
              str(ko))
    check("no BlockMask: no kernel_options", _kernel_options(None) is None)


def report():
    grids = {"brain 640x320": [(320, 160), (160, 80), (80, 40)],
             "knee  640x368": [(320, 184), (160, 92), (80, 46)],
             "knee  640x372": [(320, 188), (160, 94), (80, 47)]}
    print(f"\n  scores computed per forward (window {WIN}, prox calls {LAYERS}); "
          f"informational")
    print(f"  {'grid':<16}{'B=128':>10}{'B=64':>10}{'B=32':>10}{'B=16':>10}"
          f"{'needed':>10}{'used @128':>11}{'fine-level share':>18}")
    for name, lv in grids.items():
        row, fine = [], None
        for B in (128, 64, 32, 16):
            per = [n * tiles(H, W, B)[0] * tiles(H, W, B)[1] ** 2
                   for n, (H, W) in zip(LAYERS, lv)]
            row.append(sum(per))
            if B == 128:
                fine = per[0] / sum(per)
        need = sum(n * H * W * WIN * WIN for n, (H, W) in zip(LAYERS, lv))
        print(f"  {name:<16}" + "".join(f"{v / 1e9:>9.2f}G" for v in row)
              + f"{need / 1e9:>9.2f}G{need / row[0]:>10.0%}{fine:>17.0%}")


def main():
    for fn in (test_counter_matches_torch, test_block_size_is_exact,
               test_kernel_options):
        print(f"\n--- {fn.__name__}")
        fn()
    report()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
