"""
`forward_levels` (models/level_trace.py) records every branch of the V-cycle.

Run with `python -m tests.test_level_trace`. Small untrained MGLPDSNet on a
synthetic SENSE problem, complex state and planar state:

1. every (outer, level) has the stages scripts/level_figure.py draws: x (enters;
   None only at the cold start), x_pre (pre-smoothed), x_corr (coarse-
   corrected), x_out; the coarsest level x and x_out;
2. the trace is the forward's own arithmetic: what level l+1 enters is
   R(pre-smoothed level l), and the finest x_out + the DC kspace_pre_process
   removed is the network's output;
3. the coarse correction is what the post-smoother starts from:
   x_corr = x_pre + alpha P(coarse_delta_x);
4. tracing changes nothing: the output equals a plain forward's.
"""

from __future__ import annotations

import sys

import torch

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))


def problem(H=96, W=64, C=4):
    from operators import FFT2D, Mask, Sense
    from physics.mask import make_acc_mask
    g = torch.Generator().manual_seed(1)
    smaps = torch.randn(1, C, H, W, generator=g, dtype=torch.complex64)
    smaps = smaps / (smaps.abs().pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
    mask = make_acc_mask(shape=(H, W), accel=4, acs_lines=12, mode="uniform")
    E = Mask(mask.reshape(1, 1, H, W).to(torch.complex64)) @ FFT2D() @ Sense(smaps)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    x = (((xx ** 2 + yy ** 2) < 0.55).float() + 0.4 * ((xx ** 2 + (yy - .25) ** 2) < .1).float())
    return E, E(x.reshape(1, 1, H, W).to(torch.complex64))


def as_complex(t):
    if t is None or t.is_complex():
        return t
    h = t.shape[1] // 2
    return torch.complex(t[:, :h], t[:, h:])


def run(planar):
    from models import build_model
    from models.mg_lpds import PDObjectiveDownsample, PDVCycle, set_planar_state
    from operators.resample import prolong, restrict

    set_planar_state(planar)
    torch.manual_seed(0)
    net = build_model({"model": {"type": "MGLPDSNet", "params": dict(
        K=[2, [2, 2, 2]], M=8, C=1, P=3, s=2, degrees=1, lam0=1e-3, tau0=0.5,
        theta0=0.0, alpha0=0.3, is_complex=True, preproc="kspace",
        resize_noise=True, coarse_op="rediscretize")}}).eval()
    E, y = problem()
    with torch.no_grad():
        plain = net(y, E=E, sigma=0.01)[0]
    out, tr = net.forward_levels(y, E=E, sigma=0.01)
    out = out[0]
    tag = "planar" if planar else "complex"
    L = max(tr.levels)

    # 1. every stage
    missing = []
    for o in tr.outers:
        for l in tr.levels:
            want = ("x", "x_out") if l == L else ("x", "x_pre", "x_corr", "x_out")
            for k in want:
                if tr.get(o, l, k) is None and not (k == "x" and o == 0 and l == 0):
                    missing.append((o, l, k))
    check(f"{tag}: every V-cycle stage is recorded", not missing, str(missing))
    check(f"{tag}: the cold start is the only missing entry",
          tr.get(0, 0, "x") is None and tr.get(1, 0, "x") is not None)

    # 2. the trace is the forward's arithmetic
    kws = next(dict(julia_compat=m.julia_compat, filter=m.transfer_filter)
               for m in net.modules() if isinstance(m, PDObjectiveDownsample))
    o = tr.outers[-1]
    worst = 0.0
    for l in range(L):
        pre, nxt = as_complex(tr.get(o, l, "x_pre")), as_complex(tr.get(o, l + 1, "x"))
        worst = max(worst, float((restrict(pre, **kws) - nxt).norm() / nxt.norm()))
    check(f"{tag}: level l+1 enters R(pre-smoothed level l)", worst < 1e-5, f"{worst:.1e}")
    x0 = as_complex(tr.get(o, 0, "x_out"))
    diff = out - x0
    dc_misfit = float((diff - diff.mean()).norm() / out.norm())
    check(f"{tag}: finest x_out + a constant DC is the output", dc_misfit < 1e-5,
          f"{dc_misfit:.1e}")

    # 3. x_corr = x_pre + alpha_x P(w_c - x_c)
    # every outer cycle has its own (untied) V-cycle chain: walk the last one's
    from models.level_trace import _level_map
    lvl = _level_map(net, PDVCycle)[0]
    tops = [m for m in net.modules() if isinstance(m, PDVCycle) and lvl[id(m)] == 0]
    chain, v = [], tops[o]
    while isinstance(v, PDVCycle):
        chain.append(v)
        v = v.mglayer
    worst = 0.0
    for l, v in enumerate(chain):
        pre = as_complex(tr.get(o, l, "x_pre"))
        d = as_complex(tr.get(o, l, "coarse_delta_x"))
        want = pre + as_complex(v.alpha_x(prolong(d, filter=v.transfer_filter)))
        got = as_complex(tr.get(o, l, "x_corr"))
        worst = max(worst, float((got - want).norm() / got.norm()))
    check(f"{tag}: x_corr = x_pre + alpha P(coarse_delta_x)", worst < 1e-5, f"{worst:.1e}")

    # 4. tracing is read-only
    rel = float((out - plain).norm() / plain.norm())
    check(f"{tag}: forward_levels returns the plain forward's output", rel < 1e-6, f"{rel:.1e}")


def main():
    for planar in (False, True):
        run(planar)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
