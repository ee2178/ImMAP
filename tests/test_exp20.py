"""
exp20: `lam_level_scale` and an alpha floor with no ceiling (models/mg_lpds.py).

Run with `python -m tests.test_exp20`. CPU, a few seconds.

1. Default 1.0 is bit-identical to a net built without the key.
2. 0.5: thresholds start at lam0 / 2^l at every level, in every layer and FAS copy,
   the noise coefficient still starts at 0, and filters and steps are untouched.
3. Checkpoints load across values, and a loaded threshold is not rescaled.
4. Floor: alpha_range=[0.5, None] puts 0 back to 0.5 and leaves 5 alone, for x and z.
5. Refused: a scale <= 0.
6. The generated cells build, with names of their own and the default grid untouched.
"""

from __future__ import annotations

import os
import sys

import torch

from models.mg_lpds import MGLPDSNet, PDVCycle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import make_mg_recon_configs as gen                        # noqa: E402

FAIL = []
LAM0 = 1e-3


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _net(**kw):
    torch.manual_seed(4)
    return MGLPDSNet(K=[2, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=True, preproc="identity",
                     lam0=LAM0, degrees=1, alpha0=1.0, transfer_filter="sinc_vertex",
                     fas_dual="fixed_point", fas_copies="untied", fas_parent_pi=True, **kw)


Y = torch.randn(1, 1, 32, 32, dtype=torch.complex64, generator=torch.Generator().manual_seed(1))
SIG = torch.tensor(0.015)


def _out(net):
    with torch.no_grad():
        return net(Y, sigma=SIG)[0]


def levels(net):
    """{level: [threshold polynomials' weights]}, layers and FAS copies alike."""
    out = {}

    def walk(v, l):
        layers = list(v.lpdsA.layers) + list(v.lpdsB.layers)
        out.setdefault(l, []).extend(m.prox.prox.tau.weight for m in layers)
        out[l].append(v.dF.prox_fine.prox.tau.weight)
        out.setdefault(l + 1, []).append(v.dF.prox_coarse.prox.tau.weight)
        if isinstance(v.mglayer, PDVCycle):
            walk(v.mglayer, l + 1)
        else:
            out[l + 1].extend(m.prox.prox.tau.weight for m in v.mglayer.layers)

    for v in net.net.layers:
        walk(v, 0)
    return out


def test_default():
    check("1.0 is the default and bit-identical", torch.equal(_out(_net()), _out(_net(lam_level_scale=1.0))))


def test_scale():
    net = _net(lam_level_scale=0.5)
    lv = levels(net)
    check("three levels found", sorted(lv) == [0, 1, 2], str(sorted(lv)))
    for l, ws in lv.items():
        want = LAM0 / 2 ** l
        check(f"L{l}: every constant term is lam0 / {2 ** l}",
              all(torch.allclose(w[0], torch.full_like(w[0], want)) for w in ws), f"{len(ws)} polynomials")
        check(f"L{l}: every noise coefficient starts at 0", all(bool((w[1] == 0).all()) for w in ws))
    a, b = _net().state_dict(), net.state_dict()
    other = [k for k in a if "prox" not in k and not torch.equal(a[k], b[k])]
    check("nothing but thresholds differs", not other, str(other[:3]))
    check("same parameters", list(a) == list(b))


def test_checkpoint():
    src, dst = _net(), _net(lam_level_scale=0.5)
    dst.load_state_dict(src.state_dict())
    check("a loaded checkpoint is not rescaled", torch.equal(_out(src), _out(dst)))


def test_floor():
    net = _net(alpha_mode="clamp", alpha_range=[0.5, None])
    ms = [m for v in net.modules() if isinstance(v, PDVCycle) for m in (v.alpha_x, v.alpha_z)]
    check("starts at alpha0 = 1", all(bool((m.weight == 1).all()) for m in ms))
    with torch.no_grad():
        for i, m in enumerate(ms):
            m.weight.fill_(0.0 if i % 2 else 5.0)
    net.project()
    vals = sorted({round(float(v), 6) for m in ms for v in m.weight.flatten()})
    check("project: 0 -> 0.5, and 5 stays", vals == [0.5, 5.0], str(vals))


def test_refused():
    for bad in (0.0, -0.5):
        try:
            _net(lam_level_scale=bad)
            check(f"refused: {bad}", False)
        except ValueError:
            check(f"refused: {bad}", True)


def test_cells():
    base = gen.MODELS[gen.EXP19_BASE]["params"]
    names = set()
    for c in gen.EXP20_CELLS:
        spec = gen.MODELS[c]
        diff = {k: v for k, v in spec["params"].items() if base.get(k) != v}
        check(f"{c}: differs only as named", set(diff) <= {"lam_level_scale", "alpha_mode", "alpha_range"}, str(diff))
        names.add(gen._display_name(spec["type"], spec["params"]))
        small = dict(spec["params"], M=4, K=[1, [2, 2, 2]], P=5)
        try:
            MGLPDSNet(**small)
            check(f"{c}: builds", True)
        except Exception as e:                                       # noqa: BLE001
            check(f"{c}: builds", False, repr(e))
    check("the cells have different run names", len(names) == 3, str(names))
    check("all opt-in, so no array renumbers", all(c in gen.OPT_IN for c in gen.EXP20_CELLS))


def main():
    for t in (test_default, test_scale, test_checkpoint, test_floor, test_refused, test_cells):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
