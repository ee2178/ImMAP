"""
`alpha_mode` (models/mg_lpds.py::ALPHA_MODES): the coarse-correction steps.

Run with `python -m tests.test_alpha_mode`. CPU, a few seconds.

1. "free" is the default and bit-identical to a net built without the key.
2. "clamp": every alpha_x / alpha_z starts inside `alpha_range` and is put
   back there by `MGLPDSNet.project()` after any step -- negative and > 1
   values alike; a custom range is honoured.
3. "fixed": the correction enters with step exactly 1 whatever the weight
   holds (also after loading a "free" checkpoint with alpha ~ 0); the weight
   gets no gradient and an optimizer step does not move it.
4. Checkpoints are interchangeable across modes (same parameters).
5. Planar state: each mode gives the same output as the complex state.
6. Refused: an unknown mode, lo > hi, clamp / fixed with widen > 1.
"""

from __future__ import annotations

import sys

import torch

from models.mg_lpds import MGLPDSNet, PDVCycle

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def _net(**kw):
    torch.manual_seed(4)
    kw.setdefault("alpha0", 0.3)
    return MGLPDSNet(K=[2, [2, 2, 2]], M=4, C=1, P=5, s=2, is_complex=True,
                     preproc="identity", lam0=0.05, transfer_filter="sinc_vertex",
                     fas_dual="gradient", fas_copies="tied", fas_parent_pi=True, **kw)


def _alphas(net):
    return [m for V in net.modules() if isinstance(V, PDVCycle)
            for m in (V.alpha_x, V.alpha_z)]


Y = torch.randn(1, 1, 32, 32, dtype=torch.complex64, generator=torch.Generator().manual_seed(1))
SIG = torch.tensor(0.015)


def _out(net):
    with torch.no_grad():
        return net(Y, sigma=SIG)[0]


def test_free():
    a, b = _net(), _net(alpha_mode="free")
    check("free is the default and bit-identical", torch.equal(_out(a), _out(b))
          and a.alpha_mode == "free")


def test_clamp():
    net = _net(alpha_mode="clamp")
    check("clamp: starts inside [0, 1]",
          all(bool(((m.weight >= 0) & (m.weight <= 1)).all()) for m in _alphas(net)))
    with torch.no_grad():
        for i, m in enumerate(_alphas(net)):
            m.weight.fill_(-0.5 if i % 2 else 5.0)
    net.project()
    vals = sorted({round(float(v), 6) for m in _alphas(net) for v in m.weight.flatten()})
    check("clamp: project() puts -0.5 -> 0 and 5 -> 1", vals == [0.0, 1.0], str(vals))
    net = _net(alpha_mode="clamp", alpha_range=[0.2, 2.0])
    with torch.no_grad():
        for m in _alphas(net):
            m.weight.fill_(5.0)
    net.project()
    check("clamp: a custom range is honoured",
          all(bool(torch.allclose(m.weight, torch.full_like(m.weight, 2.0))) for m in _alphas(net)))


def test_fixed():
    fixed = _net(alpha_mode="fixed")
    ref = _net(alpha_mode="free", alpha0=1.0)
    check("fixed == a free net at alpha = 1 (alpha0 ignored)", torch.equal(_out(fixed), _out(ref)))
    check("fixed: weights not trainable", all(not m.weight.requires_grad for m in _alphas(fixed)))
    # a "free" checkpoint whose steps collapsed to ~0 still runs at step 1
    collapsed = _net(alpha_mode="free")
    with torch.no_grad():
        for m in _alphas(collapsed):
            m.weight.fill_(1e-4)
    f2 = _net(alpha_mode="fixed")
    f2.load_state_dict(collapsed.state_dict())
    ref2 = _net(alpha_mode="free")
    ref2.load_state_dict(collapsed.state_dict())
    with torch.no_grad():
        for m in _alphas(ref2):
            m.weight.fill_(1.0)
    check("fixed: step 1 whatever a loaded checkpoint holds", torch.equal(_out(f2), _out(ref2)))
    opt = torch.optim.Adam([p for p in fixed.parameters() if p.requires_grad], lr=1e-2)
    before = [m.weight.clone() for m in _alphas(fixed)]
    out = fixed(Y, sigma=SIG)[0]
    out.abs().pow(2).mean().backward()
    opt.step()
    fixed.project()
    check("fixed: no gradient, an optimizer step leaves the weights at 1",
          all(m.weight.grad is None for m in _alphas(fixed))
          and all(torch.equal(m.weight, b) for m, b in zip(_alphas(fixed), before)))


def test_checkpoints():
    free = _net(alpha_mode="free")
    ok = True
    for mode in ("clamp", "fixed"):
        n = _net(alpha_mode=mode)
        try:
            n.load_state_dict(free.state_dict())
            free.load_state_dict(n.state_dict())
        except Exception as e:                                       # noqa: BLE001
            ok = False
            print(e)
    check("checkpoints load across modes", ok)


def test_planar():
    for mode in ("free", "clamp", "fixed"):
        c = _out(_net(alpha_mode=mode))
        prev = MGLPDSNet.PLANAR_STATE
        try:
            MGLPDSNet.PLANAR_STATE = True
            n = _net(alpha_mode=mode)
            p = _out(n)
            active = n.planar_state_active()
        finally:
            MGLPDSNet.PLANAR_STATE = prev
        gap = float((p - c).abs().max() / c.abs().max())
        check(f"planar state, {mode}: same output", active and gap < 1e-5, f"{gap:.1e}")


def test_refused():
    for name, kw in (("unknown mode", dict(alpha_mode="gated")),
                     ("lo > hi", dict(alpha_mode="clamp", alpha_range=[1.0, 0.0])),
                     ("clamp with widen 2", dict(alpha_mode="clamp", widen=2)),
                     ("fixed with widen 2", dict(alpha_mode="fixed", widen=2))):
        try:
            _net(**kw)
            check(f"refused: {name}", False)
        except ValueError:
            check(f"refused: {name}", True)


def main():
    for t in (test_free, test_clamp, test_fixed, test_checkpoints, test_planar, test_refused):
        print(f"\n== {t.__name__}")
        t()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
