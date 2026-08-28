"""
True GroupLPDS: a primal-dual sweep whose prox is nonlocal.

Run with `python -m tests.test_group_lpds`.

The distinction this file pins down
-----------------------------------
`MGLPDS` / `MGGroupLPDS` USED to mean `MGCDLNet(dual=True[, W>1, Mh])`: a LISTA
layer, ONE variable, no `tau`/`theta`, whose only LPDS-like feature was the
Fenchel prox. That imitation is gone -- `dual=True` now always means the real
thing, and `MGCDLNet` refuses the flag outright.

`MGGroupLPDS` means `MGLPDSNet(window>1, Mh)`: the real primal-dual sweep,

    x+ = x - tau (E^H E x - y~ + B z - pi_x)
    xb = x+ + theta (x+ - x)                     <- the extrapolation
    z+ = prox_{g*}(z + A xb - pi_z)              <- nonlocal, at every level

with the group threshold in the prox slot, so nonlocal self-similarity is
applied on every grid rather than once at full resolution.
"""

import torch

from models import build_model
from models.mg_lpds import MGLPDSNet
from models.prox import FenchelProx, GroupThreshold, SoftThreshold

PASS, FAIL, SKIPPED = [], [], []

BASE = dict(M=16, C=1, P=3, s=2, widen=1, degrees=1, lam0=1e-3, tau0=0.5,
            theta0=0.3, alpha0=1.0, is_complex=True, preproc="identity",
            resize_noise=True)
GROUP = dict(window=5, Mh=8, dK=2, nheads=1, sim_fun="distance",
             attn_backend="gather")


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def net(**over):
    torch.manual_seed(0)
    return MGLPDSNet(K=[2, [2, 2, 2]], **dict(BASE, **over))


# ---------------------------------------------------------------------------
def test_group_prox_is_installed():
    print("\n[the prox slot really holds a group threshold]")
    plain, grouped = net(), net(**GROUP)

    def proxes(m):
        return [q for q in m.modules() if isinstance(q, FenchelProx)]

    p_plain = proxes(plain)
    p_group = proxes(grouped)
    check("both nets use the dual (Fenchel) prox", len(p_plain) > 0 and len(p_group) > 0,
          f"{len(p_plain)} / {len(p_group)} FenchelProx")
    check("plain wraps SoftThreshold",
          all(isinstance(q.prox, SoftThreshold) for q in p_plain))
    check("group wraps GroupThreshold",
          all(isinstance(q.prox, GroupThreshold) for q in p_group))

    # every level, not just the finest -- that is the point of doing this in a
    # V-cycle rather than at full resolution once
    gts = [q for q in grouped.modules() if isinstance(q, GroupThreshold)]
    widths = {q.Walpha.weight.shape[0] if q.grouped else None for q in gts}
    check("a group prox exists on every level", len(gts) >= 3,
          f"{len(gts)} GroupThresholds, Mh values {sorted(w for w in widths if w)}")


def test_extrapolation_is_live():
    print("\n[the extrapolation step is present and load-bearing]")
    from models.lpds import LPDSLayer
    layers = [m for m in net(**GROUP).modules() if isinstance(m, LPDSLayer)]
    check("layers are LPDSLayer, not LISTALayer", len(layers) > 0, f"{len(layers)}")
    check("each carries tau AND theta",
          all(hasattr(m, "tau") and hasattr(m, "theta") for m in layers))

    # theta must be WIRED IN, which is a different claim from "theta matters a
    # lot at initialisation". Drive it to its extremes rather than testing the
    # size of its effect at theta0: the magnitude at init depends on tau, on the
    # residual scale and on the grid, none of which this file is about.
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 0.3
    sig = torch.full((1, 1, 1, 1), 0.02)
    m = net(**GROUP)
    thetas = [q.theta for q in m.modules() if isinstance(q, LPDSLayer)]

    def run_with(value):
        with torch.no_grad():
            for t in thetas:
                t.weight.zero_()
                t.weight[0].fill_(value)          # constant term of the polynomial
            return m(y, E=None, sigma=sig)[0].clone()

    off, full = run_with(0.0), run_with(1.0)
    rel = float((full - off).abs().norm() / off.abs().norm())
    # fp32 noise on a tensor this size is ~1e-7; anything above that is signal
    check("theta is wired into the sweep", rel > 1e-5,
          f"theta 0 -> 1 moves the output by {rel:.3e}")

    same = run_with(0.0)
    noise = float((same - off).abs().norm() / off.abs().norm())
    check("the comparison is deterministic", noise < 1e-9,
          f"re-running at theta=0 differs by {noise:.1e}")


def test_state_is_a_pair():
    print("\n[the state is (x, z), not a lone code]")
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 0.3
    sig = torch.full((1, 1, 1, 1), 0.02)
    m = net(**GROUP)
    with torch.no_grad():
        x_hat, state = m(y, E=None, sigma=sig)
    check("forward returns (x_hat, state)", isinstance(state, tuple) and len(state) == 2)
    x, z = state
    check("x is on the image grid", x.shape[-2:] == y.shape[-2:], str(tuple(x.shape)))
    check("z is on the strided latent grid",
          z.shape[-1] == y.shape[-1] // BASE["s"] and z.shape[1] == BASE["M"],
          str(tuple(z.shape)))
    check("output is finite", bool(torch.isfinite(x_hat.abs()).all()))


def test_trains():
    print("\n[gradients flow through the nonlocal prox]")
    y = torch.randn(1, 1, 32, 32, dtype=torch.complex64) * 0.3
    gt = torch.randn_like(y)
    sig = torch.full((1, 1, 1, 1), 0.02)
    m = net(**GROUP)
    out, _ = m(y, E=None, sigma=sig)
    (out.abs() - gt.abs()).pow(2).mean().backward()

    gts = [q for q in m.modules() if isinstance(q, GroupThreshold)]
    attn = [p for q in gts for n, p in q.named_parameters()
            if n.startswith(("Walpha", "Wbeta", "Wtheta", "Wphi"))]
    live = sum(1 for p in attn if p.grad is not None and p.grad.abs().max() > 0)
    check("attention weights receive gradient", live > 0, f"{live}/{len(attn)} tensors")
    check("all gradients finite",
          all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    m.project()
    check("project() survives the group prox",
          all(torch.isfinite(p).all() for p in m.parameters()))


def test_widening_transfers_attention():
    print("\n[coarse levels inherit the fine attention weights]")
    m = net(widen=2, **GROUP)
    gts = [q for q in m.modules() if isinstance(q, GroupThreshold) and q.grouped]
    mh = sorted({q.Walpha.weight.shape[0] for q in gts})
    check("Mh widens down the hierarchy", len(mh) > 1, f"Mh values {mh}")
    check("all attention weights are finite after preload",
          all(torch.isfinite(q.Walpha.weight).all() for q in gts))


def test_flex_backend_matrix():
    """Which (attn_backend, sim_fun) a COMPLEX model may use.

    FlexAttention's score_mod sees one bilinear form, Re<q,k>. The phase-
    invariant similarities need |<q,k>|, which also needs Im<q,k> -- so on
    complex features they are unreachable from flex, and `models/prox.py`
    raises rather than silently attending on the wrong similarity.

    Only the REJECTIONS are asserted: they are pure-python guards and hold on
    any device. Whether the accepted combinations then run is a GPU question,
    reported by scripts/profile_mg.py, not decidable here.
    """
    print("\n[flex backend: what a complex model may use]")
    from models.prox import GroupThreshold

    def build(backend, sim, **over):
        return GroupThreshold(8, Mh=4, window=5, tau0=1e-3, degrees=1,
                              sim_fun=sim, attn_backend=backend,
                              nheads=1, dK=1, **over)

    z_c = torch.randn(1, 8, 16, 16, dtype=torch.complex64)
    sig = torch.full((1, 1, 1, 1), 0.02)

    # phase-invariant + complex + flex must be refused, with a message that
    # names the alternatives
    for sim in ("pidistance", "pidot"):
        try:
            build("flex", sim)(z_c, sig, {})
            check(f"flex rejects complex {sim}", False)
        except ValueError as e:
            msg = str(e)
            check(f"flex rejects complex {sim}",
                  "FlexAttention cannot form" in msg
                  and "gather" in msg and "distance" in msg)
        except Exception as e:                                # noqa: BLE001
            # torch too old to import flex_attention at all -- the guard we
            # care about sits BEFORE that, so this means it did not fire
            check(f"flex rejects complex {sim}", False, f"{type(e).__name__}")

    # a similarity flex cannot fuse at all is refused at construction
    try:
        build("flex", "not_a_sim")
        check("flex rejects an unfusable sim_fun", False)
    except ValueError as e:
        check("flex rejects an unfusable sim_fun", "FlexAttention cannot fuse" in str(e))

    # triton is the complement: it exists FOR the pi- similarities, and points
    # anything else back at flex
    try:
        build("triton", "distance")
        check("triton rejects non-pi sim_fun", False)
    except ValueError as e:
        check("triton rejects non-pi sim_fun", "flex" in str(e))
    except RuntimeError as e:
        # triton not installed locally; the sim_fun check runs first, so
        # reaching RuntimeError means the sim_fun WAS accepted -- wrong
        check("triton rejects non-pi sim_fun", False, "triton missing")

    print("        complex + flex  ->  sim_fun must be 'distance' or 'dot'")
    print("        complex + phase-invariant  ->  attn_backend='triton'")


def test_registry():
    print("\n[the registry distinguishes the real net from the cheat]")
    params = dict(BASE, K=[1, [2, 2, 2]], **GROUP)
    m = build_model({"model": {"type": "MGGroupLPDS", "params": params}})
    if m is None:
        # Only happens under a harness that stubs `models.build_model` because
        # the real `models/__init__` cannot import (no flex_attention on an
        # old torch). On the cluster this section runs for real.
        print("  [fixture] build_model is stubbed here -- registry checks SKIPPED")
        SKIPPED.append("registry (build_model stubbed)")
        return
    check("MGGroupLPDS builds an MGLPDSNet", isinstance(m, MGLPDSNet))
    check("...with a group prox",
          any(isinstance(q, GroupThreshold) for q in m.modules()))
    check("MGLPDSNet has compile_flex (train.py calls it)", hasattr(m, "compile_flex"))

    # asking for the group net without the group args must fail loudly
    for bad, why in ((dict(BASE, K=[1, [2, 2, 2]]), "no window/Mh"),
                     (dict(BASE, K=[1, [2, 2, 2]], window=5), "no Mh")):
        try:
            build_model({"model": {"type": "MGGroupLPDS", "params": bad}})
            check(f"MGGroupLPDS rejects {why}", False)
        except ValueError:
            check(f"MGGroupLPDS rejects {why}", True)

    # and the plain name must refuse a group config rather than silently build one
    try:
        build_model({"model": {"type": "MGLPDSNet",
                               "params": dict(BASE, K=[1, [2, 2, 2]], **GROUP)}})
        check("MGLPDSNet rejects window > 1", False)
    except ValueError:
        check("MGLPDSNet rejects window > 1", True)


if __name__ == "__main__":
    test_group_prox_is_installed()
    test_extrapolation_is_live()
    test_state_is_a_pair()
    test_trains()
    test_widening_transfers_attention()
    test_flex_backend_matrix()
    test_registry()
    tail = f", {len(SKIPPED)} section(s) SKIPPED: {', '.join(SKIPPED)}" if SKIPPED else ""
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed{tail}")
    for f in FAIL:
        print(f"  FAILED: {f}")
    raise SystemExit(1 if FAIL else 0)
