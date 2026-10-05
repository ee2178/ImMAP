"""
`MGLPDSNet.project()`: every constrained module projected ONCE, same result.

Run with `python -m tests.test_mg_project`.

`project()` used to walk `self.modules()` and call `project_` on everything that
had one -- which called each owner (an `LPDSLayer`, a prox) AND each child the
owner had just projected. It now stops at a module that defines `project_`.
That is only safe if every such module reaches its whole subtree, so this pins:

1. EXACT agreement with the old walk, on models whose parameters were pushed
   well outside every constraint first (so each clamp and each unit-ball
   projection actually fires), across the variants that add constrained
   modules: flat stack, V-cycle, widened V-cycle, learned transfers, group prox;
2. the call counts: one `project_` per `Polynomial`, one unit-ball projection
   per conv.
"""

import copy

import torch

import models.lpds as lpds_mod
import models.mg_lpds as mg_mod
import operators.projections as proj
from models.components import _GaussConvNd
from models.mg_lpds import MGLPDSNet
from models.prox import Polynomial

FAIL = []
BASE = dict(M=8, C=1, P=3, s=2, lam0=1e-3, tau0=0.5, theta0=0.5, alpha0=1.0,
            is_complex=True, preproc="kspace")
VARIANTS = {
    "flat K=3": dict(K=3),
    "V-cycle [2,[2,2,2]]": dict(K=[2, [2, 2, 2]]),
    "V-cycle, widen=2": dict(K=[1, [2, 2, 2]], widen=2),
    "V-cycle, learned transfer": dict(K=[1, [2, 2]], learn_transfer=True),
    "V-cycle, degrees=1": dict(K=[1, [2, 2, 2]], degrees=1),
    "V-cycle, group prox": dict(K=[1, [2, 2]], window=5, Mh=4, attn_backend="gather"),
}


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def old_project(net):
    """The walk `project()` replaced, verbatim."""
    with torch.no_grad():
        for m in net.modules():
            if m is not net and hasattr(m, "project_"):
                m.project_()


def perturbed(kws, seed=0):
    torch.manual_seed(seed)
    net = MGLPDSNet(**dict(BASE, **kws))
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in net.parameters():
            # large and signed: thresholds go negative, steps leave [0, 1],
            # filters leave the unit ball
            p.add_(3.0 * torch.randn(p.shape, generator=g))
    return net


class Count:
    def __enter__(self):
        self.n = {"uball": 0, "poly": 0}
        self._u, self._p = proj.uball_project, Polynomial.project_
        outer = self

        def u(W, dim=(2, 3)):
            outer.n["uball"] += 1
            return outer._u(W, dim)

        def p(mod, lo=0.0, hi=None):
            outer.n["poly"] += 1
            return outer._p(mod, lo, hi)

        self._mods = [m for m in (lpds_mod, mg_mod) if hasattr(m, "uball_project")]
        for m in self._mods:
            m.uball_project = u
        Polynomial.project_ = p
        return self

    def __exit__(self, *exc):
        for m in self._mods:
            m.uball_project = self._u
        Polynomial.project_ = self._p


def main():
    for name, kws in VARIANTS.items():
        try:
            a = perturbed(kws)
        except Exception as e:                                    # noqa: BLE001
            print(f"[skip] {name}: could not build here ({type(e).__name__}: {e})")
            continue
        b = copy.deepcopy(a)
        before = {k: v.clone() for k, v in a.state_dict().items()}
        old_project(a)
        with Count() as c:
            b.project()
        sa, sb = a.state_dict(), b.state_dict()
        same = all(torch.equal(sa[k], sb[k]) for k in sa)
        moved = sum(not torch.equal(before[k], sa[k]) for k in sa)
        check(f"{name}: identical to the old walk", same,
              f"{moved}/{len(sa)} tensors changed by projection")
        check(f"{name}: the fixture is not vacuous (projection moved something)",
              moved > 0)
        n_poly = sum(isinstance(m, Polynomial) for m in b.modules())
        n_conv = sum(isinstance(m, _GaussConvNd) for m in b.modules())
        check(f"{name}: one clamp per Polynomial", c.n["poly"] == n_poly,
              f"{c.n['poly']} calls for {n_poly} modules")
        check(f"{name}: one unit-ball projection per conv", c.n["uball"] == n_conv,
              f"{c.n['uball']} calls for {n_conv} convs")
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
