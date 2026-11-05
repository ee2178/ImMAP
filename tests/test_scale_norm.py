"""
exp22: `scale_norm`, MGLPDSNet run on y / s and sigma / s with RAM's signal statistic s
(models/mg_lpds.py, models/cg_init.py::ram_signal_level).

Run with `python -m tests.test_scale_norm`. CPU, a few seconds.

1. `ram_signal_level` is the mean of RAM's real (B, 2, C, H, W) tensor, and `ram_gamma`
   is unchanged by the refactor.
2. Exact scale equivariance, flat (LPDS) and V-cycle: f(c y, c sigma) = c f(y, sigma)
   for the image AND the returned (x, z), per image of a batch with different c, and for
   sigma as (B,), (B, 1, 1, 1) and a float.
3. The plain net is not equivariant (the property is the option's, not an accident), and
   scale_norm=False is bit-identical to a net built without the key.
4. It is the plain net on normalized inputs: scale_norm(y, sigma) = s . plain(y/s, sigma/s).
5. A state handed in is in data units: the warm start round-trips.
6. The exp22 cells build, change only what they say, and have names of their own.
7. Level traces are in data units: `level_images` still finds a constant DC, and the
   finest level + DC is the output.
"""

from __future__ import annotations

import math
import os
import sys

import torch

from models.cg_init import ram_gamma, ram_signal_level
from tests.test_cg_init import FAIL, check, problem, rel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import make_mg_recon_configs as gen                        # noqa: E402

KW = dict(M=4, C=1, P=5, s=2, is_complex=True, preproc="kspace", lam0=0.3, degrees=1,
          alpha0=1.0)
NETS = {"flat": dict(KW, K=3), "vcycle": dict(KW, K=[2, [2, 2, 2]])}


def build(name, **extra):
    from models.mg_lpds import MGLPDSNet
    torch.manual_seed(1)
    net = MGLPDSNet(**NETS[name], **extra)
    # A nonzero noise slope, so the sigma path is exercised (w1 starts at 0).
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.endswith("tau.weight") and p.shape[0] == 2 and "prox" in n:
                p[1].fill_(2.0)
    return net.eval()


def statistic_checks():
    E, y, sig = problem(B=2)
    real = torch.stack((y.real, y.imag), dim=1)                     # RAM's layout
    ref = real.reshape(2, -1).abs().mean(1)
    check("1. ram_signal_level = mean of RAM's real tensor", rel(ram_signal_level(y), ref) < 1e-6)
    s = sig / math.sqrt(2.0)
    snr = ref / (s + 1e-4)
    check("1. ram_gamma unchanged", rel(ram_gamma(y, sig), 1.0 / (1e-4 + 1.0 / snr)) < 1e-6)


@torch.no_grad()
def equivariance_checks():
    E, y, sig = problem(B=2)
    c = torch.tensor([7.0, 0.05])
    cy = y * c.view(-1, 1, 1, 1)
    for name in NETS:
        net = build(name, scale_norm=True)
        for label, s0, s1 in (("(B,)", sig, sig * c),
                              ("(B,1,1,1)", sig.view(-1, 1, 1, 1), (sig * c).view(-1, 1, 1, 1))):
            a, (xa, za) = net(y, E=E, sigma=s0)
            b, (xb, zb) = net(cy, E=E, sigma=s1)
            cb = c.view(-1, 1, 1, 1)
            err = max(rel(b, a * cb), rel(xb, xa * cb), rel(zb, za * cb))
            check(f"2. {name}: f(c y, c sigma) = c f(y, sigma), sigma {label}", err < 1e-4, f"{err:.1e}")
        # A float sigma and a uniform c: the statistic still rescales.
        a, _ = net(y, E=E, sigma=0.015)
        b, _ = net(y * 3.0, E=E, sigma=0.045)
        check(f"2. {name}: float sigma", rel(b, 3.0 * a) < 1e-4, f"{rel(b, 3.0 * a):.1e}")

        plain = build(name)
        a, _ = plain(y, E=E, sigma=sig)
        b, _ = plain(cy, E=E, sigma=sig * c)
        check(f"3. {name}: the plain net is NOT equivariant", rel(b, a * c.view(-1, 1, 1, 1)) > 1e-3,
              f"{rel(b, a * c.view(-1, 1, 1, 1)):.1e}")
        off = build(name, scale_norm=False)
        check(f"3. {name}: scale_norm=False is the plain net",
              torch.equal(off(y, E=E, sigma=sig)[0], a))

        s = ram_signal_level(y)
        ref, _ = plain(y / s.view(-1, 1, 1, 1), E=E, sigma=sig / s)
        out, _ = net(y, E=E, sigma=sig)
        check(f"4. {name}: = s . plain(y / s, sigma / s)", rel(out, ref * s.view(-1, 1, 1, 1)) < 1e-5)

        _, (x0, z0) = net(y, E=E, sigma=sig)
        w1, _ = net(y, E=E, sigma=sig, state=(x0, z0))
        w2, _ = net(cy, E=E, sigma=sig * c, state=(x0 * c.view(-1, 1, 1, 1), z0 * c.view(-1, 1, 1, 1)))
        check(f"5. {name}: a warm start is in data units", rel(w2, w1 * c.view(-1, 1, 1, 1)) < 1e-4)


@torch.no_grad()
def trace_checks():
    E, y, sig = problem(B=1, H=48, W=48, sigma=(0.015,))
    net = build("vcycle", scale_norm=True)
    out, imgs = net.forward_level_images(y, E=E, sigma=sig)
    ok = imgs is not None and float((imgs[0] - out[0]).abs().max() / out[0].abs().max()) < 1e-5
    check("7. level_images under scale_norm: finest level + DC = output", ok)


def generator_checks():
    from models import build_model
    names = set()
    for cell, (src, extra) in gen.EXP22.items():
        a, b = gen.MODELS[cell], gen.MODELS[src]
        diff = {k for k in set(a["params"]) | set(b["params"]) if a["params"].get(k) != b["params"].get(k)}
        same_spec = {k: v for k, v in a.items() if k != "params"} == {k: v for k, v in b.items() if k != "params"}
        check(f"6. {cell} = {src} + {sorted(extra)}", diff == set(extra) and same_spec, str(diff))
        net = build_model({"model": {"type": a["type"], "params": a["params"]}})
        check(f"6. {cell} builds with scale_norm on", net.scale_norm is True)
        check(f"6. {cell} is OPT_IN", cell in gen.OPT_IN)
        names.add(gen._display_name(a["type"], a["params"]) + "_sn")
    taken = {gen._display_name(gen.MODELS[c]["type"], gen.MODELS[c]["params"])
             for c in gen.EXP18_CELLS + gen.EXP20_CELLS}
    check("6. exp22 run names are distinct", len(names) == len(gen.EXP22) and not names & taken,
          str(sorted(names)))
    from models.mg_lpds import MGLPDSNet
    try:
        MGLPDSNet(**NETS["flat"], scale_norm="yes")
        check("6. a non-bool scale_norm is refused", False)
    except ValueError:
        check("6. a non-bool scale_norm is refused", True)


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    statistic_checks()
    equivariance_checks()
    trace_checks()
    generator_checks()
    print("\nALL OK" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
