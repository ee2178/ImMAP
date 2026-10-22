"""
scripts/level_figure.py: tracing a slice with knobs, on CACHED nets.

Run with `python -m tests.test_level_figure`. CPU, a few seconds. A tiny
fastMRI-shaped tree (tests/test_dump_eval.py's fixture) and two small
MGLPDSNet runs -- A complex state + rediscretize, B planar state + planar convs
+ Galerkin -- so the process-wide switches differ between them.

1. each net is loaded ONCE however many traces use it, and reloaded when its
   net.ckpt changes on disk;
2. knobs (coarse_op, transfer_filter, transfer_taps) take effect, and a trace
   without knobs afterwards is bit-identical to the first: nothing an earlier
   trace set survives, and the net's attributes are back to how it was built;
3. tracing B in between does not change A's result (each net runs in its own
   conv / planar mode);
4. `as_traced(live)` puts a re-used net back in the state that trace ran with.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dump_eval as de                                           # noqa: E402
import level_figure as lf                                        # noqa: E402
from tests.test_dump_eval import base_cfg, build_trees           # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_run(root, name, ks, sm, coarse_op, planar, flat=False):
    from models import build_model
    from models.components import set_complex_mode
    from models.mg_lpds import set_planar_state
    cfg = base_cfg(ks, sm)
    cfg["data"]["val"]["pad_multiple"] = 8
    cfg["model"] = {"type": "MGLPDSNet", "params": dict(
        K=4 if flat else [2, [2, 2, 2]], M=8, C=1, P=3, s=2, degrees=1, lam0=1e-3, tau0=0.5,
        theta0=0.0, alpha0=0.3, is_complex=True, preproc="kspace",
        resize_noise=True, coarse_op=coarse_op)}
    if planar:
        cfg["training"].update(planar_state=True, complex_conv="planar")
    run = os.path.join(root, name)
    os.makedirs(run)
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"))
    set_complex_mode(cfg["training"].get("complex_conv") or "gauss")
    set_planar_state(planar)
    torch.manual_seed(1 if planar else 0)
    net = build_model(cfg)
    torch.save({"model_state_dict": net.state_dict(), "step": 10},
               os.path.join(run, "net.ckpt"))
    return run


def main():
    tmp = tempfile.mkdtemp()
    ks, sm = build_trees(tmp)
    A = make_run(tmp, "mgA_R4", ks, sm, "rediscretize", planar=False)
    B = make_run(tmp, "mgB_R4", ks, sm, "galerkin", planar=True)

    loads = []
    real = de.load_run
    de.load_run = lambda *a, **k: loads.append(os.path.basename(a[0])) or real(*a, **k)
    lf.clear_cache()
    kw = dict(val_kspace_type="simulated", device="cpu")

    t0 = time.time()
    z0, m0, live0 = lf.trace_slice(A, "volA", 2, **kw)
    t_first = time.time() - t0
    t0 = time.time()
    z1, m1, _ = lf.trace_slice(A, "volA", 2, coarse_op="galerkin", transfer_filter="box", **kw)
    zt, mt, _ = lf.trace_slice(A, "volA", 2, transfer_taps=12, **kw)
    zB, mB, _ = lf.trace_slice(B, "volA", 2, **kw)
    z2, m2, live2 = lf.trace_slice(A, "volA", 2, **kw)
    t_rest = (time.time() - t0) / 4

    # 1 ------------------------------------------------------------------
    check("each net loaded once over five traces", loads == ["mgA_R4", "mgB_R4"], str(loads))
    print(f"      first trace {t_first:.2f} s, later ones {t_rest:.2f} s on average")

    # 2 ------------------------------------------------------------------
    rel = lambda a, b: float(((a - b).abs() ** 2).sum() ** .5 / (abs(b) ** 2).sum() ** .5)
    r1 = rel(torch.from_numpy(z1["recon"]), torch.from_numpy(z0["recon"]))
    rt = rel(torch.from_numpy(zt["recon"]), torch.from_numpy(z0["recon"]))
    check("knobs take effect (galerkin + box)", r1 > 1e-4 and m1["coarse_op"] == "galerkin"
          and m1["transfer_filter"] == "box", f"rel. change {r1:.1e}")
    check("transfer_taps takes effect", rt > 1e-5 and mt["transfer_taps"] == 12, f"rel. change {rt:.1e}")
    check("no knobs afterwards: bit-identical to the first trace",
          all((z2[k] == z0[k]).all() for k in z0), "")
    ent = lf._net(A, torch.device("cpu"))
    check("A's modules are back to how it was built",
          all(m.coarse_op == "rediscretize" for m, _ in ent["coarse"])
          and all(m.transfer_filter == "sinc" for m, _ in ent["filters"]))

    # 3 ------------------------------------------------------------------
    from models.components import _GaussConvNd
    from models.mg_lpds import MGLPDSNet
    check("B ran in its own modes, A back in its own",
          mB["coarse_op"] == "galerkin" and _GaussConvNd.COMPLEX_MODE == "gauss"
          and MGLPDSNet.PLANAR_STATE is False)

    # 4 ------------------------------------------------------------------
    lf.trace_slice(A, "volA", 2, transfer_filter="none", **kw)
    with lf.as_traced(live0) as model:
        state = [m.transfer_filter for m in model.modules() if hasattr(m, "transfer_filter")]
    check("as_traced restores the state a trace ran with", set(state) == {"sinc"}, str(set(state)))

    # 5: every LPDS layer, multigrid and flat -----------------------------------
    import numpy as np
    flat = make_run(tmp, "lpdsflat_R4", ks, sm, "rediscretize", planar=True, flat=True)
    for run, n_want in ((A, 12), (flat, 4)):        # K=[2,[2,2,2]]: (1+1 + 1+1 + 2) x 2
        layers, gts, lm = lf.trace_layers(run, "volA", 2, **kw)
        last = [L for L in layers if L["shape"] == layers[0]["shape"]][-1]
        zr, _, _ = lf.trace_slice(run, "volA", 2, **kw) if run == A else (None, None, None)
        check(f"{os.path.basename(run)}: one iterate per LPDS layer, last + DC = output",
              lm["n_layers"] == n_want and lm["dc_misfit"] < 1e-5
              and (zr is None or np.allclose(last["x"], zr["recon"], atol=1e-5)),
              f"{lm['n_layers']} layers, DC misfit {lm['dc_misfit']:.1e}")
    sh = set(L["shape"] for L in lf.trace_layers(A, "volA", 2, **kw)[0])
    check("multigrid layers run on every level's grid", len(sh) == 3, str(sorted(sh)))
    rows = np.tile(np.array([1.0, -1.0])[:, None], (16, 32)) + 3.0    # alternates down axis 0
    r, c = lf.nyquist_share(rows)
    check("nyquist_share: horizontal stripes are all 'rows'", r > 0.99 and c < 0.01,
          f"rows {r:.3f} cols {c:.3f}")

    # 6: the stride-2 polyphase leak (notebook section 12) ----------------------
    g = torch.Generator().manual_seed(0)
    w = torch.complex(torch.randn(4, 3, 7, 7, generator=g), torch.randn(4, 3, 7, 7, generator=g))
    lr = lf.phase_leak(w)
    check("a random 7x7 kernel leaks ~ its DC gain", all(0.3 < v < 3 for v in lr), str(lr))
    wb, wr = lf.equalize_phases(w, "both"), lf.equalize_phases(w, "rows")
    lb, lrr = lf.phase_leak(wb), lf.phase_leak(wr)
    keep = lambda a: float((a.sum((-2, -1)) - w.sum((-2, -1))).abs().max())   # noqa: E731
    check("equalize 'both': no leak on any axis, DC gains kept",
          max(lb) < 1e-5 and keep(wb) < 1e-5, f"{lb}, DC change {keep(wb):.1e}")
    check("equalize 'rows': rows + diag gone, cols untouched, DC kept",
          lrr[0] < 1e-5 and lrr[2] < 1e-5 and abs(lrr[1] - lr[1]) < 1e-5 and keep(wr) < 1e-5,
          f"{lrr} vs {lr}")
    check("equalize changes as little as it can (|dw| < |w|)",
          float((wb - w).norm()) < float(w.norm()), f"{float((wb - w).norm() / w.norm()):.2f}")

    from models.components import Conv2d, ConvTranspose2d
    for run, levels in ((A, {0, 1, 2}), (flat, {0})):
        model = lf.cached_model(run, "cpu")
        sites = lf.conv_sites(model)
        n_conv = sum(isinstance(m, (Conv2d, ConvTranspose2d)) for m in model.modules())
        check(f"{os.path.basename(run)}: conv_sites finds every conv once, on the right levels",
              len(sites) == n_conv and len({id(s['module']) for s in sites}) == n_conv
              and {s["level"] for s in sites} == levels,
              f"{len(sites)} sites / {n_conv} convs, levels {sorted({s['level'] for s in sites})}")

    model = lf.cached_model(A, "cpu")
    before = lf.leak_table(model)
    with lf.equalized_convs(model) as n:
        z_eq, _, _ = lf.trace_slice(A, "volA", 2, **kw)
        inside = lf.leak_table(model)
    z_after, _, _ = lf.trace_slice(A, "volA", 2, **kw)
    r_eq = rel(torch.from_numpy(z_eq["recon"]), torch.from_numpy(z0["recon"]))
    check("equalized_convs: every conv leak-free inside, output changes",
          n == len(before) and max(max(r["rows"], r["cols"], r["diag"]) for r in inside) < 1e-5
          and r_eq > 1e-5, f"{n} convs, output rel. change {r_eq:.1e}")
    check("equalized_convs: trained weights back on exit (bit-identical trace)",
          lf.leak_table(model) == before and all((z_after[k] == z0[k]).all() for k in z0))
    with lf.equalized_convs(model, convs=("synthesis",), levels=[2]):
        part = lf.leak_table(model)
    untouched = all(p == b for p, b in zip(part, before)
                    if not (p["level"] == 2 and p["conv"] == "synthesis"))
    hit = all(max(p["rows"], p["cols"], p["diag"]) < 1e-5 for p in part
              if p["level"] == 2 and p["conv"] == "synthesis")
    check("equalized_convs(convs, levels) touches only those", untouched and hit)
    try:
        with lf.equalized_convs(model):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("equalized_convs restores the weights after an error", lf.leak_table(model) == before)
    summ = lf.leak_summary(before)
    check("leak_summary: one row per (level, conv)",
          [(s[0], s[1]) for s in summ] == [(l, c) for l in (0, 1, 2) for c in ("analysis", "synthesis")],
          str([(s[0], s[1], s[2]) for s in summ]))

    # 1b: a new checkpoint is picked up ---------------------------------------
    time.sleep(1.1)
    os.utime(os.path.join(A, "net.ckpt"))
    lf.trace_slice(A, "volA", 2, **kw)
    check("a changed net.ckpt is reloaded",
          loads == ["mgA_R4", "mgB_R4", "lpdsflat_R4", "mgA_R4"], str(loads))
    print("\n" + lf.cache_info())

    de.load_run = real
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
