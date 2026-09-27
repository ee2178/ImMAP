"""
Filter logging for the multilevel / multigrid / wavelet nets (visualization/filters.py).

The load-bearing checks are that the image-domain atoms are RIGHT, not just
drawn: at init B = A^H, so the composed atoms must equal each layer's own
adjoint applied to a unit impulse.  Rendering needs torchvision (make_grid)
and is skipped without it.

Run with:  python -m tests.test_filter_logging
"""

import importlib.util
import sys

import torch

from models import build_model
from models.base import set_weight

PASS, FAIL = [], []
HAVE_TV = importlib.util.find_spec("torchvision") is not None


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'ok ' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


def _net(kind, K=4):
    if kind == "wavelet":
        p = dict(type="WaveletLPDSNet", params=dict(K=K, M=16, degrees=1, preproc="identity"))
    elif kind == "union":
        p = dict(type="WaveletLPDSNet", params=dict(K=K, M=32, degrees=1, preproc="identity",
                                                    family=["dtcwt", "haar"]))
    elif kind == "mllpds":
        p = dict(type="MLLPDSNet", params=dict(K=K, L=3, M=16, P=7, s=2, widen=2, degrees=1,
                                               preproc="identity"))
    elif kind == "cascade":
        p = dict(type="CascadeLPDSNet", params=dict(K=K, M=16, L=3, widen=4, degrees=1,
                                                    preproc="identity"))
    elif kind == "lpds":
        p = dict(type="MGLPDSNet", params=dict(K=K, M=16, P=7, s=2, degrees=1,
                                               preproc="identity"))
    torch.manual_seed(0)
    return build_model({"model": p})


def _vf():
    import visualization.filters as vf
    return vf


def test_banks_and_stats():
    vf = _vf()
    net = _net("mllpds", K=4)
    banks = vf.filter_banks(net)
    names = sorted(vf._short(t) for t in banks)
    check("banks grouped across iterations (ML-LPDS: 3 levels x A/B)",
          names == sorted(f"levels.{l}.{w}" for l in range(3)
                          for w in ("analysis", "synthesis"))
          and all(len(v) == 4 for v in banks.values()), str(names))

    init = vf.filter_snapshot(net)
    st = vf.filter_stats(net, init=init)
    check("at init: spread 0 (one prototype copied K times), drift 0",
          all(abs(v) < 1e-7 for v in st.values()) and len(st) == 12, f"{len(st)} scalars")

    lev = net.net.layers[2].levels[1].analysis          # train one bank of one iteration
    set_weight(lev, lev.weight * 1.5)
    st = vf.filter_stats(net, init=init)
    moved = {k: v for k, v in st.items() if v > 1e-6}
    check("a changed bank shows up in spread, and only it (drift is a median over k:"
          " one of four iterations moving leaves it at 0)",
          set(moved) == {"filters/spread/levels.1.analysis"}, str(sorted(moved)))
    # drift is a MEDIAN over iterations: one of four changed -> 0
    lev2 = net.net.layers[1].levels[1].analysis
    lev3 = net.net.layers[3].levels[1].analysis
    set_weight(lev2, lev2.weight * 1.5); set_weight(lev3, lev3.weight * 1.5)
    st = vf.filter_stats(net, init=init)
    check("drift (median over k) = 0.5 once most iterations moved by 1.5x",
          abs(st["filters/drift/levels.1.analysis"] - 0.5) < 1e-5,
          f"{st['filters/drift/levels.1.analysis']:.4f}")


@torch.no_grad()
def test_atoms_match_adjoint():
    vf = _vf()
    # wavelet: atoms = K^H e_m through the exact adjoint (Q and carries included)
    for kind in ("wavelet", "union"):
        net = _net(kind)
        atoms = vf.effective_atoms(net)
        lay = vf._atom_layer(net)[1]
        check(f"{kind}: atoms for levels 1-3", sorted(atoms) == [1, 2, 3])
        n_ch = sum(a.shape[0] for a in atoms.values())
        check(f"{kind}: one atom per deep channel", n_ch == lay.M, f"{n_ch} vs {lay.M}")
        worst = 0.0
        for lvl in (1, 2, 3):                               # every atom of every level
            chans = [i for i in range(lay.M) if lay.tags[i % lay.Cg][0] == lvl]
            z = torch.zeros(len(chans), lay.M, 16, 16, dtype=torch.complex64)
            z[torch.arange(len(chans)), chans, 8, 8] = 1
            ref = lay.adjoint(z)[:, 0].abs().pow(2).sum((-2, -1))   # B = A^H at init
            got = atoms[lvl][:, 0].abs().pow(2).sum((-2, -1))
            worst = max(worst, float(((ref - got).abs() / ref).max()))
        check(f"{kind}: every atom carries all of K^H e_m's energy (no crop clips one)",
              worst < 1e-5, f"worst rel energy err {worst:.1e}")
        widths = [atoms[l].shape[-1] for l in (1, 2, 3)]
        check(f"{kind}: crops shrink with the level's own support", widths[0] < widths[1] < widths[2],
              str(widths))

    # cascade / ML-LPDS: composed conv adjoint vs the layer's own adjoint
    net = _net("cascade")
    atoms = vf.effective_atoms(net)
    lay = vf._atom_layer(net)[1]
    z = torch.zeros(1, 256, 16, 16, dtype=torch.complex64); z[0, 5, 8, 8] = 1
    ref = lay.adjoint(z)[0, 0]
    e = abs(float(ref.abs().pow(2).sum() - atoms[3][5, 0].abs().pow(2).sum())) / float(ref.abs().pow(2).sum())
    check("cascade: level-3 atom = K^H e_m (energy)", e < 1e-5, f"{e:.1e}")
    check("cascade: 16/64/256 atoms per level",
          [atoms[l].shape[0] for l in (1, 2, 3)] == [16, 64, 256])

    net = _net("mllpds")
    atoms = vf.effective_atoms(net)
    lay = vf._atom_layer(net)[1]
    z = [None, torch.zeros(1, 16, 32, 32, dtype=torch.complex64),
         torch.zeros(1, 32, 16, 16, dtype=torch.complex64),
         torch.zeros(1, 64, 8, 8, dtype=torch.complex64)]
    z[3][0, 7, 4, 4] = 1
    ref = lay.adjoint(z)[0, 0]
    e = abs(float(ref.abs().pow(2).sum() - atoms[3][7, 0].abs().pow(2).sum())) / float(ref.abs().pow(2).sum())
    check("ML-LPDS: level-3 atom = A_(1,3)^H e_m (energy)", e < 1e-5, f"{e:.1e}")

    check("single-level LPDS: no atoms (its raw filters already are)",
          vf.effective_atoms(_net("lpds")) == {})


def test_render():
    if not HAVE_TV:
        print("[skip] test_render -- torchvision not installed")
        return
    vf = _vf()
    for kind in ("wavelet", "union", "mllpds", "cascade", "lpds"):
        imgs = vf.collect_filter_images(_net(kind))
        ok = imgs and all(t.dim() == 3 and t.shape[0] == 3 and 0 <= float(t.min())
                          and float(t.max()) <= 1 for t in imgs.values())
        check(f"{kind}: every image is (3, H, W) in [0, 1]", bool(ok), ", ".join(sorted(imgs)))
    check("E2E-VarNet: nothing to log, and no error",
          vf.collect_filter_images(build_model({"model": {"type": "E2EVarNet", "params": dict(
              num_cascades=1, sens_chans=2, sens_pools=1, chans=2, pools=1)}})) == {})


if __name__ == "__main__":
    test_banks_and_stats()
    test_atoms_match_adjoint()
    test_render()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)
