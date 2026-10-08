"""
`scripts/count_flops.py`: the counter against numbers worked out by hand.

Run with `python -m tests.test_count_flops`.

A FLOP count is only as good as its bookkeeping, and every entry in it is easy
to get wrong by a constant factor that no result would flag. So each family is
checked against a closed form on a net small enough to do by hand:

  * conv        a flat LPDS stack: K layers of one analysis + one synthesis
                conv, 3 real convs each ("gauss") or 1 real conv on doubled
                channels ("planar"), so planar counts exactly 4/3 of gauss;
  * FFT         K Grams (K - 1 layers, plus the one the "kspace"
                preprocessing takes of a constant image) and the adjoint that
                forms y~: 2 transforms per coil per Gram, 5 n log2 n each;
  * attention   the group twin: one application per prox call, pixels x
                window^2 x (stacked query dims + value channels);
  * coarse_op   Galerkin runs every FFT on the fine grid, rediscretize on
                three. The conv count drops too: a Galerkin coarse Gram
                prolongs and restricts around the fine operator, and those
                grid transfers are convolutions;
  * --shapes-only gives the SAME counts as running the convolutions;
  * the group twin of `mglpds` is the generator's `mggrouplpds` cell.

Small CPU problem. The group cases need the group prox's init (a matrix norm),
which needs LAPACK: skipped where torch has none.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import count_flops as cf                                        # noqa: E402

FAIL = []
H, W, C = 64, 48, 4
M, P, K = 8, 3, 5
NET = dict(M=M, C=1, P=P, s=2, lam0=1e-3, tau0=0.5, theta0=0.0, alpha0=1.0,
           is_complex=True, preproc="kspace", resize_noise=True)
MRI = {"R": 4, "acs_lines": 8}
GROUP = dict(window=5, Mh=6, dK=2, nheads=1, sim_fun="distance", attn_backend="flex")


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def cfg_for(k, mode="gauss", type_="MGLPDSNet", **extra):
    return {"model": {"type": type_, "params": dict(NET, K=k, **extra)},
            "training": {"complex_conv": mode}, "mri": dict(MRI)}


def count(cfg, **kw):
    return cf.count_config(cfg, "t", (H, W), C, None, torch.device("cpu"), **kw)


def have_lapack():
    try:
        torch.linalg.matrix_norm(torch.randn(4, 4), ord=2)
        return True
    except Exception:
        return False


def test_flat_closed_form():
    r = count(cfg_for(K))
    # one real conv 1 -> M, PxP, stride 2: MACs = out pixels * M * P^2. The
    # cold start runs the analysis only; every later layer runs both.
    real_conv = (H // 2) * (W // 2) * M * P * P
    convs = (K - 1) * 2 + 1
    want = 2 * 3 * real_conv * convs                    # 2 FLOPs/MAC, gauss = 3 real convs
    check("flat LPDS, gauss: conv FLOPs = 2 * 3 * (out pixels * M * P^2) * (2K - 1)",
          r["flops"]["conv"] == want, f"{r['flops']['conv']:.0f} vs {want}")
    n = H * W
    # y~ = E^H y (1 transform per coil); then K Grams at 2 per coil: the
    # "kspace" preprocessing's E^H E 1, and one per layer after the cold start
    transforms = C * (1 + 2 * K)
    want_fft = 5 * n * math.log2(n) * transforms
    check("flat LPDS: FFT FLOPs = 5 n log2 n * coils * (1 + 2 K)",
          abs(r["flops"]["fft"] - want_fft) < 1e-6 * want_fft
          and r["fft_grids"] == {f"{H}x{W}": transforms},
          f"{r['flops']['fft']:.0f} vs {want_fft:.0f}; {r['fft_grids']}")
    check("flat LPDS: no attention", r["flops"].get("attention", 0) == 0)

    p = count(cfg_for(K, mode="planar"))
    check("planar counts exactly 4/3 of gauss (one 2C -> 2M conv vs three C -> M)",
          3 * p["flops"]["conv"] == 4 * r["flops"]["conv"] and p["complex_conv"] == "planar",
          f"{p['flops']['conv']:.0f} vs {r['flops']['conv']:.0f}")
    check("...and the FFT count does not depend on the conv mode",
          p["flops"]["fft"] == r["flops"]["fft"])


def test_shapes_only():
    for tag, cfg in (("flat", cfg_for(K)), ("V-cycle", cfg_for([2, [2, 2, 2]])),
                     ("V-cycle planar + rediscretize",
                      cfg_for([2, [2, 2, 2]], mode="planar", coarse_op="rediscretize"))):
        a, b = count(cfg), count(cfg, shapes_only=True)
        check(f"--shapes-only gives identical counts: {tag}",
              a["flops"] == b["flops"] and a["calls"] == b["calls"]
              and a["fft_grids"] == b["fft_grids"],
              f"{a['total']:.0f} vs {b['total']:.0f}")


def test_coarse_op():
    g = count(cfg_for([2, [2, 2, 2]]))
    r = count(cfg_for([2, [2, 2, 2]], coarse_op="rediscretize"))
    check("coarse_op: rediscretize also drops the transfer convs inside the coarse Grams",
          r["flops"]["conv"] < g["flops"]["conv"],
          f"{r['flops']['conv']:.0f} vs {g['flops']['conv']:.0f}")
    check("galerkin: every FFT runs on the fine grid", set(g["fft_grids"]) == {f"{H}x{W}"},
          str(g["fft_grids"]))
    check("rediscretize: FFTs on three grids, and fewer FFT FLOPs",
          set(r["fft_grids"]) == {f"{H}x{W}", f"{H // 2}x{W // 2}", f"{H // 4}x{W // 4}"}
          and r["flops"]["fft"] < g["flops"]["fft"],
          f"{r['fft_grids']}; {r['flops']['fft']:.0f} vs {g['flops']['fft']:.0f}")


def test_group():
    if not have_lapack():
        print("[skip] the group prox's init needs LAPACK (matrix_norm); run on the cluster")
        return
    base = count(cfg_for(K))
    grp = count(cfg_for(K, type_="MGGroupLPDS", **GROUP))
    # one application per prox call (K of them), on the stride-2 latent grid:
    # queries/keys are Mh complex dims stacked to 2 Mh real, values Mh channels
    pixels, w2, Mh = (H // 2) * (W // 2), GROUP["window"] ** 2, GROUP["Mh"]
    want = 2 * pixels * w2 * (2 * Mh + Mh) * K
    check("group twin: attention FLOPs = 2 * pixels * window^2 * (2 Mh + Mh) * K",
          grp["flops"]["attention"] == want and grp["attn_grids"] == {f"{H // 2}x{W // 2}": K},
          f"{grp['flops']['attention']:.0f} vs {want}; {grp['attn_grids']}")
    # W_theta, W_phi (once per adjacency build) and W_alpha, W_beta (every
    # call) are real 1x1 maps M <-> Mh, applied to re and im separately except
    # W_beta, which acts on a real envelope
    one = pixels * M * Mh
    builds = math.ceil(K / GROUP["dK"])
    extra = 2 * one * (2 * 2 * builds + 2 * K + K)
    check("group twin: the 1x1 maps add 2 * pixels * M * Mh * (4 builds + 3 K) conv FLOPs",
          grp["flops"]["conv"] - base["flops"]["conv"] == extra,
          f"{grp['flops']['conv'] - base['flops']['conv']:.0f} vs {extra} ({builds} builds)")
    check("group twin: FFT count unchanged", grp["flops"]["fft"] == base["flops"]["fft"])

    gather = count(cfg_for(K, type_="MGGroupLPDS", **dict(GROUP, attn_backend="gather")))
    want_g = 2 * pixels * w2 * (2 * Mh * builds + Mh * K)
    check("gather backend: similarity counted once per build, not per application",
          gather["flops"]["attention"] == want_g,
          f"{gather['flops']['attention']:.0f} vs {want_g}")

    twin = cf.group_twin(cfg_for([2, [2, 2, 2]]))
    from make_mg_recon_configs import MODELS
    gp = MODELS["mggrouplpds"]["params"]
    check("group_twin(): MGGroupLPDS with the generator's group settings",
          twin["model"]["type"] == "MGGroupLPDS"
          and all(twin["model"]["params"][k] == gp[k]
                  for k in ("window", "Mh", "dK", "nheads", "sim_fun", "attn_backend"))
          and twin["model"]["params"]["M"] == M
          and cf.group_twin({"model": {"type": "E2EVarNet", "params": {}}}) is None)


def main():
    for fn in (test_flat_closed_form, test_shapes_only, test_coarse_op, test_group):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
