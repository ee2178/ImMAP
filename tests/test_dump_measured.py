"""
The dump + viewer pipeline under the MEASURED protocol, with the real models.

Run with `python -m tests.test_dump_measured`.

`tests/test_dump_eval.py` covers the file contract and the viewer's helpers with
a stub network on simulated k-space. This runs the path the current experiments
actually take, end to end, through `scripts/dump_eval.py::main`:

  * measured k-space + AWGN, Julia mask (center_frac, adjust_accel), operator
    maps estimated ONLINE by ESPIRiT from each measurement's ACS;
  * three run directories holding real model classes at toy size --
      lpdsnet      MGLPDSNet, flat stack
      mg           MGLPDSNet V-cycle, coarse_op="rediscretize",
                   training.complex_conv="planar"   (the exp8 / exp9 cells)
      varnetmaps   E2EVarNet on the operator's maps, SENSE output
    each with a config.json and a net.ckpt, as train.py leaves them;
  * then `figures.common`, exactly as the viewer reads a dump: the volumes all
    columns share, the comparability check, and each column's stack.

It pins what would break a figure silently: a column that fails to dump, columns
that sample different slices or measurements, a zero-filled reference that comes
out dark (the placeholder-map failure `varnet` had), and a comparability check
that either misses or false-alarms on the new config keys.

ESPIRiT needs LAPACK: skipped where torch has none (the local anaconda build),
runs on the cluster.
"""

import os
import sys
import tempfile

import h5py
import numpy as np
import torch

import figures.common as fc

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import dump_eval as de                                          # noqa: E402
from models import build_model
from operators.fourier import fftc

FAIL = []
NC, H, W, NS = 4, 64, 64, 3
VOLS = ("volA", "volB")

MRI = {"R": 2, "acs_lines": None, "mask_dist": "uniform", "mask_offset": 0,
       "kspace_type": "measurement_awgn", "whiten_kspace": False,
       "center_frac": 0.2, "adjust_accel": True, "online_smaps": "espirit",
       "online_smaps_kws": {"thresh_eig": 0.0, "kernel_size": 4, "maxit": 10}}
LPDS = dict(M=8, C=1, P=3, s=2, lam0=1e-3, tau0=0.5, theta0=0.0, alpha0=1.0,
            is_complex=True, preproc="kspace", resize_noise=True)
RUNS = {
    "lpdsnet_R2": dict(model={"type": "MGLPDSNet", "params": dict(LPDS, K=2)},
                       pad=2, training={}),
    "mg_R2": dict(model={"type": "MGLPDSNet",
                         "params": dict(LPDS, K=[1, [2, 2, 2]], coarse_op="rediscretize")},
                  pad=8, training={"complex_conv": "planar"}),
    "varnetmaps_R2": dict(model={"type": "E2EVarNet",
                                 "params": dict(num_cascades=2, sens_chans=4, sens_pools=2,
                                                chans=4, pools=2, mask_center=True,
                                                use_smaps=True, output="sense",
                                                acs_lines="mask")},
                          pad=1, training={}),
}


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def have_lapack():
    try:
        torch.linalg.eigh(torch.eye(3, dtype=torch.complex64))
        torch.linalg.svd(torch.randn(4, 4, dtype=torch.complex64))
        return True
    except Exception:
        return False


def build_trees(root):
    """fastMRI-shaped trees whose k-space IS the encoded image, so the online
    ESPIRiT estimate has real coil structure to find."""
    ks, sm = os.path.join(root, "k"), os.path.join(root, "s")
    os.makedirs(ks); os.makedirs(sm)
    g = torch.Generator().manual_seed(0)
    yy = (torch.arange(H)[:, None] - H / 2).double()
    xx = (torch.arange(W)[None, :] - W / 2).double()
    coils = []
    for c in range(NC):
        a = 2 * np.pi * c / NC
        mag = torch.exp(-((yy - 0.5 * H * np.sin(a)) ** 2 + (xx - 0.5 * W * np.cos(a)) ** 2)
                        / (2 * (0.45 * H) ** 2))
        coils.append(mag * torch.exp(1j * (a + 0.02 * (yy + xx))))
    smaps = torch.stack(coils)
    smaps = (smaps / smaps.abs().pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)
    obj = (((yy / (0.4 * H)) ** 2 + (xx / (0.35 * W)) ** 2) < 1).float()
    for v in VOLS:
        img = torch.stack([obj * (1 + 0.3 * torch.rand(H, W, generator=g)) * (1 + s)
                           for s in range(NS)]).to(torch.complex64)
        k = fftc(smaps[None] * img[:, None])
        with h5py.File(os.path.join(ks, v + ".h5"), "w") as f:
            f.attrs["acquisition"] = "AXT2"
            f["kspace"] = k.numpy()
        with h5py.File(os.path.join(sm, v + ".h5"), "w") as f:
            f["smaps"] = (smaps[None] * (obj > 0)).expand(NS, -1, -1, -1).numpy()
            f["image"] = img.numpy()
    return ks, sm


def make_runs(root, ks, sm):
    import json
    for name, spec in RUNS.items():
        cfg = {
            "task": "recon",
            "experiment": {"name": name},
            "model": spec["model"],
            "data": {"val": dict(name="fastmri", task="recon", anatomy="brain",
                                 kspace_root=ks, smap_root=sm, scale_fac=1.0,
                                 pad_multiple=spec["pad"], batch_size=1, start_slice=0,
                                 end_slice=1, crop_size=None, center_crop=None,
                                 random_flips=False, organ_mask_source="smaps")},
            "training": dict({"use_organ_mask": True, "noise_dist": "uniform",
                              "noise_std": [0.01, 0.02], "val_noise_std": 0.015},
                             **spec["training"]),
            "mri": dict(MRI),
        }
        d = os.path.join(root, name)
        os.makedirs(d)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        torch.manual_seed(0)
        net = build_model(cfg)
        torch.save({"model_state_dict": net.state_dict(), "step": 0},
                   os.path.join(d, "net.ckpt"))


def main():
    if not have_lapack():
        print("[skip] online ESPIRiT needs LAPACK (svd, eigh); run this on the cluster")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        ks, sm = build_trees(tmp)
        runs, out = os.path.join(tmp, "runs"), os.path.join(tmp, "dumped")
        os.makedirs(runs)
        make_runs(runs, ks, sm)

        argv = sys.argv
        sys.argv = ["dump_eval.py", "--runs", runs, "--out", out, "--slices", "0:2",
                    "--n-volumes", "2", "--metrics", "psnr", "ssim", "nrmse",
                    "--workers", "0", "--device", "cpu"]
        try:
            de.main()
        finally:
            sys.argv = argv

        for name in RUNS:
            got = sorted(os.listdir(os.path.join(out, name))) if os.path.isdir(
                os.path.join(out, name)) else []
            check(f"{name}: dumped one file per volume", got == [v + ".h5" for v in VOLS],
                  str(got))

        # ---- read it back the way the viewer does ---------------------------
        fc.DUMP_ROOT = out
        fc.COLUMNS = [("Zero-filled", "@zero_filled"), ("LPDSNet", "lpdsnet_R2"),
                      ("MG-LPDS", "mg_R2"), ("E2E-VarNet", "varnetmaps_R2"),
                      ("Ground truth", None)]
        fc.NAME = fc.VARIANT = "measured"
        fc.HERE = tmp
        check("viewer: every column shares both volumes", fc.volumes() == list(VOLS),
              str(fc.volumes()))
        warn = fc.check_comparable("volA")
        check("viewer: the three columns are comparable (no warnings)", warn == [], str(warn))

        vols = {n: fc.load(n, "volA") for n in RUNS}
        ref = vols["lpdsnet_R2"].ref
        for name, v in vols.items():
            # the viewer's Volume holds MAGNITUDES (the h5 stacks are complex)
            ok = (v.img.shape == (2, H, W) and np.isfinite(v.img).all()
                  and float(v.img.max()) > 0 and sorted(v.metrics) == ["nrmse", "psnr", "ssim"]
                  and all(np.isfinite(m).all() for m in v.metrics.values()))
            check(f"{name}: a (2, {H}, {W}) magnitude stack, finite and nonzero, with metrics", ok,
                  f"{v.img.shape} psnr {np.round(v.metrics['psnr'], 1)}")
            check(f"{name}: same ground truth as the other columns",
                  np.allclose(v.ref, ref))
            a = v.attrs
            check(f"{name}: provenance carries the measured protocol",
                  a["kspace_type"] == "measurement_awgn" and a["online_smaps"] == "espirit"
                  and float(a["center_frac"]) == 0.2 and bool(a["adjust_accel"])
                  and int(a["acs_lines"]) == -1,
                  str({k: a[k] for k in ("kspace_type", "online_smaps", "center_frac",
                                         "adjust_accel", "acs_lines")}))

        # the measurement is shared: same seed -> same mask, noise, online maps
        with h5py.File(os.path.join(out, "lpdsnet_R2", "volA.h5")) as f1, \
                h5py.File(os.path.join(out, "varnetmaps_R2", "volA.h5")) as f2, \
                h5py.File(os.path.join(out, "mg_R2", "volA.h5")) as f3:
            zf = [f["zero_filled"][:] for f in (f1, f2, f3)]
            masks = [f["sampling_mask"][:] for f in (f1, f2, f3)]
            organ = f1["organ_mask"][:] > 0
            refa = np.abs(f1["reference"][:])
        check("all columns saw the same sampling mask",
              all((m == masks[0]).all() for m in masks))
        check("all columns saw the same measurement and online maps (zero-filled identical)",
              all(np.allclose(z, zf[0], atol=1e-5) for z in zf),
              f"max diff {max(float(np.abs(z - zf[0]).max()) for z in zf):.1e}")
        ratio = float(np.abs(zf[1])[organ].mean() / refa[organ].mean())
        check("the zero-filled reference is at the ground truth's scale, not dark "
              "(VarNet's column uses the operator's maps)", 0.3 < ratio < 1.5,
              f"mean |zf| / mean |gt| over the organ = {ratio:.2f}")

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
