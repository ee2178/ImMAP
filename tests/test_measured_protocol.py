"""
The measured protocol: measured k-space + known AWGN, SENSE ground truth.

Run with `python -m tests.test_measured_protocol`.

What has to hold for the retrained grid to mean what it says:

1. THE GROUND TRUTH IS THE STORED SENSE COMBINATION by default, and the RSS of
   the acquired coils only when asked for (`--target rss`).
2. THE MEASUREMENT IS THE SCAN: `y = mask . (k + sigma n)` from measured k-space,
   zero off the mask, with `sigma` the level the network is told.
3. SIGMA MEANS THE SAME THING as under `mri_awgn`, so a noise level carries over
   between the two protocols.
4. The generator writes the protocols as described, `legacy` reproduces the
   old configs key for key, and legacy + RSS (a phase-free phantom) is refused.
"""

import json
import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import torch

from operators.fourier import fftc, ifftc
from operators.noise import kspace_awgn, mri_awgn
from physics.mask import make_acc_mask

FAIL = []
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(name, ok, detail=""):
    print(f"[{'ok ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
def test_noise_convention_matches_mri_awgn():
    """Per-element noise power is sigma^2 under both measurement models."""
    g = torch.Generator().manual_seed(0)
    B, C, H, W = 2, 8, 64, 64
    ones = torch.ones(1, 1, H, W)
    zero_k = torch.zeros(B, C, H, W, dtype=torch.complex64)

    y, s = kspace_awgn(zero_k, ones, [0.05, 0.05], generator=g)
    emp = float(y.abs().pow(2).mean().sqrt())
    check("kspace_awgn: per-element RMS equals sigma", abs(emp - 0.05) < 0.002,
          f"{emp:.4f} vs 0.05")
    check("kspace_awgn: sigma is (B,1,1,1)", tuple(s.shape) == (B, 1, 1, 1),
          f"{tuple(s.shape)}")

    # mri_awgn on a zero image with unit-RSS maps: the same number
    sm = torch.ones(B, C, H, W, dtype=torch.complex64) / C ** 0.5
    y2, _, _ = mri_awgn(torch.zeros(B, 1, H, W, dtype=torch.complex64), ones, sm,
                        [0.05, 0.05], generator=torch.Generator().manual_seed(1))
    emp2 = float(y2.abs().pow(2).mean().sqrt())
    check("mri_awgn gives the same per-element RMS (same sigma convention)",
          abs(emp2 - emp) < 0.003, f"{emp2:.4f} vs {emp:.4f}")


def test_measurement_awgn_branch():
    from training.common import prepare_measurement

    B, C, H, W = 2, 6, 48, 40
    k = torch.randn(B, C, H, W, dtype=torch.complex64)
    sm = torch.randn(B, C, H, W, dtype=torch.complex64)
    img = torch.zeros(B, 1, H, W, dtype=torch.complex64)
    m = make_acc_mask((H, W), accel=4, center_frac=0.1, dim=1, mode="uniform",
                      adjust_accel=True)

    try:
        prepare_measurement(image=img, kspace=k, mask=m, smaps=sm,
                            kspace_type="measurement_awgn", noise_std=[0.05, 0.05],
                            noise_dist="uniform", whiten_kspace=False)
        check("the offline maps are refused as operator maps", False, "no error")
    except ValueError:
        check("the offline maps are refused as operator maps", True)

    # A stand-in estimator, so the branch runs without LAPACK. What matters is
    # that the operator gets ITS output, never the dataset's maps.
    import physics.online_smaps as om
    real, est = om.online_smaps, torch.randn(B, C, H, W, dtype=torch.complex64)
    om.online_smaps = lambda y, mask, method="espirit", **kw: est
    try:
        y, sig, extra = prepare_measurement(
            image=img, kspace=k, mask=m, smaps=sm, kspace_type="measurement_awgn",
            noise_std=[0.04, 0.06], noise_dist="uniform", whiten_kspace=False,
            generator=torch.Generator().manual_seed(0), online_smaps="espirit")
    finally:
        om.online_smaps = real
    off = (m == 0).expand_as(y)
    check("y is zero wherever the mask is", float(y[off].abs().max()) == 0.0)
    on = (m > 0).expand_as(y)
    check("y is the measured k-space (plus noise) where sampled",
          float((y[on] - k[on]).abs().mean()) < 0.1,
          f"mean |y-k| on the mask = {float((y[on] - k[on]).abs().mean()):.3f}")
    check("sigma is sampled inside noise_std, shape (B,1,1,1)",
          tuple(sig.shape) == (B, 1, 1, 1)
          and bool(((sig >= 0.04) & (sig <= 0.06)).all()),
          f"{sig.flatten().tolist()}")
    check("the operator gets the ONLINE maps, not the dataset's",
          torch.equal(extra["smaps"], est) and not torch.equal(extra["smaps"], sm))
    check("the dataset maps are kept aside (smaps_data), not used",
          torch.equal(extra["smaps_data"], sm))

    try:
        prepare_measurement(image=img, kspace=k, mask=m, smaps=sm,
                            kspace_type="measurement_awgn", noise_std=[0.05, 0.05],
                            noise_dist="uniform", whiten_kspace=True,
                            online_smaps="espirit")
        check("whitening is refused under measurement_awgn", False, "no error")
    except ValueError:
        check("whitening is refused under measurement_awgn", True)


def _fake_volume(root, C=4, S=3, H=32, W=24, seed=0):
    """A kspace file + a coil-combined file, laid out like the real ones."""
    rng = np.random.default_rng(seed)
    os.makedirs(f"{root}/ks", exist_ok=True)
    os.makedirs(f"{root}/sm", exist_ok=True)
    coil = (rng.standard_normal((S, C, H, W))
            + 1j * rng.standard_normal((S, C, H, W))).astype(np.complex64)
    k = fftc(torch.from_numpy(coil)).numpy()
    with h5py.File(f"{root}/ks/file_brain_AXT2_0.h5", "w") as f:
        f.create_dataset("kspace", data=k)
        f.attrs["acquisition"] = "AXT2"
    sm = (rng.standard_normal((S, C, H, W)) + 0j).astype(np.complex64)
    stored = np.full((S, H, W), 7.0 + 3.0j, dtype=np.complex64)   # NOT the RSS
    with h5py.File(f"{root}/sm/file_brain_AXT2_0.h5", "w") as f:
        f.create_dataset("smaps", data=sm)
        f.create_dataset("image", data=stored)
    return coil, stored


def test_loader_target_is_rss():
    from datasets.fastmri.loader import FastMRIDataset

    tmp = tempfile.mkdtemp()
    coil, stored = _fake_volume(tmp)
    scale = 3.0
    common = dict(anatomy="brain", kspace_root=f"{tmp}/ks", smap_root=f"{tmp}/sm",
                  scale_fac=scale, start_slice=1, end_slice=2, random_flips=False)

    ds = FastMRIDataset(target="rss", **common)
    _, _, image, mask, _ = ds[0]
    want = np.sqrt((np.abs(coil[1] * scale) ** 2).sum(0))[None]
    got = image.numpy()
    check("target='rss' returns the RSS of the acquired coils (scaled)",
          np.allclose(np.abs(got), want, rtol=1e-4, atol=1e-5),
          f"max err {np.abs(np.abs(got) - want).max():.2e}")
    check("the RSS target is real (zero imaginary part), complex dtype kept",
          image.dtype == torch.complex64 and float(image.imag.abs().max()) == 0.0)
    check("the organ mask covers the image grid",
          tuple(mask.shape[-2:]) == tuple(image.shape[-2:]))

    ds2 = FastMRIDataset(target="sense", **common)
    _, _, image2, _, _ = ds2[0]
    check("target='sense' still returns the stored combination",
          np.allclose(image2.numpy(), stored[1][None] * scale))

    try:
        FastMRIDataset(target="nope", **common)
        check("an unknown target is rejected", False, "no error")
    except ValueError:
        check("an unknown target is rejected", True)


def _dry_run(*extra):
    out = subprocess.run(
        [sys.executable, "scripts/make_mg_recon_configs.py", "--dry-run",
         "--only", "lpdsnet", "--anatomy", "brain", "--accels", "16", *extra],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    body = out.split("---\n", 1)[1].split("\n--- ")[0]
    return json.loads(body)


def test_generator_protocols():
    meas = _dry_run()
    mri, data = meas["mri"], meas["data"]["train"]
    check("default protocol is measured: measured k-space + AWGN",
          mri["kspace_type"] == "measurement_awgn")
    check("measured: ground truth is the stored SENSE combination (no target key)",
          "target" not in data, f"target={data.get('target')!r}")
    check("measured: Julia masks (center_frac 0.04, adjust_accel)",
          mri.get("center_frac") == 0.04 and mri.get("adjust_accel") is True)
    check("measured: operator maps estimated online with ESPIRiT",
          mri.get("online_smaps") == "espirit")

    rss = _dry_run("--target", "rss")
    check("--target rss switches only the ground truth",
          rss["data"]["train"].get("target") == "rss"
          and rss["mri"]["kspace_type"] == "measurement_awgn")

    bad = subprocess.run(
        [sys.executable, "scripts/make_mg_recon_configs.py", "--dry-run",
         "--only", "lpdsnet", "--protocol", "legacy", "--target", "rss"],
        cwd=ROOT, capture_output=True, text=True)
    check("legacy + rss (a phase-free phantom) is refused",
          bad.returncode != 0 and "needs --protocol measured" in (bad.stderr + bad.stdout))

    leg = _dry_run("--protocol", "legacy")
    mri, data = leg["mri"], leg["data"]["train"]
    check("legacy: synthetic k-space", mri["kspace_type"] == "simulated")
    off = subprocess.run(
        [sys.executable, "scripts/make_mg_recon_configs.py", "--dry-run",
         "--only", "lpdsnet", "--online-smaps", "off"],
        cwd=ROOT, capture_output=True, text=True)
    check("measured + --online-smaps off (offline maps in the operator) is refused",
          off.returncode != 0 and "reserved for the ground truth" in (off.stderr + off.stdout))
    check("legacy writes none of the new keys (old run dirs still match)",
          not any(k in mri for k in ("center_frac", "adjust_accel", "online_smaps"))
          and "target" not in data,
          f"mri keys {sorted(mri)}")

    walsh = _dry_run("--online-smaps", "walsh")
    check("--online-smaps walsh keeps everything else",
          walsh["mri"]["online_smaps"] == "walsh"
          and walsh["mri"]["kspace_type"] == "measurement_awgn")


def main():
    for fn in (test_noise_convention_matches_mri_awgn, test_measurement_awgn_branch,
               test_loader_target_is_rss, test_generator_protocols):
        print(f"\n--- {fn.__name__}")
        fn()
    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
