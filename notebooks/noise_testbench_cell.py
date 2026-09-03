# =============================================================================
# Is the grid actually adding a meaningful amount of noise?
#
# Paste into the notebook after `gnd_truth`, `smaps`, `mask` and `cg` exist.
#
# THE POINT. sigma is ABSOLUTE, not relative. `mri_awgn` adds
# `sigma * randn` (E|n|^2 = sigma^2) in the coil-image domain, so
# `noise_std=[0.04, 0.06]` means something completely different depending on
# what scale the image is on -- and the dataset multiplies the image by
# `scale_fac` (knee 5000, brain 2000) before the training loop ever sees it.
# The number that decides whether 0.05 is "a lot" is sigma / peak|x|, and this
# cell measures it rather than assuming it.
#
# Uses the repo's own prepare_measurement, so whatever it reports is literally
# what training does -- a re-implementation here could agree with the paper and
# still disagree with the code, which is the failure this is meant to catch.
# =============================================================================
import json

import matplotlib.pyplot as plt
import torch

from operators import FFT2D, Mask, Sense
from training.common import prepare_measurement

CONFIG = "config/knee/mg/lpdsnet_R8.json"   # the cell whose setup to replicate

# Set to 1.0 if `gnd_truth` in this notebook is ALREADY scaled (i.e. you loaded
# it through the dataset). Leave as the config's value if you read the h5
# directly, which is what the dataset does before it multiplies.
APPLY_SCALE_FAC = True

with open(CONFIG) as f:
    cfg = json.load(f)

scale_fac = cfg["data"]["train"].get("scale_fac", 1.0)
noise_std = cfg["training"]["noise_std"]
val_sigma = cfg["training"]["val_noise_std"]
R, acs = cfg["mri"]["R"], cfg["mri"]["acs_lines"]

x = gnd_truth * (scale_fac if APPLY_SCALE_FAC else 1.0)      # noqa: F821
sm = smaps                                                   # noqa: F821
while x.dim() < 4:
    x = x.unsqueeze(0)
while sm.dim() < 4:
    sm = sm.unsqueeze(0)
m = mask                                                     # noqa: F821
while m.dim() < 4:
    m = m.unsqueeze(0)

peak = float(x.abs().max())
rss_err = float((sm.abs().pow(2).sum(1).sqrt() - 1).abs().mean())

print(f"config      {CONFIG}")
print(f"            R={R} acs={acs}  noise_std={noise_std}  val={val_sigma}")
print(f"scale_fac   {scale_fac}  (applied: {APPLY_SCALE_FAC})")
print(f"peak |x|    {peak:.4g}")
print(f"maps        mean |RSS - 1| = {rss_err:.2e}   "
      f"{'unit-RSS, so sigma IS the adjoint noise std' if rss_err < 1e-2 else 'NOT unit-RSS -- sigma does not mean what the docstring says'}")
print()

# -----------------------------------------------------------------------------
# What each sigma actually does. The old range is included so the change this
# grid just made is visible as a number, not a claim.
# -----------------------------------------------------------------------------
def probe(sigma, seed=0):
    """One (y, adjoint, cg) at a pinned sigma, plus the SNR it produced."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    y, sig, extra = prepare_measurement(
        image=x, kspace=None, mask=m, smaps=sm,
        kspace_type="simulated", noise_std=[sigma, sigma],
        noise_dist=cfg["training"]["noise_dist"],
        whiten_kspace=False, generator=g,
    )
    smn = extra["smaps"]
    E = Mask(m) @ FFT2D() @ Sense(smn)

    # the SAME realisation at sigma=0, so the difference is noise alone and not
    # the undersampling -- comparing against x would fold both together and
    # report the aliasing as if it were noise
    g0 = torch.Generator(device=x.device).manual_seed(seed)
    y0, _, _ = prepare_measurement(
        image=x, kspace=None, mask=m, smaps=sm,
        kspace_type="simulated", noise_std=[0.0, 0.0],
        noise_dist=cfg["training"]["noise_dist"],
        whiten_kspace=False, generator=g0,
    )

    adj, adj0 = E.adjoint(y), E.adjoint(y0)
    npow = float((adj - adj0).abs().pow(2).mean())
    spow = float(x.abs().pow(2).mean())
    snr = 10 * torch.log10(torch.tensor(spow / max(npow, 1e-30)))
    return y, adj, float(sig.flatten()[0]), float(snr), E


print(f"  {'sigma':>7s} {'sigma/peak':>11s} {'adjoint SNR':>12s}   what it is")
print("  " + "-" * 62)
rows = [(0.0, "noiseless"), (0.005, "OLD val level"),
        (0.01, "OLD range, top"), (noise_std[0], "NEW range, bottom"),
        (val_sigma, "NEW val level"), (noise_std[1], "NEW range, top")]
probes = {}
for s, label in rows:
    y_s, adj_s, sig_s, snr_s, E_s = probe(s)
    probes[s] = (y_s, adj_s, E_s)
    rel = s / peak if peak else float("nan")
    print(f"  {s:7.4g} {rel:11.2e} {snr_s:11.2f} dB   {label}")

rel_new = val_sigma / peak if peak else float("nan")
print()
if rel_new < 1e-3:
    print("  *** WARNING: sigma is < 0.1% of peak signal. At this scale the")
    print("  *** noise is negligible and the grid is effectively noiseless --")
    print("  *** which is the symptom you set out to check. Either gnd_truth")
    print("  *** here is on a different scale than the dataset feeds the")
    print("  *** trainer (check APPLY_SCALE_FAC above), or noise_std needs to")
    print("  *** move with scale_fac.")
elif rel_new > 0.5:
    print("  *** WARNING: sigma exceeds half the peak signal. This is past")
    print("  *** 'challenging' and into 'the target is not recoverable'.")
else:
    print(f"  sigma is {rel_new:.1%} of peak signal at the val level. That is a")
    print("  real perturbation; compare the adjoint SNR column against the")
    print("  0.005 row to see how much the grid actually changed.")

# -----------------------------------------------------------------------------
# What a classical solver gets. This is the direct answer to "each
# reconstruction looks too good": CG-SENSE has no learned prior, so if IT still
# reconstructs cleanly at the new sigma, the task is genuinely easy and the
# problem is not the network.
# -----------------------------------------------------------------------------
lam = 0.1                                    # your Tikhonov, kept for comparability


def cg_sense(y, E, lam=lam, iters=200, tol=1e-3):
    def A(v):
        return E.normal(v) + lam * v
    out, ok = cg(A, E.adjoint(y), max_iter=iters, tol=tol, verbose=False)  # noqa: F821
    return out, ok


def psnr(a, b):
    rng = b.abs().max()
    mse = (a.abs() - b.abs()).pow(2).mean()
    return float(10 * torch.log10(rng ** 2 / (mse + 1e-30)))


show = [0.0, 0.01, val_sigma]
fig, axes = plt.subplots(2, len(show), figsize=(4.2 * len(show), 8))
print(f"\n  {'sigma':>7s} {'CG-SENSE PSNR':>14s} {'adjoint PSNR':>13s}  converged")
print("  " + "-" * 52)
for j, s in enumerate(show):
    y_s, adj_s, E_s = probes[s]
    rec, ok = cg_sense(y_s, E_s)
    print(f"  {s:7.4g} {psnr(rec, x):13.2f} {psnr(adj_s, x):13.2f}  {bool(ok)}")

    for i, (img, name) in enumerate(((adj_s, "adjoint"), (rec, "CG-SENSE"))):
        ax = axes[i, j]
        ax.imshow(img.detach().abs().squeeze().cpu(), cmap="gray",
                  vmin=0, vmax=float(x.abs().max()))
        ax.set_title(f"{name}  sigma={s:g}\nPSNR {psnr(img, x):.1f} dB", fontsize=10)
        ax.axis("off")
fig.suptitle(f"{CONFIG}  |  R={R}  |  peak|x|={peak:.3g}  |  "
             f"all panels on the ground truth's scale", fontsize=11)
fig.tight_layout()
plt.show()

print("\nRead it this way: if CG-SENSE at the new sigma is still close to its")
print("noiseless PSNR, the noise is not doing anything and the models will")
print("keep looking identical no matter what prior they carry.")
