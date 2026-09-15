#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Are the shipped brain ESPIRiT maps good enough? Small ACS vs large ACS, on one
volume, scored and drawn side by side.

    python -m scripts.check_espirit_acs --anatomy brain --split val
    python -m scripts.check_espirit_acs --kspace /path/to/file1000000.h5 \
        --acs 20,48,96 --slices mid --out espirit_acs.png

The maps on disk were made with `--acs 20` (scripts/make_espirit_smaps.py), and
a 20x20 calibration region is SMALL for a 20-coil 8x8 kernel -- small enough
that the ESPIRiT calibration matrix is wider than it is tall, which is the first
thing this prints. Everything else here is the consequence: what the maps do to
the coil-combined image that the recon nets are trained to produce.

WHAT IT MEASURES, in the order the failure usually happens

  1. calibration shape     rows = (ax-ks+1)(ay-ks+1) patches, cols = ks^2 * C.
                           ESPIRiT splits the calibration matrix's row space
                           from its null space; when rows < cols there IS no
                           estimated null space -- the SVD returns at most
                           `rows` singular vectors out of a `cols`-dimensional
                           kernel space, and the retained subspace is whatever
                           those few patches happened to span. Flagged as
                           UNDERDETERMINED. For C=20, ks=8 you need ax >= 44.
  2. support               fraction of the FOV where the maps are nonzero, and
                           the fraction of the BRAIN they fail to cover. An
                           eigenvalue threshold that is too tight punches holes
                           in the object, and those pixels are unrecoverable by
                           any downstream net -- the forward model says they do
                           not exist.
  3. unit RSS              max | ||s|| - 1 | inside the support. operators/noise.py
                           ::mri_awgn assumes sum_c |s_c|^2 = 1; that assumption
                           is what makes sigma the noise std of the adjoint.
  4. model residual        || c - s x || / || c || over the brain, for the
                           coil-combined x = sum_c conj(s_c) c_c. THE headline
                           number: the fraction of the fully sampled coil data
                           the single-map SENSE model cannot represent. A net
                           handed these maps cannot do better than this, no
                           matter how good it is -- which is exactly the gap an
                           E2E-VarNet closes by estimating its own maps.
  5. signal retention      ||x|| / ||RSS|| over the brain, and the 5th percentile
                           of |x| / RSS. Oversmoothed maps combine the coils
                           incoherently and LOSE signal; this shows as shading
                           and as retention < 1 even where the support is fine.
  6. phase coherence       |boxcar mean of exp(i arg x)| over the brain, 1.0 =
                           smooth. `espirit()` references every map's phase to
                           COIL 0, so wherever coil 0 is dark its phase is noise
                           and the combined image inherits it. If this is low at
                           EVERY acs, the phase reference is the problem and the
                           ACS size is not.
  7. CG-SENSE at R         the maps used the way training uses them. NRMSE of
                           the accelerated reconstruction against the fully
                           sampled RSS, inside the brain. `--recon-R 0` skips it.

The comparison between map sets is done on the coil-combined IMAGES, never
element-wise on the maps: maps are defined only up to a per-pixel phase, so a
map-to-map diff measures the convention, not the quality.

DISPLAY. Every magnitude panel shares ONE window taken from the RSS of the
displayed slice, and every difference panel shares one symmetric window at
`--diff-scale` times that. No per-panel autoscaling -- a map set that loses 30%
of the signal has to LOOK 30% darker instead of being renormalised back into
agreement.

MEMORY. ESPIRiT's kernel images are (coils x retained kernels x the full grid),
complex, and the retained-kernel count grows with the ACS -- this is gigabytes
per slice and it is why the generator chunks. A large ACS here can be tens of
GB. The estimated upper bound is printed when a call fails, an OOM is caught and
reported rather than killing the run, and `--crop-readout` halves the grid by
removing brain's 2x readout oversampling (which also makes a square ACS a fairer
comparison: 20 of 640 readout samples is a 32-pixel map resolution over an FOV
that is half empty).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.fourier import fftc, ifftc                  # noqa: E402
from physics.smaps import espirit, espirit_soft            # noqa: E402

KSPACE_ROOTS = {
    "brain": "/home/ee2178/scratch/ee2178/datasets/fastmri/brain/multicoil_{split}",
    "knee": "/home/ee2178/scratch/ee2178/datasets/fastmri/knee/multicoil_{split}",
}
ACQ_FILTER = {"brain": ("T2",), "knee": ("CORPD_FBK",)}


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
def parse_acs(s):
    """'20,48x64,96' -> [(20, 20), (48, 64), (96, 96)]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if "x" in tok:
            a, b = tok.split("x")[:2]
        else:
            a = b = tok
        out.append((int(a), int(b)))
    if not out:
        raise ValueError("--acs is empty")
    return out


def pick_volume(root, anatomy, index):
    """The `index`-th .h5 under `root` whose acquisition matches the anatomy."""
    files = sorted(f for f in os.listdir(root) if f.endswith(".h5"))
    keep = []
    for f in files:
        try:
            with h5py.File(os.path.join(root, f), "r") as h:
                acq = str(h.attrs.get("acquisition", ""))
        except OSError:
            continue
        if any(t in acq for t in ACQ_FILTER.get(anatomy, ("",))):
            keep.append(f)
    if not keep:
        raise SystemExit(f"no volume in {root} matching {ACQ_FILTER.get(anatomy)}")
    return os.path.join(root, keep[index % len(keep)])


def resolve_slices(spec, S):
    if spec.strip() == "mid":
        return [S // 2]
    if spec.strip() == "all":
        return list(range(S))
    idx = [int(t) for t in spec.split(",") if t.strip()]
    bad = [i for i in idx if not 0 <= i < S]
    if bad:
        raise SystemExit(f"slice(s) {bad} out of range for a {S}-slice volume")
    return idx


def load_slices(path, spec):
    with h5py.File(path, "r") as f:
        if "kspace" not in f:
            raise SystemExit(f"{path} has no `kspace` dataset")
        ds = f["kspace"]
        if ds.ndim != 4:
            raise SystemExit(f"expected (S, C, H, W) kspace, got {ds.shape}")
        idx = resolve_slices(spec, ds.shape[0])
        k = np.stack([np.asarray(ds[i]) for i in idx])
    return torch.from_numpy(k).to(torch.complex64), idx


def crop_readout(k):
    """Drop the 2x readout oversampling: centre half of dim -2, via the image domain.

    fastMRI's frequency-encode axis (-2) is acquired at 2x FOV. Cropping it is a
    resampling of the object, so it happens in image space -- truncating k-space
    instead would blur the readout direction by a factor of two.
    """
    img = ifftc(k)
    H = img.shape[-2]
    img = img[..., H // 4: H // 4 + H // 2, :]
    return fftc(img)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def calib_shape(acs, ks, C):
    ax, ay = acs
    rows = max(ax - ks + 1, 0) * max(ay - ks + 1, 0)
    return rows, ks * ks * C


def boxcar(x, ks):
    """Same-size boxcar mean of a complex (B, 1, H, W) image."""
    w = torch.ones(1, 1, ks, ks, device=x.device, dtype=torch.float32) / (ks * ks)
    return torch.complex(F.conv2d(x.real, w, padding=ks // 2),
                         F.conv2d(x.imag, w, padding=ks // 2))


def subspace_residual(S, coil, brain, eps=1e-12):
    """|| c - P_span(S) c || / || c || over `brain`, for S as (B, M, C, H, W).

    M = 1 is the ordinary SENSE model residual; M > 1 is the soft-SENSE one. The
    maps from `espirit_soft` are orthonormal per pixel (eigenvectors of a
    Hermitian matrix), so the projector is the sum of the rank-1 projectors and
    no Gram inverse is needed.
    """
    proj = torch.zeros_like(coil)
    for m in range(S.shape[1]):
        s = S[:, m]
        n2 = s.abs().pow(2).sum(1, keepdim=True)
        proj = proj + s * ((s.conj() * coil).sum(1, keepdim=True) / (n2 + eps))
    num = (coil - proj).abs().pow(2).sum(1)[brain].sum()
    den = coil.abs().pow(2).sum(1)[brain].sum()
    return float((num / (den + eps)).sqrt())


def map_metrics(sm, coil, rss, brain, phase_ks):
    """Everything scored from one map set. `sm` (B, C, H, W), `coil` (B, C, H, W)."""
    x = (sm.conj() * coil).sum(1, keepdim=True)           # coil-combined (B,1,H,W)
    nrm = sm.abs().pow(2).sum(1).sqrt()                   # (B,H,W)
    sup = nrm > 1e-6

    inside = nrm[sup]
    rss_err = float((inside - 1).abs().max()) if inside.numel() else float("nan")

    xb, rb = x[:, 0][brain].abs(), rss[:, 0][brain]
    ratio = xb / rb.clamp_min(1e-12)

    # Phase coherence only where there IS a phase. Outside the support x is
    # exactly 0, whose angle is 0 -- a constant, i.e. perfectly "coherent". A
    # map set that dropped the whole object would otherwise score 1.00 here.
    coh = boxcar(torch.exp(1j * x.angle()), phase_ks).abs()
    lit = brain & (x[:, 0].abs() > 1e-6 * rss[:, 0].amax())
    coh_m = float(coh[:, 0][lit].mean()) if bool(lit.any()) else float("nan")

    return dict(
        x=x,
        support=float(sup.float().mean()),
        uncovered=float((brain & ~sup).float().sum() / brain.float().sum().clamp_min(1)),
        rss_err=rss_err,
        residual=subspace_residual(sm[:, None], coil, brain),
        retention=float(xb.norm() / rb.norm().clamp_min(1e-12)),
        p5=float(torch.quantile(ratio.float(), 0.05)) if ratio.numel() else float("nan"),
        phase_coh=coh_m,
    )


def compare(x, xref, brain, eps=1e-12):
    """Coil-combined `x` against the reference map set's `xref`, over the brain.

    Two numbers, because they answer different questions. The magnitude NRMSE is
    what a magnitude-domain metric (PSNR/SSIM on |x|) would see. The complex one
    removes a single GLOBAL phase first -- the whole image rotating is a
    convention difference and harmless; anything left is a per-pixel phase
    disagreement, which a complex-valued net sees as structure.
    """
    a, b = x[:, 0][brain], xref[:, 0][brain]
    mag = float((a.abs() - b.abs()).norm() / b.abs().norm().clamp_min(eps))
    phi = torch.angle(torch.sum(a.conj() * b))
    cpx = float((a * torch.exp(1j * phi) - b).norm() / b.norm().clamp_min(eps))
    return mag, cpx


def cg_sense_nrmse(sm, k, rss, brain, R, acs_lines, lamda, max_iter):
    """NRMSE of a CG-SENSE recon at acceleration `R` against the fully sampled RSS."""
    from physics.gfactor import cg_sense
    from physics.mask import make_acc_mask

    H, W = k.shape[-2:]
    # already (1, 1, H, W); `dim=1` undersamples the COLUMNS, the phase-encode
    # axis in fastMRI's (S, C, readout, phase) layout
    mask = make_acc_mask((H, W), accel=R, acs_lines=acs_lines, dim=1,
                         mode="uniform", device=k.device)
    recon = cg_sense(sm, mask, lamda=lamda, max_iter=max_iter)
    xr = recon(mask * k)
    a, b = xr[:, 0][brain].abs(), rss[:, 0][brain]
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def draw(rss, brain, runs, acs_list, coil, title, diff_scale, path=None):
    """The panel grid. Saves to `path` if given, and returns the figure either way.

    The backend is the CALLER's: `main` selects Agg before it gets here, and a
    notebook keeps its inline one. Choosing a backend inside a drawing function
    would make the same layout undrawable in the place it is most useful.
    """
    import matplotlib.pyplot as plt

    ok = [(a, r) for a, r in zip(acs_list, runs) if r is not None]
    if not ok:
        print("  nothing to draw -- every setting failed")
        return None
    n = len(ok)

    # ONE window for every magnitude panel, from the RSS of this slice, and one
    # symmetric window for every difference panel. Nothing is autoscaled.
    vmax = float(rss.abs().max())
    dmax = diff_scale * vmax
    g = dict(cmap="gray", vmin=0.0, vmax=vmax)
    d = dict(cmap="bwr", vmin=-dmax, vmax=dmax)

    fig, ax = plt.subplots(5, n + 1, figsize=(3.0 * (n + 1), 14.0), squeeze=False)
    for row in ax:
        for a in row:
            a.set_xticks([])
            a.set_yticks([])

    def show(a, im, t, **kw):
        a.imshow(np.asarray(im.squeeze().detach().cpu()), **kw)
        a.set_title(t, fontsize=8)

    def show_diff(a, im, t):
        """A difference panel, with its peak in the title.

        The window is FIXED across every difference panel, so a blank one is
        ambiguous on its own -- it means either 'no difference' or 'a difference
        smaller than +-{dmax}'. The number says which, without letting the panel
        rescale itself to make a 1% difference look alarming.
        """
        peak = float(im.abs().max())
        show(a, im, f"{t}   peak {peak / vmax:.1%} of RSS max", **d)

    show(ax[0][0], rss.abs(), "RSS (reference)", **g)
    show(ax[1][0], brain.float(), "brain mask", cmap="gray", vmin=0, vmax=1)
    show(ax[2][0], coil[0, 0].abs(), "coil 0 image", **g)
    ax[3][0].axis("off")
    if n >= 2:
        first, last = ok[0][1]["x"], ok[-1][1]["x"]
        show_diff(ax[4][0], (first[:, 0].abs() - last[:, 0].abs()),
                  f"|x| acs{ok[0][0][0]} - acs{ok[-1][0][0]}")
    else:
        ax[4][0].axis("off")

    for j, (acs, r) in enumerate(ok, start=1):
        tag = f"acs {acs[0]}x{acs[1]}"
        show(ax[0][j], r["x"][:, 0].abs(), f"{tag}  combined", **g)
        show(ax[1][j], (r["sm"].abs().pow(2).sum(1).sqrt() > 1e-6).float(),
             f"support {r['support']:.0%}", cmap="gray", vmin=0, vmax=1)
        show(ax[2][j], r["sm"][0, 0].abs(), "|s| coil 0", cmap="gray", vmin=0, vmax=1)
        show(ax[3][j], r["x"][:, 0].angle(), f"arg x  (coh {r['phase_coh']:.2f})",
             cmap="twilight", vmin=-np.pi, vmax=np.pi)
        show_diff(ax[4][j], (r["x"][:, 0].abs() - rss[:, 0]), "|x| - RSS")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {path}")
    return fig


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kspace", default=None,
                   help="one multicoil .h5; overrides --anatomy/--split")
    p.add_argument("--kspace-root", default=None)
    p.add_argument("--anatomy", choices=("brain", "knee"), default="brain")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--volume", type=int, default=0,
                   help="index into the filtered volume list")
    p.add_argument("--slices", default="mid",
                   help="'mid', 'all', or a comma list of indices")

    p.add_argument("--acs", default="20,48,96",
                   help="calibration sizes to compare; the FIRST is treated as "
                        "the shipped setting and the LAST as the reference")
    p.add_argument("--kernel-size", type=int, default=8)
    p.add_argument("--thresh-eig", type=float, default=0.95)
    p.add_argument("--thresh-rowspace", type=float, default=0.05)
    p.add_argument("--maxit", type=int, default=100,
                   help="espirit power-method iterations")
    p.add_argument("--soft-maps", type=int, default=0,
                   help="also score an M-map soft-SENSE fit at each ACS (0 = off). "
                        "If the 1-map residual is high and the 2-map one is not, "
                        "the model is the problem, not the calibration.")

    p.add_argument("--crop-readout", action="store_true",
                   help="remove the 2x readout oversampling before estimating")
    p.add_argument("--mask-thresh", type=float, default=0.05,
                   help="brain mask is RSS > this, with RSS scaled to max 1")
    p.add_argument("--phase-ks", type=int, default=5,
                   help="phase-coherence boxcar side")
    p.add_argument("--recon-R", type=int, default=4,
                   help="CG-SENSE acceleration; 0 to skip")
    p.add_argument("--recon-acs-lines", type=int, default=24)
    p.add_argument("--recon-lamda", type=float, default=1.0e-3)
    p.add_argument("--recon-iters", type=int, default=64)

    p.add_argument("--out", default="espirit_acs_check.png",
                   help="figure path; '' to skip")
    p.add_argument("--diff-scale", type=float, default=0.2,
                   help="difference panels use +-this x the RSS window")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    acs_list = parse_acs(args.acs)
    device = torch.device(args.device)

    path = args.kspace
    if path is None:
        root = args.kspace_root or KSPACE_ROOTS[args.anatomy].format(split=args.split)
        if not os.path.isdir(root):
            raise SystemExit(f"missing: {root}  (pass --kspace or --kspace-root)")
        path = pick_volume(root, args.anatomy, args.volume)

    kvol, sl_idx = load_slices(path, args.slices)
    if args.crop_readout:
        kvol = crop_readout(kvol)
    C = kvol.shape[1]
    Nx, Ny = kvol.shape[-2], kvol.shape[-1]

    print(f"volume   {path}")
    print(f"  slices {sl_idx}   grid {(Nx, Ny)}   coils {C}"
          f"{'   (readout oversampling cropped)' if args.crop_readout else ''}")
    print(f"  espirit kernel={args.kernel_size} thresh_eig={args.thresh_eig} "
          f"thresh_rowspace={args.thresh_rowspace} maxit={args.maxit}  device={device}")

    # ---- the calibration geometry, before computing anything --------------
    print(f"\ncalibration matrix  (cols = ks^2 x C = {args.kernel_size ** 2 * C})")
    print(f"  {'acs':>9}{'rows':>8}{'rows/cols':>11}{'map res (px)':>15}   verdict")
    for acs in acs_list:
        rows, cols = calib_shape(acs, args.kernel_size, C)
        res = f"{Nx / max(acs[0], 1):.0f} x {Ny / max(acs[1], 1):.0f}"
        verdict = ("IMPOSSIBLE -- the kernel does not fit in the ACS" if rows == 0
                   else "UNDERDETERMINED -- no null space is actually estimated"
                   if rows < cols else "ok" if rows >= 2 * cols else "marginal")
        print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}{rows:>8}{rows / cols:>11.2f}"
              f"{res:>15}   {verdict}")
    need = args.kernel_size - 1 + int(np.ceil(np.sqrt(args.kernel_size ** 2 * C)))
    print(f"  square ACS needs >= {need} per side for rows >= cols at "
          f"ks={args.kernel_size}, C={C}")

    # ---- per slice --------------------------------------------------------
    fig_state = None
    for si, s in enumerate(sl_idx):
        k = kvol[si:si + 1].to(device)
        coil = ifftc(k)
        rss = coil.abs().pow(2).sum(1, keepdim=True).sqrt()
        scale = rss.amax().clamp_min(1e-12)
        k, coil, rss = k / scale, coil / scale, rss / scale
        brain = rss[:, 0] > args.mask_thresh

        print(f"\nslice {s}   brain {float(brain.float().mean()):.1%} of FOV")
        print(f"  {'acs':>9}{'support':>9}{'brain unc':>11}"
              f"{'|RSS-1|':>10}{'residual':>10}{'retention':>11}{'p5 |x|/RSS':>12}"
              f"{'phase coh':>11}{'sec':>7}")

        runs = []
        for acs in acs_list:
            rows, cols = calib_shape(acs, args.kernel_size, C)
            if rows == 0:
                print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}  skipped: no patch "
                      f"fits -- kernel_size {args.kernel_size} exceeds the ACS")
                runs.append(None)
                continue
            ub = C * min(rows, cols) * Nx * Ny * 8 / 2 ** 30
            t0 = time.time()
            try:
                sm = espirit(k, acs_size=acs, kernel_size=args.kernel_size,
                             thresh_rowspace=args.thresh_rowspace,
                             thresh_eig=args.thresh_eig, maxit=args.maxit)
            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}  FAILED: "
                      f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")
                print(f"            kernel images need up to {ub:.1f} GB here -- try "
                      f"--crop-readout, --device cpu, or a smaller --acs")
                runs.append(None)
                continue
            dt = time.time() - t0

            r = map_metrics(sm, coil, rss, brain, args.phase_ks)
            r["sm"] = sm
            runs.append(r)
            print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}{r['support']:>9.1%}"
                  f"{r['uncovered']:>11.2%}{r['rss_err']:>10.1e}{r['residual']:>10.4f}"
                  f"{r['retention']:>11.4f}{r['p5']:>12.3f}{r['phase_coh']:>11.3f}"
                  f"{dt:>7.1f}")

            if args.soft_maps > 1:
                try:
                    sms = espirit_soft(k, acs_size=acs, kernel_size=args.kernel_size,
                                       thresh_rowspace=args.thresh_rowspace,
                                       thresh_eig=args.thresh_eig,
                                       num_maps=args.soft_maps)
                    rs = subspace_residual(sms, coil, brain)
                    del sms
                    print(f"  {'':>9}{'':>9}{'':>11}{'':>10}{rs:>10.4f}"
                          f"   <- {args.soft_maps}-map soft-SENSE residual")
                except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"  {'':>9}  soft-SENSE skipped: {type(e).__name__}")

        live = [(a, r) for a, r in zip(acs_list, runs) if r is not None]
        if len(live) >= 2:
            ref_acs, ref = live[-1]
            print(f"\n  coil-combined images vs the acs {ref_acs[0]}x{ref_acs[1]} "
                  f"reference, over the brain:")
            print(f"  {'acs':>9}{'|x| NRMSE':>12}{'complex NRMSE':>16}")
            for acs, r in live[:-1]:
                mag, cpx = compare(r["x"], ref["x"], brain)
                print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}{mag:>12.4f}{cpx:>16.4f}")

        if args.recon_R and live:
            print(f"\n  CG-SENSE at R={args.recon_R} (acs_lines={args.recon_acs_lines}, "
                  f"lamda={args.recon_lamda:g}), NRMSE vs fully sampled RSS:")
            for acs, r in live:
                try:
                    e = cg_sense_nrmse(r["sm"], k, rss, brain, args.recon_R,
                                       args.recon_acs_lines, args.recon_lamda,
                                       args.recon_iters)
                    print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}{e:>12.4f}")
                except Exception as exc:                      # noqa: BLE001
                    print(f"  {str(acs[0]) + 'x' + str(acs[1]):>9}  skipped: {exc}")

        # Only the first slice is drawn; the rest keep their numbers and drop
        # the full-grid tensors, which are the memory here.
        if fig_state is None:
            fig_state = (rss, brain, runs, coil)
        else:
            for r in runs:
                if r is not None:
                    r.pop("sm", None)
                    r.pop("x", None)

    if args.out and fig_state is not None:
        import matplotlib
        matplotlib.use("Agg")                 # headless; a notebook keeps its own
        rss, brain, runs, coil = fig_state
        draw(rss, brain, runs, acs_list, coil,
             f"{os.path.basename(path)}  slice {sl_idx[0]}   "
             f"ks={args.kernel_size} eig={args.thresh_eig} "
             f"rowspace={args.thresh_rowspace}",
             args.diff_scale, path=args.out)

    print("\nHOW TO READ IT")
    print("  residual and retention barely move between 20x20 and the large ACS")
    print("    -> the ACS is not what is hurting the nets; look at the phase")
    print("       coherence column and at the 1-map vs soft-SENSE residual.")
    print("  residual drops / retention rises with the larger ACS")
    print("    -> the shipped maps are calibration-starved; regenerate with")
    print("       scripts/make_espirit_smaps.py --acs <larger> and a new --out,")
    print("       remembering it rewrites `image` too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
