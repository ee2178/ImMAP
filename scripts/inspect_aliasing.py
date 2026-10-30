"""
Look at the aliasing in a dumped volume: error images and their wavelet details.

For each (volume, slice) this renders, from the `eval_dump/<volume>.h5` files
`scripts/dump_eval.py` writes:

  errors.png   reference | zero-filled - ref | [CG-SENSE - ref] | <net> - ref ...
  wavelet.png  stationary wavelet details of zero-filled - ref (and of each
               net's error), levels 1..L, three orientations each

The phase-encode (PE) axis is the one the sampling mask varies along. Aliasing
from equispaced sampling is copies of the image's PE-high-pass part, shifted
along PE by the peaks of the mask's point spread function, so it belongs in
the subbands that are DETAIL along PE ("PE-hi"); the "RO-hi / PE-lo" subband
should hold almost none. The PSF shifts are printed and drawn as ticks.

Errors are MAGNITUDE errors, |x| - |ref|. The complex ones are meaningless
here: the reference's phase comes from different coil maps than the
operator's (online ESPIRiT), so recon and zero-filled both differ from it by a
per-pixel phase with fine structure that no smooth correction removes. A copy
`a` laid over tissue `x` shows up in |x + a| - |x| as Re(a conj(x) / |x|), and
over background as |a|, so the copies stay visible.

`cg_sense` is shown only when the dump has it (dump_eval.py --cg-sense-lam).

    python scripts/inspect_aliasing.py --anatomy knee --volume file1000264 --slice 14
"""

from __future__ import annotations

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pywt  # noqa: E402

NETS = (("lpdsnet", "LPDSNet"), ("varnetmaps", "E2E-VarNet"), ("mg81v6", "MGLPDS"))


def load(root, anatomy, run, volume, sl):
    path = os.path.join(root, anatomy, run, "eval_dump", f"{volume}.h5")
    with h5py.File(path) as h:
        idx = list(h["slice_index"][:])
        if sl not in idx:
            raise SystemExit(f"{path}: slice {sl} not dumped (have {idx})")
        i = idx.index(sl)
        out = {k: h[k][i] for k in ("reference", "recon", "zero_filled",
                                    "sampling_mask", "organ_mask", "cg_sense") if k in h}
        out["attrs"] = dict(h.attrs)
    return out


def pe_axis(mask):
    """The axis the sampling pattern varies along."""
    return 1 if (mask != mask[:, :1]).any() else 0


def psf_shifts(mask, axis, n_peaks=4):
    """Aliasing shifts (px) along PE for an equispaced mask: k N / spacing.

    The spacing is the most common gap between sampled lines (the ACS block's
    gaps of 1 are a minority). Peak-picking the PSF itself is fragile: the
    ACS block's sinc ripple sits on top of the comb at small shifts.
    """
    line = mask.take(0, axis=1 - axis).astype(bool)
    N = len(line)
    gaps = np.diff(np.nonzero(line)[0])
    gaps = gaps[gaps > 1]
    if not len(gaps):
        return []
    vals, counts = np.unique(gaps, return_counts=True)
    spacing = vals[np.argmax(counts)]
    return [int(round(k * N / spacing)) for k in range(1, n_peaks + 1)
            if round(k * N / spacing) <= N // 2]


def crop_box(organ, pad=8):
    rows, cols = np.nonzero(organ)
    if not len(rows):
        return slice(None), slice(None)
    return (slice(max(rows.min() - pad, 0), rows.max() + pad + 1),
            slice(max(cols.min() - pad, 0), cols.max() + pad + 1))


def swt_details(x, levels, wavelet):
    """|stationary wavelet detail| per level, keyed by orientation.

    Complex input: real and imaginary parts are transformed separately and
    combined in magnitude. Keys follow pywt.swtn: one letter per axis
    (axis0, axis1), 'a' approximation, 'd' detail.
    """
    step = 2 ** levels
    H, W = x.shape
    ph, pw = (-H) % step, (-W) % step
    xp = np.pad(x, ((0, ph), (0, pw)))
    parts = [pywt.swtn(np.real(xp), wavelet, levels, axes=(0, 1)),
             pywt.swtn(np.imag(xp), wavelet, levels, axes=(0, 1))]
    out = []
    # swtn returns coarsest level first
    for lev_re, lev_im in zip(parts[0][::-1], parts[1][::-1]):
        out.append({k: np.hypot(lev_re[k], lev_im[k])[:H, :W]
                    for k in ("ad", "da", "dd")})
    return out                                      # out[0] = finest level


def subband_names(axis_pe):
    """pywt keys -> readable names, given which axis is PE."""
    if axis_pe == 1:   # rows = readout (axis 0), columns = PE (axis 1)
        return {"ad": "PE-hi / RO-lo", "da": "RO-hi / PE-lo", "dd": "PE-hi / RO-hi"}
    return {"da": "PE-hi / RO-lo", "ad": "RO-hi / PE-lo", "dd": "PE-hi / RO-hi"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="trained_nets/mg_recon")
    p.add_argument("--anatomy", required=True)
    p.add_argument("--volume", required=True)
    p.add_argument("--slice", type=int, required=True)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--wavelet", default="db2")
    p.add_argument("--out", default="results/artifact_inspect")
    args = p.parse_args()

    tag = f"{args.anatomy}_{args.volume}_s{args.slice}_R{args.R}"
    os.makedirs(args.out, exist_ok=True)

    base = load(args.root, args.anatomy, f"lpdsnet_R{args.R}", args.volume, args.slice)
    ref, zf, mask = base["reference"], base["zero_filled"], base["sampling_mask"]
    organ = base.get("organ_mask", np.ones(ref.shape, np.uint8))
    ax = pe_axis(mask)
    shifts = psf_shifts(mask, ax)
    N = mask.shape[ax]
    print(f"[{tag}] kspace_type={base['attrs'].get('kspace_type')}  PE axis={ax} (N={N})  "
          f"sampled {int(mask.take(0, axis=1 - ax).sum())} lines  PSF peaks at {shifts} px")

    def err(x):
        return np.abs(x) - np.abs(ref)

    errs = [("zero-filled - ref", err(zf))]
    if "cg_sense" in base:
        lam = base["attrs"].get("cg_sense_lam")
        errs.append((f"CG-SENSE (lam={lam:g}) - ref", err(base["cg_sense"])))
    for run, label in NETS:
        try:
            d = load(args.root, args.anatomy, f"{run}_R{args.R}", args.volume, args.slice)
        except (OSError, SystemExit) as e:
            print(f"  skip {run}: {e}")
            continue
        if d["attrs"].get("kspace_type") != base["attrs"].get("kspace_type"):
            print(f"  NOTE {run}: kspace_type {d['attrs'].get('kspace_type')} vs "
                  f"{base['attrs'].get('kspace_type')} for the zero-filled")
        errs.append((f"{label} - ref", err(d["recon"])))

    rs, cs = crop_box(organ)
    ref_c = np.abs(ref)[rs, cs]
    # two display scales: the linear recons, and the nets (much smaller errors)
    lin = [i for i, (n, _) in enumerate(errs) if n.startswith(("zero", "CG"))]
    net = [i for i in range(len(errs)) if i not in lin]
    vmax_ref = np.percentile(ref_c, 99.5)
    vmax_lin = np.percentile(np.abs(errs[0][1])[rs, cs], 99.5)
    vmax_net = (np.percentile(np.concatenate([np.abs(errs[i][1])[rs, cs].ravel() for i in net]),
                              99.5) if net else vmax_lin)

    # ---- error images ---------------------------------------------------------
    n = 1 + len(errs)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.2 * ref_c.shape[0] / ref_c.shape[1] + 0.8))
    axs[0].imshow(ref_c, cmap="gray", vmax=vmax_ref)
    axs[0].set_title("reference")
    for a, (name, e), i in zip(axs[1:], errs, range(len(errs))):
        vm = vmax_lin if i in lin else vmax_net
        a.imshow(e[rs, cs], cmap="RdBu_r", vmin=-vm, vmax=vm)
        a.set_title(f"{name}\n(scale {vm / vmax_lin:.2g}x ZF)" if i in net else name, fontsize=9)
    for a in axs:
        a.set_xticks([]), a.set_yticks([])
    # PE shift ticks on the reference panel: where a copy of its centre would land
    c0 = (cs.start or 0) if ax == 1 else (rs.start or 0)
    for s in shifts:
        if ax == 1:
            axs[0].axvline(ref.shape[1] // 2 - c0 + s, color="c", lw=0.6, alpha=0.6)
        else:
            axs[0].axhline(ref.shape[0] // 2 - c0 + s, color="c", lw=0.6, alpha=0.6)
    axs[0].set_xlim(-0.5, ref_c.shape[1] - 0.5)       # a tick past the crop must not widen it
    axs[0].set_ylim(ref_c.shape[0] - 0.5, -0.5)
    fig.suptitle(f"{tag}  |  PE axis {'horizontal' if ax == 1 else 'vertical'}, "
                 f"PSF shifts {shifts} px (cyan: centre + shift)", fontsize=10)
    fig.tight_layout()
    f1 = os.path.join(args.out, f"{tag}_errors.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # ---- wavelet details --------------------------------------------------------
    names = subband_names(ax)
    keys = [k for k in ("ad", "da", "dd") if names[k].startswith("PE-hi / RO-lo")] + \
           [k for k in ("ad", "da", "dd") if names[k].startswith("RO-hi / PE-lo")] + ["dd"]
    rows = []
    for name, e in errs:
        rows.append((name, swt_details(e, args.levels, args.wavelet)))
    ncol = 3 * len(rows)
    fig, axs = plt.subplots(args.levels, ncol,
                            figsize=(2.3 * ncol, 2.3 * args.levels * ref_c.shape[0] / ref_c.shape[1] + 0.8),
                            squeeze=False)
    energy = []
    for r, (name, dets) in enumerate(rows):
        for lev in range(args.levels):
            # one colour scale per error and level, shared by its three orientations
            vm = max(np.percentile(dets[lev][k][rs, cs], 99.5) for k in keys) + 1e-12
            for j, k in enumerate(keys):
                a = axs[lev, 3 * r + j]
                a.imshow(dets[lev][k][rs, cs], cmap="magma", vmax=vm)
                a.set_xticks([]), a.set_yticks([])
                if lev == 0:
                    a.set_title(f"{name}\n{names[k]}", fontsize=7)
                if 3 * r + j == 0:
                    a.set_ylabel(f"level {lev + 1}", fontsize=8)
            energy.append((name, lev + 1, *[float((dets[lev][k][rs, cs] ** 2).sum()) for k in keys]))
    fig.suptitle(f"{tag}  |  |stationary {args.wavelet} detail| of each error "
                 f"(per error and level: one scale for its 3 orientations)", fontsize=10)
    fig.tight_layout()
    f2 = os.path.join(args.out, f"{tag}_wavelet.png")
    fig.savefig(f2, dpi=110)
    plt.close(fig)

    print(f"  energy share per level  [{', '.join(names[k] for k in keys)}]")
    for name, lev, *es in energy:
        tot = sum(es) + 1e-30
        print(f"    {name:<22} L{lev}  " + "  ".join(f"{v / tot:5.1%}" for v in es))
    print(f"  wrote {f1}\n  wrote {f2}")


if __name__ == "__main__":
    main()
