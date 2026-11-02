#!/usr/bin/env python3
"""
Does the clipped dual notice a known ghost? A test of z ~ lam sign(A x) on a real artifact.

    python scripts/ghost_force.py --dump .../lpdsnet_R8/eval_dump/file_brain_AXT2_200_2000182.h5 \
        --slice 0 --box 412 442 128 140 --out results/ghost_force/brain_R8_s0
    python scripts/ghost_force.py ... --ckpt trained_nets/.../net.ckpt        # trained filters and lam

The ghost g is the residual |recon| - |reference| inside --box (tapered). With x the reference, and
for each ghost amplitude a, one dual step from rest:

    z(a) = clip(A (x + a g), lam)

    changed   share of the box's codes whose clipped value moves by more than 10% of lam when the
              ghost is added: 0 means the prior cannot see it at all
    flipped   share of the box's codes on the clip whose sign the ghost reverses
    force     <g, B (z(a) - z(0))> / <g, g>: the prior's restoring pull on the ghost. The primal
              step is x - tau B z, so positive removes it. A linear prior would give
              a ||A g||^2 / ||g||^2 (`linear`), so force / linear is how much of that response
              survives the clip.

The clip of a complex code keeps its phase, so a ghost can move z without flipping anything.

Control: the same amplitudes on an ANATOMICAL structure of the same energy from the same box
(reference minus its blur, rescaled to ||g||). If the ghost's row looks like the control's, the prior
treats them alike. Both are read at the trained lam and at larger values. Magnitude images, since
the dumps' phases are not comparable. Without --ckpt the filters are random with unit norm, the
scale training reaches, so the numbers show the mechanism and not what a trained prior does.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def clip(c, lam):
    return c * torch.clamp(lam / c.abs().clamp_min(1e-12), max=1.0)


def random_bank(M, seed):
    from code_stats import first_layer
    from models.lpds import set_weight
    L = first_layer(M, seed)
    w = L.analysis.weight.detach()
    n = w.abs().pow(2).sum((1, 2, 3)).sqrt().view(-1, 1, 1, 1)
    set_weight(L.analysis, w / n)
    set_weight(L.synthesis, w.conj() / n)
    return [("random", L, None)]


def trained_banks(path, M, sigma):
    """One bank per fine-level layer: its analysis, synthesis and thresholds from the checkpoint."""
    from models.lpds import make_lpds_layer, set_weight
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    pres = sorted({k.split(".analysis.")[0] for k in sd if ".analysis." in k and ".dF." not in k
                   and "mglayer" not in k})
    out = []
    for pre in pres:
        a = torch.complex(sd[f"{pre}.analysis.conv_real.weight"], sd[f"{pre}.analysis.conv_imag.weight"])
        b = torch.complex(sd[f"{pre}.synthesis.conv_real.weight"], sd[f"{pre}.synthesis.conv_imag.weight"])
        L = make_lpds_layer(1, a.shape[0], spectral_init=False, P=a.shape[-1], stride=2, is_complex=True,
                            lam0=1e-3, tau0=0.5, theta0=0.0, degrees=1)
        set_weight(L.analysis, a)
        set_weight(L.synthesis, b)
        w = sd[f"{pre}.prox.prox.tau.weight"]
        w = w.reshape(w.shape[0], -1)
        lam = (w[0] + (w[1] * sigma if w.shape[0] > 1 else 0)).view(1, -1, 1, 1)
        out.append((pre, L, lam))
    return out, ck.get("step")


def measure(L, x, g, lam, box, amp):
    r0, r1, c0, c1 = box
    s = (slice(None), slice(None), slice(r0 // 2, r1 // 2), slice(c0 // 2, c1 // 2))
    gt = torch.from_numpy(g.astype(np.complex64))[None, None]
    xt = torch.from_numpy(x.astype(np.complex64))[None, None]
    with torch.no_grad():
        ax, ag = L.analysis(xt), L.analysis(gt)
        z0 = clip(ax, lam)
        z = clip(ax + amp * ag, lam)
        lam_map = lam * torch.ones_like(ax.real)
        changed = float(((z - z0).abs()[s] > 0.1 * lam_map[s]).float().mean())
        on_clip = z0.abs()[s] >= 0.99 * lam_map[s]
        turned = (z * z0.conj()).real[s] < 0
        flipped = float((turned & on_clip).float().sum() / on_clip.float().sum().clamp_min(1))
        back = L.synthesis(z - z0)
        gg = (gt.conj() * gt).real.sum()
        force = float((gt.conj() * back).real.sum() / gg)
        linear = float(amp * (ag.conj() * ag).real.sum() / gg)
    return dict(changed=changed, flipped=flipped, force=force, linear=linear, sat=float(on_clip.float().mean()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--slice", type=int, required=True)
    p.add_argument("--box", type=int, nargs=4, required=True, metavar=("R0", "R1", "C0", "C1"))
    p.add_argument("--ckpt")
    p.add_argument("--M", type=int, default=81)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.015)
    p.add_argument("--lams", type=float, nargs="+", default=[0.0132, 0.04, 0.13, 0.4],
                   help="scalar thresholds to try, when there is no --ckpt")
    p.add_argument("--lam-scales", type=float, nargs="+", default=[1, 3, 10],
                   help="with --ckpt: multiples of each channel's trained threshold")
    p.add_argument("--amps", type=float, nargs="+", default=[0.25, 1, 4])
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    from models.components import set_complex_mode
    set_complex_mode("planar")
    with h5py.File(a.dump, "r") as f:
        i = list(f["slice_index"][:]).index(a.slice)
        rec, ref = np.abs(f["recon"][i]), np.abs(f["reference"][i])
    r0, r1, c0, c1 = a.box
    win = np.zeros(ref.shape)
    win[r0:r1, c0:c1] = 1
    win = ndi.gaussian_filter(win, 1.5)
    ghost = (rec - ref) * win
    anat = (ref - ndi.gaussian_filter(ref, 2.0)) * win
    anat *= np.linalg.norm(ghost) / (np.linalg.norm(anat) + 1e-12)
    x = ref - ref.mean()

    if a.ckpt:
        banks, step = trained_banks(a.ckpt, a.M, a.sigma)
        head = f"trained filters, {len(banks)} fine layers (median over layers), step {step}"
        settings = [(f"lam x{s:g}", s) for s in a.lam_scales]
    else:
        banks = random_bank(a.M, a.seed)
        head = f"random unit-norm filters, M={a.M}, seed {a.seed}: the mechanism only, not a trained prior"
        settings = [(f"lam {v:g}", v) for v in a.lams]

    lines = [f"{os.path.basename(a.dump)} slice {a.slice}, box rows {r0}-{r1} cols {c0}-{c1}. {head}",
             f"||ghost|| = {np.linalg.norm(ghost):.3f}, the anatomical control is scaled to match",
             "", f"{'setting':<14}{'added':<9}{'amp':>5}{'on clip':>9}{'changed':>9}{'flipped':>9}"
             f"{'force':>9}{'linear':>9}{'force/lin':>10}"]
    for label, v in settings:
        for name, g in (("ghost", ghost), ("anatomy", anat)):
            for amp in a.amps:
                ms = []
                for _, L, lam in banks:
                    use = (lam * v) if lam is not None else torch.tensor(float(v))
                    ms.append(measure(L, x, g, use, a.box, amp))
                m = {k: float(np.median([q[k] for q in ms])) for k in ms[0]}
                lines.append(f"{label:<14}{name:<9}{amp:>5g}{m['sat']:>9.0%}{m['changed']:>9.0%}{m['flipped']:>9.1%}"
                             f"{m['force']:>9.3f}{m['linear']:>9.3f}{m['force'] / max(m['linear'], 1e-9):>10.3f}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
