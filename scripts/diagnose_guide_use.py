#!/usr/bin/env python3
"""
Does a trained guided net (LGGS and its variants) actually USE its guide?

    python scripts/diagnose_guide_use.py trained_nets/guided_recon/knee/s3_noise-vlow/lggs_R8_g1
    python scripts/diagnose_guide_use.py RUN_A RUN_B ... --n 30 --json out.json

For each run, on its own fixed val subset (scripts/eval_guided_recon.py: the slices, mask, noise
and seed its validation uses), two measurements:

1. GUIDE ABLATION at test time -- the same trained weights, the same measurement, three guides:
     true      the guide the net was trained with
     rotated   the same guide rotated 180 degrees: identical contrast, intensities and
               statistics, WRONG anatomy -- the control for "uses the guide's content"
     zero      no guide signal (all zeros: no keys to match, no guide energy)
     x<s>      the true guide scaled by s (--scales; default 0.35 0.1): LGGS's guide shares the
               target's dictionary and W's, so its scale relative to the target is fixed by the
               data, not learned. An accelerated, zero-filled guide arrives at about 1/sqrt(R) of
               the fully-sampled one's amplitude (0.35 at R = 8). If a scaled-down guide scores
               BETTER than the true one, the guide's influence is too strong or miscalibrated.
   PSNR / SSIM / NRMSE on magnitudes, per slice against its own peak, as training validates.
   If `rotated` scores about as well as `true`, the net is not using what the guide shows; if
   `zero` scores as well too, it is not using the guide at all.

2. ATTENTION MASS ON THE GUIDE -- per layer, the share of the joint softmax that falls on the
   guide branches (vs the self branch), averaged over latent pixels and over the val slices,
   with the true guide. Near 0: the guide is ignored. High mass does not by itself mean the
   guide helps -- read it next to (1).

Runs whose net takes no guide (LPDSNet) are scored once, as a reference row.

Reads checkpoints only; writes nothing but the optional --json. Compiled glue is switched off
for the run (the attention-mass hook needs the eager joint softmax; same numbers).
"""

import argparse
import json
import os
import statistics
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.eval_guided_recon import (load_recon_run, measure, reconstruct,  # noqa: E402
                                       val_subset)
from training.metrics import compute_metrics                                # noqa: E402
from training.recon import _wants_guide                                     # noqa: E402
import models.guided_prox as gp                                             # noqa: E402
from models import compile_glue                                             # noqa: E402
from models.circulant_attention import Circulant                            # noqa: E402


# --------------------------------------------------------------------------------------------
# the attention-mass hook
# --------------------------------------------------------------------------------------------
class GuideMass:
    """Records, per prox call (= per layer, in order), the mean joint-softmax weight on the guide
    branches. Fused backends: from merge_joint's branch weights (softmax over the branches'
    log-sum-exps). Gather: from the materialised guide adjacencies' row sums."""

    def __init__(self):
        self.calls = []
        self._orig_merge = gp.merge_joint
        self._orig_gather = gp.GuidedGroupThreshold._gather_adjacencies

    def __enter__(self):
        rec = self.calls
        orig_merge, orig_gather = self._orig_merge, self._orig_gather

        def merge(outs, lses, nheads):
            B, C, H, W = outs[0].shape
            L = torch.stack([l.reshape(B, nheads, H, W) for l in lses], dim=0)
            pi = torch.softmax(L.float(), dim=0)
            rec.append(float(1.0 - pi[0].mean()))            # branch 0 is the self branch
            return orig_merge(outs, lses, nheads)

        def gather(self_, q, k_self, k_guides, spatial):
            Phi, Omegas = orig_gather(self_, q, k_self, k_guides, spatial)
            if self_.joint_softmax and Omegas and isinstance(Omegas[0], Circulant):
                rec.append(float(sum(Om.values.sum(-1).mean() for Om in Omegas)))
            return Phi, Omegas

        gp.merge_joint = merge
        gp.GuidedGroupThreshold._gather_adjacencies = gather
        return self

    def __exit__(self, *exc):
        gp.merge_joint = self._orig_merge
        gp.GuidedGroupThreshold._gather_adjacencies = self._orig_gather


# --------------------------------------------------------------------------------------------
def _ablate(guide, how):
    if guide is None or how == "true":
        return guide
    if how.startswith("x"):
        return guide * float(how[1:])
    if how == "rotated":
        return torch.flip(guide, dims=(-2, -1))
    if how == "zero":
        return torch.zeros_like(guide)
    raise ValueError(how)


def _score(run, m, recon):
    rng = run["cfg"]["training"].get("metric_range", 1.0)
    out = compute_metrics(m["gt"].abs(), recon.abs(), data_range=rng)
    return {k: float(v.mean() if torch.is_tensor(v) else v) for k, v in out.items()}


def diagnose(run_dir, n, noise, device, scales=(0.35, 0.1)):
    run = load_recon_run(run_dir, device=device)
    ds = val_subset(run["cfg"])
    n = min(n, len(ds))
    guided = _wants_guide(run["net"])
    conds = (("true", "rotated", "zero") + tuple(f"x{s:g}" for s in scales)
             if guided else ("true",))
    scores = {c: [] for c in conds}
    mass = []                                                  # per slice: [per layer]
    for i in range(n):
        m = measure(run, ds[i], noise=noise, seed=i)
        for c in conds:
            mm = dict(m, guide=_ablate(m["guide"], c))
            if c == "true" and guided:
                with GuideMass() as gm:
                    recon = reconstruct(run, mm)
                mass.append(gm.calls)
            else:
                recon = reconstruct(run, mm)
            scores[c].append(_score(run, m, recon))
    res = {"run": run["name"], "guided": guided, "n_slices": n}
    for c in conds:
        res[c] = {k: statistics.mean(s[k] for s in scores[c]) for k in scores[c][0]}
        if c != "true":
            res[f"{c}_minus_true_psnr"] = [a["psnr"] - b["psnr"]
                                           for a, b in zip(scores[c], scores["true"])]
    if mass and mass[0]:
        L = min(len(x) for x in mass)
        res["guide_mass_per_layer"] = [statistics.mean(x[l] for x in mass) for l in range(L)]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="trained run directories (config.json + ckpt)")
    ap.add_argument("--n", type=int, default=20, help="val slices per run")
    ap.add_argument("--noise", type=float, default=None, help="added noise std (None: the run's)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--scales", type=float, nargs="*", default=[0.35, 0.1],
                    help="guide scale factors to test (empty: none)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    compile_glue.set_mode(None)          # eager joint softmax, so the mass hook sees it

    results = []
    for rd in a.runs:
        r = diagnose(rd, a.n, a.noise, a.device, scales=a.scales)
        results.append(r)
        t = r["true"]
        print(f"\n== {r['run']}  ({r['n_slices']} val slices)")
        print(f"  {'guide':10s}{'PSNR':>9s}{'SSIM':>9s}{'NRMSE':>9s}")
        for c in [k for k in r if k in ("true", "rotated", "zero") or
                  (k.startswith("x") and isinstance(r[k], dict))]:
            if c in r:
                s = r[c]
                print(f"  {c:10s}{s['psnr']:9.3f}{s['ssim']:9.4f}{s['nrmse']:9.4f}")
        for c in [k[:-len("_minus_true_psnr")] for k in r if k.endswith("_minus_true_psnr")]:
            if c in r:
                d = r[f"{c}_minus_true_psnr"]
                worse = sum(x < 0 for x in d)
                print(f"  {c} - true PSNR: mean {statistics.mean(d):+.3f} dB, "
                      f"worse on {worse}/{len(d)} slices")
        if "guide_mass_per_layer" in r:
            gm = r["guide_mass_per_layer"]
            pick = sorted({0, len(gm) // 4, len(gm) // 2, 3 * len(gm) // 4, len(gm) - 1})
            print("  softmax mass on the guide, by layer: "
                  + ", ".join(f"L{l}: {gm[l]:.2f}" for l in pick)
                  + f"  (mean {statistics.mean(gm):.2f})")
        if not r["guided"]:
            print("  (no guide: reference row)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
