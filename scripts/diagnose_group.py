#!/usr/bin/env python3
"""
Where does a group (MGGroupLPDS) net go numerically wrong?

    python scripts/diagnose_group.py --run trained_nets/mg_recon/brain/mggroup169v6_R8

A ONE-OFF DIAGNOSTIC, not training logging. It loads a run's `net.ckpt`, pushes
a few validation slices through it, and reports what the group thresholding did
at every layer, plus what the parameters look like and where one backward
pass's gradient goes. Nothing is written into the run except
`<run>/diagnose_group.json`.

WHAT `net.ckpt` IS: the checkpoint is only written on an IMPROVING epoch, so it
is the run's BEST state, not its diverged one. A run that blew up afterwards
left no copy of that. What this shows is the state it diverged FROM -- which is
where a precursor (an unclipped code, a dead threshold, a zeroed W_beta column)
would already be visible.

The group clip is `z * min(1, tau / (xi + eps))`, `xi` being a neighbourhood
energy envelope. Unlike the local clip, whose denominator is `|z|` itself, it
does not bound the code: where `xi` is small, `z` passes through whatever its
size. So per layer this reports

    max|z in|, max|z out|      the code before and after the clip
    out/tau                    max |z out| / tau. A LOCAL clip keeps this <= 1
    unclipped px / energy      share of pixels, and of the code's energy, that
                               passed with scale 1
    min xi, xi<tau             the envelope's floor, and how often it is under
                               the threshold (= unclipped)
    gain                       max |z| / (xi + eps) where the clip is active:
                               the factor in its derivatives
    self w                     (flex only) the attention weight a pixel gives
                               ITSELF. Query and key transforms are tied only
                               at init, so this is not guaranteed to be large
    max|x|                     the primal iterate after the layer

and for the parameters: thresholds at exactly zero, all-zero W_beta columns
(a channel with one is never clipped again), how far W_theta has moved from
W_phi, rho at its clamp, and the primal / coarse-correction step sizes.

The statistics pass runs the complex, eager path (the planar state and the
fused kernels compute the same map; they are switched off so every quantity
can be read). The gradient probe runs the run's OWN configuration.

--synthetic H W COILS uses a synthetic SENSE problem instead of the dataset
(the code statistics then say less; the parameter report is unaffected).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
#  per-layer statistics
# ---------------------------------------------------------------------------
class Recorder:
    """Wraps `GroupThreshold.forward` and `LPDSLayer.forward` while active."""

    def __init__(self):
        self.records, self.on = [], False

    def __enter__(self):
        from models.circulant_flex import FlexAdjacency
        from models.lpds import LPDSLayer
        from models.prox import GroupThreshold

        self._gt, self._layer = GroupThreshold, LPDSLayer
        self._orig_gt, self._orig_layer = GroupThreshold.forward, LPDSLayer.forward
        rec = self

        def gt_forward(prox, z, sigma=None, cache=None):
            if not rec.on or not prox.grouped:
                return rec._orig_gt(prox, z, sigma, cache)
            seen, beta = {}, prox.beta_apply

            def capture(x):
                seen["xi"] = beta(x)
                return seen["xi"]

            prox.beta_apply = capture
            try:
                out, cache = rec._orig_gt(prox, z, sigma, cache)
            finally:
                del prox.beta_apply
            try:
                rec.records.append(rec._stats(prox, z, sigma, seen["xi"], cache, FlexAdjacency))
            except Exception as e:                                # noqa: BLE001
                rec.records.append(dict(error=f"{type(e).__name__}: {e}"))
            return out, cache

        def layer_forward(layer, state, *a, **k):
            n = len(rec.records)
            out = rec._orig_layer(layer, state, *a, **k)
            if rec.on and len(rec.records) > n:
                x, z = out[0]
                rec.records[-1].update(
                    x_max=float(x.detach().abs().max()),
                    finite=bool(torch.isfinite(torch.view_as_real(x) if x.is_complex()
                                               else x).all()))
            return out

        GroupThreshold.forward, LPDSLayer.forward = gt_forward, layer_forward
        return self

    def __exit__(self, *exc):
        self._gt.forward, self._layer.forward = self._orig_gt, self._orig_layer

    @staticmethod
    @torch.no_grad()
    def _stats(prox, z, sigma, xi, cache, FlexAdjacency):
        eps = prox.eps
        tau = prox.tau(sigma, ref=z)
        a = z.abs()
        q = tau / (xi + eps)
        s = q.clamp_max(1.0)                                   # the dual clip's scale
        out = a * s
        free = q >= 1.0                                        # passed unclipped
        e = a * a
        tot = float(e.sum())
        tpos = tau.expand_as(a)
        r = dict(
            grid=f"{z.shape[-2]}x{z.shape[-1]}",
            z_in=float(a.max()), z_out=float(out.max()),
            out_over_tau=float(torch.where(tpos > 0, out / tpos.clamp_min(1e-30),
                                           torch.zeros_like(out)).max()),
            free_px=float(free.float().mean()),
            free_energy=(float(e[free].sum()) / tot) if tot > 0 else 0.0,
            free_z_max=float(torch.where(free, a, torch.zeros_like(a)).max()),
            xi_min=float(xi.min()), xi_med=float(xi.flatten().median()),
            xi_tiny=float((xi < 1e-6).float().mean()),
            gain=float(torch.where(~free, a / (xi + eps), torch.zeros_like(a)).max()),
            tau_min=float(tau.min()), tau_med=float(tau.flatten().median()),
            nonfinite=int((~torch.isfinite(a)).sum() + (~torch.isfinite(xi)).sum()),
        )
        G = (cache or {}).get("Gamma")
        try:
            if isinstance(G, FlexAdjacency) and G._lse is not None:
                # weight a pixel gives itself: exp(score_jj - log Z_j)
                score = (G.q * G.k).sum(-1) - (0.5 * G.ksq if G.sim in ("distance", "pidistance")
                                               else 0.0)
                w = torch.exp(score - G._lse).flatten()
                r.update(self_med=float(w.median()),
                         self_p01=float(torch.quantile(
                             w[:: max(1, w.numel() // 1_000_000)], 0.01)),
                         self_below_uniform=float((w < 1.0 / prox.window ** 2).float().mean()),
                         feat_sq_max=float(G.ksq.max()))
        except Exception as e:                                    # noqa: BLE001
            r["self_error"] = f"{type(e).__name__}: {e}"          # the rest still stands
        return r


# ---------------------------------------------------------------------------
#  parameters
# ---------------------------------------------------------------------------
@torch.no_grad()
def parameter_report(net, sigma):
    from models.lpds import LPDSLayer
    from models.prox import GroupThreshold

    s = torch.full((1, 1, 1, 1), float(sigma), device=next(net.parameters()).device)
    rows, steps = [], []
    for name, m in net.named_modules():
        if isinstance(m, GroupThreshold) and m.grouped:
            tau = m.tau(s, ref=s).flatten()
            wb = m.Wbeta.weight[:, :, 0, 0]                       # (Mh, M / heads)
            wt, wp = m.Wtheta.weight, m.Wphi.weight
            rho = m.rho(s, ref=s).flatten()
            rows.append(dict(
                name=name, M=m.M,
                tau_zero=int((tau <= 0).sum()), tau_min_pos=float(tau[tau > 0].min())
                if bool((tau > 0).any()) else 0.0, tau_med=float(tau.median()),
                beta_zero_cols=int((wb.abs().sum(0) == 0).sum()),
                beta_zero_frac=float((wb == 0).float().mean()),
                beta_col_min=float(wb.abs().sum(0).min()),
                alpha_fro=float(m.Walpha.weight.norm()),
                theta_phi_rel=float((wt - wp).norm() / wt.norm().clamp_min(1e-30)),
                rho_min=float(rho.min()), rho_at_clamp=int((rho <= 0.1 + 1e-6).sum())))
        elif isinstance(m, LPDSLayer):
            steps.append(dict(name=name, tau=float(m.tau(s, ref=s).max()),
                              theta=float(m.theta(s, ref=s).max())))
    alphas = {n: float(p.abs().max()) for n, p in net.named_parameters()
              if "alpha_x" in n or "alpha_z" in n}
    return rows, steps, alphas


def print_parameters(rows, steps, alphas):
    print("\n================ parameters ================")
    if not rows:
        print("  no grouped GroupThreshold in this net")
        return
    n = len(rows)
    tot = lambda k: sum(r[k] for r in rows)                                    # noqa: E731
    print(f"  {n} group prox modules, M={rows[0]['M']} channels each")
    print(f"  thresholds at (or below) zero:     {tot('tau_zero')} of {n * rows[0]['M']} "
          f"channel-slots; smallest positive {min(r['tau_min_pos'] for r in rows):.2e}, "
          f"median {sorted(r['tau_med'] for r in rows)[n // 2]:.2e}")
    print(f"  W_beta columns that are ALL zero:  {tot('beta_zero_cols')} "
          f"(such a channel has xi = 0 and is never clipped); zero entries "
          f"{100 * sum(r['beta_zero_frac'] for r in rows) / n:.1f}% on average, "
          f"smallest column sum {min(r['beta_col_min'] for r in rows):.2e}")
    print(f"  |W_theta - W_phi| / |W_theta|:     median "
          f"{sorted(r['theta_phi_rel'] for r in rows)[n // 2]:.2f}, max "
          f"{max(r['theta_phi_rel'] for r in rows):.2f}   (0 = still tied: a pixel is its "
          f"own best match)")
    print(f"  |W_alpha| (Frobenius):             min {min(r['alpha_fro'] for r in rows):.2e}, "
          f"max {max(r['alpha_fro'] for r in rows):.2e}")
    print(f"  rho: min {min(r['rho_min'] for r in rows):.3f}; channel-slots at the 0.1 "
          f"clamp: {tot('rho_at_clamp')}")
    worst = sorted(rows, key=lambda r: (-r["beta_zero_cols"], -r["tau_zero"]))[:3]
    for r in worst:
        if r["beta_zero_cols"] or r["tau_zero"]:
            print(f"    {r['name']}: {r['beta_zero_cols']} zero W_beta columns, "
                  f"{r['tau_zero']} zero thresholds")
    if steps:
        print(f"  primal step tau over {len(steps)} LPDS layers: min "
              f"{min(s['tau'] for s in steps):.3f}, max {max(s['tau'] for s in steps):.3f} "
              f"(init 0.5, clamped below only); theta max "
              f"{max(s['theta'] for s in steps):.3f}")
    if alphas:
        print(f"  coarse-correction steps (unclamped): max |alpha| "
              f"{max(alphas.values()):.3f} over {len(alphas)} tensors")


# ---------------------------------------------------------------------------
#  tables
# ---------------------------------------------------------------------------
def print_layers(records, title):
    good = [r for r in records if "error" not in r]
    bad = [r for r in records if "error" in r]
    print(f"\n================ {title} ================")
    if bad:
        print(f"  {len(bad)} layers could not be read: {bad[0]['error']}")
    if not good:
        return
    flex = any("self_med" in r for r in good)
    errs = [r["self_error"] for r in good if "self_error" in r]
    if errs:
        print(f"  self-weight could not be read on {len(errs)} layers: {errs[0]}")
    print(f"  {'#':>3} {'grid':<9}{'max|z in|':>10}{'max|z out|':>11}{'out/tau':>9}"
          f"{'unclip px':>10}{'energy':>8}{'min xi':>9}{'gain':>9}{'max|x|':>10}"
          + (f"{'self w':>8}{'<unif':>7}" if flex else ""))
    for i, r in enumerate(good):
        # a prox call outside an LPDS sweep is the V-cycle's correction term
        x = f"{r['x_max']:>10.2e}" if "x_max" in r else f"{'(V corr)':>10}"
        print(f"  {i:>3} {r['grid']:<9}{r['z_in']:>10.2e}{r['z_out']:>11.2e}"
              f"{r['out_over_tau']:>9.1f}{100 * r['free_px']:>9.1f}%"
              f"{100 * r['free_energy']:>7.1f}%{r['xi_min']:>9.1e}{r['gain']:>9.1e}{x}"
              + (f"{r['self_med']:>8.3f}{100 * r['self_below_uniform']:>6.0f}%"
                 if "self_med" in r else "")
              + ("   NON-FINITE" if r["nonfinite"] or not r.get("finite", True) else ""))
    print("  (V corr) = a prox evaluated for the V-cycle's coarse correction, not an "
          "LPDS sweep; self w = median\n  attention weight a pixel gives itself, <unif = "
          "share of pixels giving themselves less than 1/window^2.")


def print_levels(all_records):
    print("\n================ by grid level, all slices ================")
    grids = sorted({r["grid"] for r in all_records if "error" not in r},
                   key=lambda g: -int(g.split("x")[0]))
    print(f"  {'grid':<9}{'layers':>7}{'max|z in|':>10}{'max|z out|':>11}{'max out/tau':>12}"
          f"{'unclip px':>10}{'energy':>8}{'min xi':>9}{'xi<1e-6':>9}{'max gain':>10}")
    for g in grids:
        rs = [r for r in all_records if r.get("grid") == g]
        mean = lambda k: sum(r[k] for r in rs) / len(rs)                       # noqa: E731
        print(f"  {g:<9}{len(rs):>7}{max(r['z_in'] for r in rs):>10.2e}"
              f"{max(r['z_out'] for r in rs):>11.2e}{max(r['out_over_tau'] for r in rs):>12.1f}"
              f"{100 * mean('free_px'):>9.1f}%{100 * mean('free_energy'):>7.1f}%"
              f"{min(r['xi_min'] for r in rs):>9.1e}{100 * mean('xi_tiny'):>8.2f}%"
              f"{max(r['gain'] for r in rs):>10.1e}")
    print("  out/tau: a local clip keeps it <= 1. `energy` is the share of the code's "
          "energy that passed unclipped.")


# ---------------------------------------------------------------------------
#  gradient probe
# ---------------------------------------------------------------------------
FAMILIES = (("threshold tau", ".prox.prox.tau."), ("W_beta", ".Wbeta."), ("W_alpha", ".Walpha."),
            ("W_theta", ".Wtheta."), ("W_phi", ".Wphi."), ("rho", ".rho."),
            ("gamma", ".gamma."), ("alpha (coarse step)", "alpha_"),
            ("analysis", ".analysis."), ("synthesis", ".synthesis."),
            ("primal tau / theta", ".tau."), ("primal tau / theta", ".theta."))


def gradient_probe(net, loss_of, clip):
    net.train()
    net.zero_grad(set_to_none=True)
    loss = loss_of()
    loss.backward()
    fam, top, bad = {}, [], 0
    for n, p in net.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().pow(2).sum())
        if not math.isfinite(g2):
            bad += 1
            continue
        key = next((label for label, pat in FAMILIES if pat in "." + n), "other")
        fam[key] = fam.get(key, 0.0) + g2
        top.append((g2, n, float(p.grad.detach().abs().max())))
    total = math.sqrt(sum(fam.values())) if fam else float("nan")
    print("\n================ one backward pass (the run's own configuration) ================")
    print(f"  loss {float(loss):.4f}   gradient norm {total:.3e}"
          + (f"   -> clipped to {clip:g}: every gradient scaled by {min(1.0, clip / total):.2e}"
             if clip and total > 0 else "")
          + (f"   {bad} tensors with a NON-FINITE gradient" if bad else ""))
    for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22}{math.sqrt(v):>11.3e}   {100 * v / max(total ** 2, 1e-300):>5.1f}% of "
              f"the squared norm")
    print("  largest single tensors:")
    for g2, n, mx in sorted(top, reverse=True)[:5]:
        print(f"    {math.sqrt(g2):>10.3e}  (max entry {mx:.2e})  {n}")
    net.zero_grad(set_to_none=True)
    net.eval()
    return dict(loss=float(loss), grad_norm=total, nonfinite=bad,
                families={k: math.sqrt(v) for k, v in fam.items()})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir holding config.json and net.ckpt")
    ap.add_argument("--ckpt", default=None, help="another checkpoint for the same config")
    ap.add_argument("--n", type=int, default=3, help="validation slices")
    ap.add_argument("--sigma", type=float, default=None, help="default: the run's val noise")
    ap.add_argument("--synthetic", type=int, nargs=3, default=None, metavar=("H", "W", "COILS"))
    ap.add_argument("--no-grad-probe", action="store_true")
    ap.add_argument("--all-layers", action="store_true",
                    help="print the per-layer table for every slice, not just the first")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from evaluation.tasks import build_adapter, default_sigma
    from models import build_model
    from models.components import set_complex_mode
    from models.mg_lpds import MGLPDSNet, set_planar_state
    from models.prox import GroupThreshold, SoftThreshold
    from training.losses import LOSS_REGISTRY

    device = torch.device(args.device)
    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    train = cfg.get("training", {})
    set_complex_mode(train.get("complex_conv"))
    planar = set_planar_state(bool(train.get("planar_state", False)))
    net = build_model(cfg).to(device)
    path = args.ckpt or os.path.join(args.run, "net.ckpt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt.get("model_state_dict", ckpt))
    net.eval()
    if getattr(net, "attn_backend", None) == "flex" and device.type == "cuda":
        net.compile_flex()
    sigma = args.sigma if args.sigma is not None else default_sigma(cfg)
    steps_per_epoch = int(train.get("steps_per_epoch", 1) or 1)
    step = ckpt.get("step") if isinstance(ckpt, dict) else None
    print(f"{args.run}\n  {cfg['model']['type']}  checkpoint {path}"
          + (f"  step {step} (epoch {step // steps_per_epoch})" if step else "")
          + (f"  best_loss {ckpt.get('best_loss'):.4f}" if isinstance(ckpt, dict)
             and ckpt.get("best_loss") is not None else "")
          + (f"  backtracks {ckpt.get('backtrack_count')}" if isinstance(ckpt, dict) else "")
          + f"\n  sigma {sigma}   planar state in training: {planar}   device {device}")

    rows, steps, alphas = parameter_report(net, sigma)
    print_parameters(rows, steps, alphas)

    loss_fn = LOSS_REGISTRY[train.get("loss_type", "magnitude-nl1-nl2")]
    use_mask = bool(train.get("use_organ_mask", False))

    def masked(gt, out, m):
        return (gt, out) if (m is None or not use_mask) else (gt * m, out * m)

    # ---- the slices -------------------------------------------------------
    if args.synthetic:
        from scripts.time_net import _import_repo, build_problem
        from training.common import embed_for_net
        import scripts.time_net as tn
        if not hasattr(tn, "Mask"):
            _import_repo()
        H, W, coils = args.synthetic

        def problems():
            for i in range(args.n):
                y, E0, image = build_problem((H, W), coils, cfg.get("mri", {}),
                                             cfg.get("mri", {}).get("R", 8), device, seed=i)
                E, T = embed_for_net(net, E0, image, None)
                s = torch.full((1, 1, 1, 1), float(sigma), device=device)
                yield f"synthetic {i}", (lambda y=y, E=E, T=T, s=s, image=image:
                                         (image, T.forward(net(y, E=E, sigma=s)[0]), None))
    else:
        from datasets.registry import build_loader
        import datasets                                        # noqa: F401  registers loaders
        val = dict(cfg["data"]["val"])
        val.update(batch_size=1, num_workers=0)
        loader = build_loader(val, shuffle=False, drop_last=False)
        adapter = build_adapter(cfg["task"])
        every = max(1, len(loader) // max(args.n, 1))

        def problems():
            gen = torch.Generator(device=device).manual_seed(args.seed)
            k = 0
            for i, batch in enumerate(loader):
                if k >= args.n:
                    break
                if i % every:
                    continue
                k += 1
                yield f"val item {i}", (lambda batch=batch:
                                        adapter(net, batch, cfg, device, sigma, gen))

    # ---- statistics pass: complex state, eager, so everything can be read --
    keep = (MGLPDSNet.PLANAR_STATE, SoftThreshold.FUSED, GroupThreshold.FUSED_CHAIN)
    MGLPDSNet.PLANAR_STATE, SoftThreshold.FUSED, GroupThreshold.FUSED_CHAIN = False, False, False
    all_records, per_slice, first_call = [], [], None
    try:
        with Recorder() as rec, torch.no_grad():
            for i, (tag, run) in enumerate(problems()):
                rec.records, rec.on = [], True
                gt, out, m = run()
                rec.on = False
                first_call = first_call or run
                g, o = masked(gt, out, m)
                nrmse = float((g - o).abs().pow(2).sum().sqrt()
                              / g.abs().pow(2).sum().sqrt().clamp_min(1e-30))
                loss = float(loss_fn(g, o, sigma))
                finite = bool(torch.isfinite(torch.view_as_real(out)).all())
                print(f"\n{tag}: {tuple(gt.shape[-2:])}  NRMSE {nrmse:.4f}  loss {loss:.4f}"
                      + ("" if finite else "   OUTPUT NON-FINITE"))
                per_slice.append(dict(tag=tag, nrmse=nrmse, loss=loss, finite=finite,
                                      layers=rec.records))
                all_records += rec.records
                if i == 0 or args.all_layers:
                    print_layers(rec.records, f"per layer, {tag}")
    finally:
        MGLPDSNet.PLANAR_STATE, SoftThreshold.FUSED, GroupThreshold.FUSED_CHAIN = keep
    if all_records:
        print_levels(all_records)

    grad = None
    if not args.no_grad_probe and first_call is not None:
        def loss_of():
            gt, out, m = first_call()
            g, o = masked(gt, out, m)
            return loss_fn(g, o, sigma)
        with torch.enable_grad():
            grad = gradient_probe(net, loss_of, train.get("clip_grad"))

    dest = os.path.join(args.run, "diagnose_group.json")
    with open(dest, "w") as f:
        json.dump(dict(run=args.run, ckpt=path, step=step, sigma=sigma, parameters=rows,
                       lpds_steps=steps, alphas=alphas, slices=per_slice, gradient=grad),
                  f, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
