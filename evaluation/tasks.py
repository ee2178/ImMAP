"""
Task adapters: how to turn one dataloader batch into a `(gt, recon)` pair.

This is the only part of the sweep that knows what a task IS. Everything else
-- metrics, aggregation, CSV -- is task agnostic, so adding a task means adding
one function here and nothing else.

An adapter has the signature

    adapter(net, batch, cfg, device, sigma, generator, extras=None)
        -> (gt, recon, organ_mask_or_None)

both complex or both real, shape `(B, C, H, W)`, on `device`.

`extras`, when a dict is passed, is FILLED IN with whatever else that task's
forward pass happened to produce -- the zero-filled adjoint, the sampling
mask, the realised noise level. `scripts/dump_eval.py` wants those for the
viewer; the metric sweep does not and passes nothing, so the two share one
forward pass instead of the dump reimplementing it and drifting. An adapter
that has nothing to offer may ignore the argument.

`sigma` is PINNED by the caller rather than sampled per batch. Evaluation wants
a fixed operating point (or a sweep of them); sampling would average over the
training noise distribution and hide the shape of the degradation.

Each adapter mirrors the corresponding validation loop in `training/` -- that
is deliberate, and the reason to change them together: if this drifts from
`training/recon.py`'s val block, the sweep stops measuring what training
reported.
"""

from __future__ import annotations

import torch

from operators import FFT2D, Identity, Mask, Sense
from operators.fourier import ifftc
from operators.noise import awgn
from physics.mask import get_mask_cached as get_mask, resolve_acs_lines
from training.common import embed_for_net, prepare_measurement


def _call_net(net, x, E, sigma):
    """Uniform model call. Unet variants take real-valued stacked input."""
    if net.__class__.__name__ in ("Unet", "NormUnet"):
        out = net(torch.view_as_real(x))
        return torch.view_as_complex(out.contiguous())
    out = net(x, E=E, sigma=sigma)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
def recon(net, batch, cfg, device, sigma, generator=None, extras=None, call=None):
    """CS-MRI reconstruction. Mirrors the val block of `training/recon.py`.

    `call(net, y, E, sigma) -> image` replaces the plain forward -- how
    scripts/level_figure.py runs `forward_levels` on exactly this measurement
    and operator. It receives the EMBEDDED operator; the result is truncated
    back here as usual.

    Returns `(gt, recon, organ_mask)`. The mask is None unless the config asks
    for it, so the sweep restricts its metrics to exactly the region the run
    was trained and validated on -- a sweep that scored the whole image on a
    mask-trained run would not be summarising the curves it sits next to.
    """
    # A paired (guided) loader adds the side scan: (..., side) or (..., side k, side maps).
    from training.recon import _unpack
    kspace, smaps, image, organ_mask, pad_hw, side = _unpack(batch)
    kspace, smaps, image, organ_mask, pad_hw = (
        b.to(device, non_blocking=True) for b in (kspace, smaps, image, organ_mask, pad_hw))
    mri = cfg["mri"]
    if not cfg.get("training", {}).get("use_organ_mask", False):
        organ_mask = None

    mask = get_mask(image, R=mri["R"], acs_lines=mri["acs_lines"],
                    mode=mri.get("mask_dist", "uniform"),
                    offset=mri.get("mask_offset", 0),
                    center_frac=mri.get("center_frac"),
                    adjust_accel=mri.get("adjust_accel", False))

    y, sigma_n, extra = prepare_measurement(
        image=image, kspace=kspace, mask=mask, smaps=smaps,
        # The VALIDATION measurement, as training validates: the run's
        # `training.val_kspace_type` if it has one (an eval script's
        # --val-kspace-type writes it there), else the training kspace_type.
        kspace_type=val_kspace_type(cfg),
        noise_std=sigma,                        # a number: pinned, not sampled
        noise_dist=cfg["training"].get("noise_dist", "uniform"),
        whiten_kspace=mri.get("whiten_kspace", False),
        # Must mirror training: a run trained on maps estimated from its own
        # ACS, evaluated against the dataset's precomputed ones, is a different
        # forward model at test time and the sweep would not summarise its
        # curves.
        online_smaps=mri.get("online_smaps"),
        online_smaps_kws=dict(mri.get("online_smaps_kws") or {},
                              acs_lines=resolve_acs_lines(
                                  image.shape[-1], mri["acs_lines"],
                                  mri.get("center_frac"))),
        need_smaps=getattr(net, "uses_operator_smaps", True),
        generator=generator,
    )

    E = Mask(mask) @ FFT2D() @ Sense(extra["smaps"])
    # Same image-domain embedding the training loop uses -- literally the same
    # function. `pad_hw` is the loader's decision and has to be honoured:
    # re-deriving the embedded size from `pad_stride` alone would build a
    # different operator here than the val loop built, and the sweep would stop
    # reproducing the curves it is meant to summarise.
    E, T = embed_for_net(net, E, image, pad_hw)
    # The guide, exactly as training/recon.py's val block builds it: the side scan undersampled
    # per guide_R at the validation line offset, in the domain the net takes it, under the
    # operator's maps. None for an unguided net (and for unpaired data).
    guide = None
    if side is not None:
        from training.recon import _check_paired, _guide
        _check_paired(net, val_kspace_type(cfg), mri.get("online_smaps"))
        guide = _guide(net, side, extra["smaps"], T, device, mri.get("guide_R") or 1,
                       mri.get("guide_acs_lines", 0), mri.get("guide_mask_dist", "uniform"),
                       guide_offset=mri.get("guide_val_offset") or 0)
    if guide is None:
        out = T.forward((call or _call_net)(net, y, E, sigma_n))
    else:
        if call is not None:
            raise ValueError("call= replaces an unguided forward; this net takes a guide")
        from training.recon import _forward
        out = _forward(net, y, E, sigma_n, guide)
        out = T.forward(out[0] if isinstance(out, tuple) else out)

    if mri.get("whiten_kspace", False) and "Zinv" in extra:
        out = extra["Zinv"] * out

    if extras is not None:
        # The zero-filled adjoint is the artifact the network had to remove, so
        # a viewer showing recon without it cannot answer "removed, or moved?".
        zf = T.forward(E.adjoint(y))
        if mri.get("whiten_kspace", False) and "Zinv" in extra:
            zf = extra["Zinv"] * zf
        extras.update(zero_filled=zf, sampling_mask=mask, sigma_n=sigma_n)
        if side is not None:
            # The side scan for the viewer, FULLY SAMPLED and coil-combined under the operator's
            # maps (the centre slice when the net takes several) -- the same image for every net
            # in a comparison, guided or not, whatever domain a net took its own guide in. Coil
            # RSS when the operator holds placeholder maps (a net estimating its own).
            s = side["kspace"][:, side["kspace"].shape[1] // 2] if isinstance(side, dict) else side
            c = ifftc(s.to(device, non_blocking=True))
            extras["guide"] = (Sense(extra["smaps"]).adjoint(c)
                               if getattr(net, "uses_operator_smaps", True)
                               else c.abs().pow(2).sum(dim=1, keepdim=True).sqrt())

        # Optional Tikhonov CG-SENSE, (E^H E + lam I) x = E^H y, on the SAME
        # measurement and operator the net saw: the linear recon's leftover
        # aliasing, for comparison with the net's (scripts/dump_eval.py
        # --cg-sense-lam). Asked for by putting `cg_sense_lam` in `extras`.
        # The maps are the run's: the online estimate (`online_smaps`, e.g.
        # ESPIRiT from the measurement's ACS) when it has one. A net that
        # estimates its own maps gets flat placeholders in E, so no CG-SENSE.
        lam = extras.get("cg_sense_lam")
        if lam is not None and not getattr(net, "uses_operator_smaps", True):
            print(f"[cg_sense] skipped: {type(net).__name__} does not use the "
                  f"operator's coil maps, so E holds placeholders")
            lam = None
        if lam is not None:
            from solvers.cg import batched_cg
            lam = float(lam)

            def solve(Eop):
                with torch.no_grad():
                    return batched_cg(lambda v: Eop.adjoint(Eop(v)) + lam * v, Eop.adjoint(y),
                                      tol=float(extras.get("cg_sense_tol", 1e-6)),
                                      max_iter=int(extras.get("cg_sense_iters", 100)))

            sense = T.forward(solve(E))
            if mri.get("whiten_kspace", False) and "Zinv" in extra:
                sense = extra["Zinv"] * sense
            extras["cg_sense"] = sense
            # `cg_sense_stored_maps` (flag) -> `cg_sense_stored`: the same solve on the same
            # y, but through the STORED maps
            # (the ones the reference S^H c is made with) instead of the run's online
            # estimate. Where the two disagree, aliasing left by the operator-map solve but
            # not by this one is coil-map mismatch, which no prior removes
            # (scripts/cg_sense_check.py). Unwhitened only: the stored maps have no Zinv.
            if extras.get("cg_sense_stored_maps") and mri.get("online_smaps"):
                if mri.get("whiten_kspace", False):
                    print("[cg_sense] stored-map solve skipped: whitened k-space")
                else:
                    E_s, T_s = embed_for_net(net, Mask(mask) @ FFT2D() @ Sense(smaps),
                                             image, pad_hw)
                    extras["cg_sense_stored"] = T_s.forward(solve(E_s))

    return image, out, organ_mask


def denoiser(net, batch, cfg, device, sigma, generator=None, extras=None):
    """Gaussian denoising. Mirrors the val block of `training/denoiser.py`.

    A loader may hand back either a clean image (noise added here) or a
    pre-noised `(noisy, gt, sigma)` triple; the second form's own sigma wins,
    because the pair was generated together and re-noising would be wrong.
    """
    if isinstance(batch, (list, tuple)):
        noisy, gt, sigma_n = (b.to(device, non_blocking=True) for b in batch)
    else:
        gt = batch.to(device, non_blocking=True)
        noisy, sigma_n = awgn(gt, [sigma, sigma],
                              dist=cfg["training"].get("noise_dist", "uniform"))

    if extras is not None:
        # `zero_filled` is the degraded input under any task; for denoising
        # that is the noisy image, so the viewer's "input" column means the
        # same thing across tasks without knowing which one it is looking at.
        extras.update(zero_filled=noisy, sigma_n=sigma_n)

    # No organ mask: denoising has no sensitivity support to restrict to.
    return gt, _call_net(net, noisy, Identity(), sigma_n), None


_BRIDGES = {}


def _bridge_for(cfg, net, device):
    """The run's bridge schedule, built once per (run config, device)."""
    key = (id(cfg), str(device))
    if key not in _BRIDGES:
        from sb.base import build_schedule
        i2 = cfg["i2sb"]
        b = build_schedule(kind=i2.get("kind", "brownian"), tau=i2["tau"],
                           n_points=i2.get("n_points", 1000), beta_max=i2.get("beta_max", 0.3),
                           device=device)
        if hasattr(net, "assert_schedule_matches"):
            net.assert_schedule_matches(b)
        _BRIDGES.clear()                              # one run at a time
        _BRIDGES[key] = b
    return _BRIDGES[key]


def i2sb_recon(net, batch, cfg, device, sigma, generator=None, extras=None):
    """I2SB(-Recon) reconstruction: the FULL reverse-bridge sampler on one batch. Mirrors
    training/i2sb_recon.py's full_recon validation -- the same `_measure` (measurement, bridge
    start x1, the net's guide), the same sampler, CG data-consistency and support settings.

    Sampler settings: `cfg["eval"]` (scripts/dump_eval.py --nfe / --dc / --inversion) overrides,
    else the run's own `cfg["i2sb"]`: val_nfe NFE; dc_iters / dc_lam the CG step on every x0
    prediction (I2SB-Recon, arXiv 2411.14269; 0 = plain I2SB); support_thresh / support_state;
    "inversion": I2SB-Inversion's recon -> inversion -> recon (sb/i2sb_recon.py).

    The bridge noise comes from torch's global RNG; it is seeded per slice from `generator`
    (dump_eval resets that per run), so every run sees the same draws on the same slice. With
    CG on and `extras`, the plain sampler is run as well on the SAME draws and returned in
    extras["alt_recons"]["nodc"] (dump_eval stores it as `recon_nodc`)."""
    from training.i2sb_recon import _measure, _unpack, measurement_support
    from sb.i2sb_recon import (DC_LAM, bridge_noise_scale, i2sb_inversion_sample,
                               i2sb_recon_sample)
    from operators.accessors import get_mask as op_mask, get_sensitivity_map

    i2, mri, tr = cfg["i2sb"], cfg["mri"], cfg.get("training", {})
    ev = cfg.get("eval", {}) or {}
    bridge = _bridge_for(cfg, net, device)
    meas = dict(R=mri["R"], acs_lines=mri.get("acs_lines", 24),
                mask_dist=mri.get("mask_dist", "uniform"), mask_offset=mri.get("mask_offset", 0),
                center_frac=mri.get("center_frac"), adjust_accel=mri.get("adjust_accel", False),
                noise_dist=tr.get("noise_dist", "uniform"), online_smaps=mri.get("online_smaps"),
                online_smaps_kws=mri.get("online_smaps_kws"),
                x1_source=i2.get("x1_source", "adjoint"),
                x0_source=i2.get("x0_source", "operator"),
                guide_R=mri.get("guide_R", 1), guide_acs_lines=mri.get("guide_acs_lines", 0),
                guide_mask_dist=mri.get("guide_mask_dist", "uniform"),
                guide_maps=i2.get("guide_maps", "operator"))
    _, x1, y, E, sigma_n, organ_mask, gt, guide = _measure(
        batch, net, device, kspace_type=val_kspace_type(cfg), noise_std=sigma,
        generator=generator, guide_offset=0, **meas)
    if not tr.get("use_organ_mask", False):
        organ_mask = None

    nfe = int(ev.get("nfe") or i2.get("val_nfe", 20))
    dc_iters = int(ev["dc_iters"] if "dc_iters" in ev else i2.get("dc_iters", 0))
    dc_lam = float(ev["dc_lam"] if "dc_lam" in ev else i2.get("dc_lam", DC_LAM))
    thr = ev["support_thresh"] if "support_thresh" in ev else i2.get("support_thresh")
    sup = None if thr is None else measurement_support(y, E, meas, thr)
    kappa = bridge_noise_scale(x1, i2.get("tau_scale", "none"), i2.get("tau_ref"))
    seed = 0 if generator is None else int(torch.randint(
        0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device).item())
    devs = [device] if torch.device(device).type == "cuda" else []

    def sample(iters):
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(seed)                   # the same bridge noise for every setting
            if ev.get("inversion"):
                return i2sb_inversion_sample(net, y, E, bridge, x1, sigma_n=sigma_n, nfe=nfe,
                                             dc_iters=iters, dc_lam=dc_lam, guide=guide,
                                             noise_scale=kappa,
                                             posterior=i2.get("posterior", "ddpm"))[0]
            return i2sb_recon_sample(net, y, E, bridge, sigma_n=sigma_n, x1=x1, nfe=nfe,
                                     deterministic=bool(i2.get("deterministic", False)),
                                     posterior=i2.get("posterior", "ddpm"), verbose=False,
                                     guide=guide, noise_scale=kappa, dc_iters=iters,
                                     dc_lam=dc_lam, support=sup,
                                     support_state=bool(ev.get("support_state",
                                                               i2.get("support_state", False))))[0]

    rec = sample(dc_iters)

    if extras is not None:
        extras.update(zero_filled=E.adjoint(y), sampling_mask=op_mask(E), sigma_n=sigma_n,
                      bridge_start=x1)
        if dc_iters > 0 and ev.get("also_nodc", True):
            extras.setdefault("alt_recons", {})["nodc"] = sample(0)
        side = _unpack(batch)[5]
        if side is not None:
            # as recon(): the side scan FULLY sampled, coil-combined under the operator's maps
            s = side["kspace"][:, side["kspace"].shape[1] // 2] if isinstance(side, dict) else side
            extras["guide"] = Sense(get_sensitivity_map(E)).adjoint(
                ifftc(s.to(device, non_blocking=True)))
    return gt, rec, organ_mask


REGISTRY = {
    "recon": recon,
    "denoiser": denoiser,
    "i2sb_recon": i2sb_recon,
}


def build_adapter(task):
    """An adapter maps `(net, batch, cfg, device, sigma, gen, extras=None)` to
    `(gt, recon, organ_mask_or_None)`."""
    if task not in REGISTRY:
        raise ValueError(
            f"no evaluation adapter for task {task!r}; registered: "
            f"{sorted(REGISTRY)}. Add one to evaluation/tasks.py.")
    return REGISTRY[task]


def metric_mask_source(cfg):
    """Which organ mask this run's metrics use: the val loader's
    `organ_mask_source` when the run applies the mask, else "none"."""
    if not cfg.get("training", {}).get("use_organ_mask", False):
        return "none"
    return cfg.get("data", {}).get("val", {}).get("organ_mask_source", "smaps")


def force_organ_mask(cfg):
    """Score this run inside the organ mask from its STORED coil maps, whatever
    it was trained with (2026-10-20): `use_organ_mask` on, and the val loader's
    `organ_mask_source = "smaps"` -- the loader's own `smaps.abs().sum(0) > 0`,
    the support the stored SENSE ground truth is nonzero on, so it costs
    nothing to compute (datasets/fastmri/loader.py). Every net in a table is
    then judged on the same pixels, and the background -- which only some nets
    were trained to leave empty -- is out of every metric. Modifies `cfg`."""
    cfg.setdefault("training", {})["use_organ_mask"] = True
    cfg.setdefault("data", {}).setdefault("val", {})["organ_mask_source"] = "smaps"
    return cfg


#: What every evaluation script measures on unless told otherwise (their
#: --val-kspace-type default): MEASURED k-space + AWGN, how the nets train and
#: how results/main has always been scored. Set explicitly rather than read from
#: the run, because configs generated 2026-10-20 .. 2026-10-22 carry
#: `training.val_kspace_type: "simulated"` and would otherwise be scored on the
#: easier measurement. "simulated" stays available per call (noise sweeps).
EVAL_KSPACE_TYPE = "measurement_awgn"


def val_kspace_type(cfg):
    """How a validation / evaluation measurement is made for this run.

    `training.val_kspace_type` when set -- the eval scripts write their
    --val-kspace-type there (default EVAL_KSPACE_TYPE); configs generated
    2026-10-20 .. 2026-10-22 carry "simulated" -- otherwise the run's training
    `kspace_type`, which is what training validation follows since 2026-10-22.
    """
    return (cfg.get("training", {}).get("val_kspace_type")
            or cfg.get("mri", {}).get("kspace_type", "simulated"))


def default_sigma(cfg):
    """The noise level to evaluate at when none is given.

    `val_noise_std` if the config pins one -- that is what training validated
    against, so the sweep reproduces the number on the val curve. Otherwise the
    midpoint of the training range, which is what a pinned validation would
    have used anyway.
    """
    t = cfg.get("training", {})
    if t.get("val_noise_std") is not None:
        return float(t["val_noise_std"])
    std = t.get("noise_std", 0.0)
    if isinstance(std, (list, tuple)):
        return float(sum(std) / len(std))
    return float(std)
