"""
Metric registry for the evaluation sweep.

Every metric here takes `(gt, recon)` and returns ONE VALUE PER SAMPLE, shape
`(B,)`. That is the difference from `training.metrics.compute_metrics`, which
reduces over the whole tensor including the batch axis: at `batch_size=1` the
two agree exactly, but above it `compute_metrics` returns a batch-pooled number
whose value depends on the batch size. A sweep has to be able to change the
batch size without moving the numbers, and it needs per-sample values to report
a spread rather than just a mean.

The underlying definitions are `training.metrics`' own, called per sample, so
the means this produces are directly comparable to what training logged.

Add a metric by writing `fn(gt, recon) -> (B,)` and putting it in `REGISTRY`.
`higher_is_better` is carried so a summary table can be read without knowing
each metric's convention.
"""

from __future__ import annotations

import torch

from training.metrics import nrmse as _nrmse
from training.metrics import psnr as _psnr
from training.metrics import ssim as _ssim


# ---------------------------------------------------------------------------
def _mag(x):
    """Magnitude image.

    Every metric here is computed on magnitudes, which is what
    `training/recon.py` does (`compute_metrics(image.abs(), recon.abs())`) and
    therefore what makes these numbers comparable to the logged val curves.
    It is also load-bearing, not cosmetic: `training.metrics.nrmse` calls
    `.max()` on the stacked pair, which raises on a complex tensor, and
    `psnr` on complex inputs would fold phase error into the value -- a
    different quantity from the one training reports.
    """
    return x.abs() if torch.is_complex(x) else x


def _per_sample(fn, gt, recon):
    """Apply a batch-reducing metric one sample at a time."""
    gt, recon = _mag(gt), _mag(recon)
    return torch.stack([fn(gt[i:i + 1], recon[i:i + 1]).reshape(())
                        for i in range(gt.shape[0])])


def psnr(gt, recon):
    return _per_sample(_psnr, gt, recon)


def nrmse(gt, recon):
    return _per_sample(_nrmse, gt, recon)


def ssim(gt, recon):
    out = _ssim(_mag(gt), _mag(recon))         # already per-sample
    return out.reshape(out.shape[0]) if out.dim() else out.reshape(1)


# ---------------------------------------------------------------------------
#  LPIPS
# ---------------------------------------------------------------------------
_LPIPS_CACHE = {}


def _lpips_model(net, device):
    key = (net, str(device))
    if key not in _LPIPS_CACHE:
        try:
            import lpips as _lpips_pkg
        except ImportError as e:               # noqa: BLE001
            raise RuntimeError(
                "the 'lpips' metric needs the lpips package "
                f"(pip install lpips): {e}") from e
        try:
            _LPIPS_CACHE[key] = _lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
        except Exception as e:                 # noqa: BLE001
            raise RuntimeError(
                f"could not build LPIPS(net={net!r}). It downloads pretrained "
                f"weights on first use, which fails on an offline node -- warm "
                f"the cache on a login node, or drop 'lpips' from the metric "
                f"list. Original error: {e}") from e
    return _LPIPS_CACHE[key]


def _to_lpips_input(x, lo, hi):
    """magnitude -> [-1, 1] -> 3 channels, which is what LPIPS expects."""
    x = x.abs().float()
    if x.shape[1] != 1:                        # collapse any channel axis first
        x = x.mean(dim=1, keepdim=True)
    x = (x - lo) / (hi - lo + 1e-12)
    return (2.0 * x - 1.0).clamp(-1.0, 1.0).repeat(1, 3, 1, 1)


def lpips(gt, recon, net="alex"):
    """Perceptual distance on JOINTLY min-max normalised magnitude images.

    LPIPS is not scale invariant, so the normalisation is part of the metric's
    definition, not a detail. Normalising gt and recon TOGETHER (per sample, by
    their shared min/max) keeps the pair on one intensity scale -- normalising
    each separately would hide a global brightness error, which for
    reconstruction is a real error.

    Lower is better. It is a distance, not a similarity.
    """
    model = _lpips_model(net, gt.device)
    out = []
    for i in range(gt.shape[0]):
        g, r = gt[i:i + 1].abs().float(), recon[i:i + 1].abs().float()
        lo = torch.minimum(g.min(), r.min())
        hi = torch.maximum(g.max(), r.max())
        with torch.no_grad():
            d = model(_to_lpips_input(g, lo, hi), _to_lpips_input(r, lo, hi))
        out.append(d.reshape(()))
    return torch.stack(out)


# ---------------------------------------------------------------------------
REGISTRY = {
    "psnr":  dict(fn=psnr,  higher_is_better=True,  fmt="{:.2f}"),
    "ssim":  dict(fn=ssim,  higher_is_better=True,  fmt="{:.4f}"),
    "nrmse": dict(fn=nrmse, higher_is_better=False, fmt="{:.4f}"),
    "lpips": dict(fn=lpips, higher_is_better=False, fmt="{:.4f}"),
}

DEFAULT_METRICS = ["psnr", "ssim", "nrmse"]     # lpips is opt-in: it downloads weights


def build_metrics(names):
    """Resolve metric names to callables, failing loudly on a typo."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(
            f"unknown metric(s) {unknown}; registered: {sorted(REGISTRY)}")
    return {n: REGISTRY[n]["fn"] for n in names}


def warm_up(names, device):
    """Build anything expensive BEFORE the sweep starts.

    LPIPS fetches pretrained weights on first use. Doing that inside the loop
    means a run that is 40 minutes in dies on a network error; doing it here
    means it dies in the first second, with a message that says what to do.
    """
    if "lpips" in names:
        _lpips_model("alex", device)
