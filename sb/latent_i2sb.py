"""
sb/latent_i2sb.py -- latent-domain I2SB sampling. Two approaches, selected by `bridge_domain`
(see training/latent_i2sb.py); both regress the T1ce latent z0 and decode via the frozen D_t1ce:

    latent_i2sb_sample            bridge_domain="latent" (True Latent I2SB): the bridge runs ENTIRELY
                                  in the M-channel latent. The prior latent z1 = encode(D_joint, cond)
                                  is the joint encoding of the CONDITIONING contrasts (a cond-only dict,
                                  C = n_cond, e.g. [FLAIR, T1, T2]; NO x_t channel) and is BOTH the
                                  bridge start and R's conditioning (R.C = 2M, cat[z_t, z1]). The
                                  attention-heavy encode runs ONCE; only the final latent is decoded.
    latent_i2sb_sample_imgdomain  bridge_domain="image" (Latent Regression, Image-Domain Bridge): the
                                  bridge state is a T1ce IMAGE x_t. Each step the joint dict (C = 1 +
                                  n_cond, i.e. [x_t, cond]) encodes [x_t, cond] at sigma(t) -> z_t, and
                                  R regresses z0 UNCONDITIONALLY (R.C = M -- the conditioning is baked
                                  into z_t, so R needs no extra channels).

Both reuse sb.base.reverse_sample (space-agnostic); only the x0-predictor differs. The encode /
decode / regress helpers are shared with training/latent_i2sb.py.
"""

import torch

from operators import Identity
from sb.base import reverse_sample, forward_std


# ---------------------------------------------------------------------------
# the latent composition  (encode -> regress -> decode), all frozen except R
# ---------------------------------------------------------------------------
def encode(D, img, sigma=0.0):
    """Frozen encode of a (clean) image into D's M-channel sparse code z. The clean endpoints are
    encoded at sigma=0 (the bridge stochasticity is injected later, in latent space)."""
    H, W = img.shape[-2:]
    if H % D.sc or W % D.sc:
        raise ValueError(
            f"Input {H}x{W} must be divisible by the dictionary stride sc={D.sc} (the decode calls "
            f"D_t1ce.D directly, bypassing GroupCDL.forward's internal pad/crop). Set crop_size / "
            f"center_crop to a multiple of {D.sc}.")
    with torch.no_grad():
        _, z = D(img, E=Identity(), sigma=sigma)
    return z.detach()


def decode(D_t1ce, z0_hat, dc=None):
    """Frozen decode of a predicted code back to the T1ce image via the synthesis dictionary
    D = B[0] (a transposed convolution). Gradients flow through this fixed operator into R."""
    x = D_t1ce.D(z0_hat)
    return x if dc is None else x + dc


def _decode_dc(x1, decode_dc):
    """DC term re-added at decode. 'x1_mean' = per-image mean of the prior (available at train AND
    inference); 'none' = 0. Mirrors GroupCDL's own mean(dim=(1,2,3)) convention."""
    if decode_dc in (None, "none"):
        return None
    if decode_dc == "x1_mean":
        return x1.mean(dim=(1, 2, 3), keepdim=True)
    raise ValueError(f"unknown decode_dc {decode_dc!r} (use 'x1_mean' or 'none')")


def latent_regress(R, zt, cond_lat, sigma, M):
    """R's latent -> latent step E[z0 | zt (, z1)]. cond_lat is z1 (conditional) or None. We keep
    the first M channels of the reconstruction as the z0 estimate (the zt slot)."""
    net_in = zt if cond_lat is None else torch.cat([zt, cond_lat], dim=1)
    recon, _ = R(net_in, E=Identity(), sigma=sigma)
    return recon[:, :M]


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
@torch.no_grad()
def latent_i2sb_sample(D_joint, R, D_t1ce, x1, cond, sched, *, M=None, nfe=20,
                       deterministic=False, posterior="ddpm", decode_dc="x1_mean",
                       conditional=None, verbose=False):
    """bridge_domain="latent" (True Latent I2SB): reverse the bridge z1 -> z0 ENTIRELY in the latent,
    then decode ONCE to the T1ce image. z1 = encode(D_joint, cond) (the cond-only joint dict) is the
    bridge start AND R's conditioning (R.C = 2M). `conditional` is accepted for back-compat and
    ignored -- this mode is always conditional. Returns (B, 1, H, W)."""
    device = sched.std_fwd.device
    M = M or D_joint.M
    x1, cond = x1.to(device), cond.to(device)
    z1 = encode(D_joint, cond)                                  # prior latent = bridge start (once)
    dc = _decode_dc(x1, decode_dc)

    def pred_x0_fn(z_t, step):
        step_t = torch.full((z_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=z_t.shape[1:])
        return latent_regress(R, z_t, z1, sigma, M)             # R conditions on z1: cat[z_t, z1]

    z0_hat, _, _ = reverse_sample(sched, pred_x0_fn, z1, nfe=nfe, deterministic=deterministic,
                                  posterior=posterior, log_count=1, verbose=verbose)
    return decode(D_t1ce, z0_hat, dc=dc)


@torch.no_grad()
def latent_i2sb_sample_imgdomain(D_joint, R, D_t1ce, x1, cond, sched, *, M=None, nfe=20,
                                 deterministic=False, posterior="ddpm", decode_dc="x1_mean",
                                 conditional=None, verbose=False):
    """bridge_domain="image" (Latent Regression, Image-Domain Bridge): the bridge state is a T1ce
    IMAGE x_t and the reverse posterior runs there. Each step the 4-channel joint dict encodes
    [x_t, cond] at sigma(t) -> z_t, R regresses z0 UNCONDITIONALLY (R.C = M), and D_t1ce decodes to
    the image. `conditional` accepted for back-compat and ignored. Returns (B, 1, H, W)."""
    device = sched.std_fwd.device
    M = M or D_joint.M
    x1, cond = x1.to(device), cond.to(device)
    dc = _decode_dc(x1, decode_dc)

    def pred_x0_fn(x_t, step):
        step_t = torch.full((x_t.shape[0],), step, device=device, dtype=torch.long)
        sigma = forward_std(sched, step_t, xdim=x_t.shape[1:])
        z_t = encode(D_joint, torch.cat([x_t, cond], dim=1), sigma=sigma)   # 4-ch joint encode of the image state
        z0_hat = latent_regress(R, z_t, None, sigma, M)                     # R unconditional (cond baked into z_t)
        return decode(D_t1ce, z0_hat, dc=dc)

    recon, _, _ = reverse_sample(sched, pred_x0_fn, x1, nfe=nfe, deterministic=deterministic,
                                 posterior=posterior, log_count=1, verbose=verbose)
    return recon
