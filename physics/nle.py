"""
PyTorch translation of Julia noise-level estimation and multi-coil whitening utilities.

Tensor layout convention: standard PyTorch  (B, C, H, W)
    B – batch, C – coils/channels, H – height, W – width

CDF 9/7 wavelet coefficients are used as a default high-pass filter for
noise estimation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from operators.fourier import fftc, ifftc
from torch import Tensor
from typing import Optional, Union, List
import os

# ---------------------------------------------------------------------------
# CDF 9/7 high-pass wavelet filter
# ---------------------------------------------------------------------------

cdf97 = torch.tensor(
    [
         0.091271763114,
        -0.057543526229,
        -0.591271763114,
         1.11508705,
        -0.591271763114,
        -0.057543526229,
         0.091271763114,
    ],
    dtype=torch.float64,
)

# Helper function to extract acs columns from a given mask 
def extract_acs_columns(mask: torch.Tensor):
    """
    Extract ACS region when mask has fully sampled columns.

    Parameters
    ----------
    mask : Tensor
        shape (H, W), (B, H, W), or (B, 1, H, W)
        binary mask with fully sampled columns

    Returns
    -------
    acs_mask : same shape as input
    bbox     : (w_start, w_end)
    """

    m = mask

    # reduce to 2D
    if m.dim() == 4:
        m2 = m[0, 0]
    elif m.dim() == 3:
        m2 = m[0]
    else:
        m2 = m

    H, W = m2.shape

    # column is fully sampled if ALL rows are 1
    full_cols = (m2.sum(dim=0) == H)   # shape (W,)

    idx = torch.where(full_cols)[0]

    if len(idx) == 0:
        raise ValueError("No fully sampled columns found.")

    # enforce contiguity (ACS should be central block)
    # find largest contiguous run
    diffs = torch.diff(idx)
    breaks = torch.where(diffs > 1)[0]

    # split into contiguous groups
    groups = []
    start = 0
    for b in breaks.tolist():
        groups.append(idx[start:b+1])
        start = b + 1
    groups.append(idx[start:])

    # pick the group closest to center
    center = W // 2
    def dist_to_center(g):
        return torch.abs(g.float().mean() - center)

    acs_group = min(groups, key=dist_to_center)

    w_start = acs_group[0].item()
    w_end   = acs_group[-1].item() + 1

    # build ACS mask
    acs = torch.zeros_like(m2)
    acs[:, w_start:w_end] = 1

    # restore shape
    if mask.dim() == 4:
        acs = acs.unsqueeze(0).unsqueeze(0).expand_as(mask)
    elif mask.dim() == 3:
        acs = acs.unsqueeze(0).expand_as(mask)

    return acs, (w_start, w_end)
    
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _real_dtype(x: Tensor) -> torch.dtype:
    if x.is_complex():
        return torch.float32 if x.dtype == torch.complex64 else torch.float64
    return x.dtype


def _eps(x: Tensor) -> float:
    return torch.finfo(_real_dtype(x)).eps


def _default_filter(x: Tensor) -> Tensor:
    return cdf97.to(device=x.device, dtype=_real_dtype(x))


def _separable_conv(x: Tensor, f: Tensor) -> Tensor:
    """
    Depthwise separable 2-D convolution with 1-D filter f.

    Parameters
    ----------
    x : (B, C, H, W) – real or complex
    f : (k,)

    Returns
    -------
    Tensor, same shape as x.
    """
    B, C, H, W = x.shape
    k   = f.shape[0]
    pad = k // 2

    # Merge batch+channel for grouped depthwise conv
    x_pt = x.reshape(1, B * C, H, W)

    f_ = f.to(dtype=_real_dtype(x))
    fh = f_.reshape(1, 1, k, 1).expand(B * C, 1, k, 1).contiguous()
    fw = f_.reshape(1, 1, 1, k).expand(B * C, 1, 1, k).contiguous()

    def _conv_real(t: Tensor) -> Tensor:
        t = F.conv2d(t, fh, padding=(pad, 0), groups=B * C)
        t = F.conv2d(t, fw, padding=(0, pad), groups=B * C)
        return t[..., :H, :W]

    if x.is_complex():
        out = torch.complex(_conv_real(x_pt.real), _conv_real(x_pt.imag))
    else:
        out = _conv_real(x_pt)

    return out.reshape(B, C, H, W)


def _mul_channel(M: Tensor, t: Tensor) -> Tensor:
    """
    Apply a per-batch (C x C) matrix to the channel dimension of t.

    Parameters
    ----------
    M : (B, C, C)
    t : (B, C, H, W)

    Returns
    -------
    Tensor, shape (B, C, H, W)
    """
    B, C, H, W = t.shape
    t_flat = t.reshape(B, C, H * W)          # (B, C, H*W)
    out    = torch.bmm(M, t_flat)            # (B, C, H*W)
    return out.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Noise-Level Estimation (MAD)
# ---------------------------------------------------------------------------

def nle_mad(x: Tensor, f: Optional[Tensor] = None) -> Tensor:
    """
    Median Absolute Deviation noise-level estimator.

    Parameters
    ----------
    x : (B, C, H, W) – real or complex multicoil signal
    f : 1-D filter, defaults to cdf97

    Returns
    -------
    Scalar Tensor – estimated noise standard deviation.
    For complex input, returns the std-dev of a circularly-symmetric
    complex normal (scaled by sqrt(2)).
    """
    if f is None:
        f = _default_filter(x)

    if x.is_complex():
        y = torch.cat([x.real, x.imag], dim=1)   # (B, 2C, H, W)
        return nle_mad(y, f) * (2.0 ** 0.5)

    z = _separable_conv(x, f)
    # Divide by 2: 1-D filter applied twice (H and W)
    return torch.median(z.abs()) / (2.0 * 0.6745)


# ---------------------------------------------------------------------------
# Noise Covariance Estimation
# ---------------------------------------------------------------------------

def ncov_est(x: Tensor, f: Optional[Tensor] = None) -> Tensor:
    """
    Estimate the inter-channel noise covariance matrix.

    Parameters
    ----------
    x : (B, C, H, W)
    f : 1-D filter, defaults to cdf97

    Returns
    -------
    Tensor, shape (B, C, C)
    """
    if f is None:
        f = _default_filter(x)

    B, C, H, W = x.shape

    # Filter each channel independently via depthwise conv
    z = _separable_conv(x, f) / 2.0          # (B, C, H, W)

    # Do a mean subtracted covariance estimation
    Z = z.reshape(B, C, H * W)
    mu = Z.mean(dim=-1, keepdim=True)     # (B, C, 1)
    Zc = Z - mu                           # centered
    Sigma = torch.bmm(Zc, Zc.conj().transpose(-2, -1))

    # Sigma[b] = Z[b] @ Z[b]^H / N_mask      shape (B, C, C)
    Sigma = torch.bmm(Z, Z.conj().transpose(-2, -1))   # (B, C, C)

    # Normalise by number of non-zero pixels per batch element
    mask   = (x.abs().pow(2).sum(dim=1) > 0)           # (B, H, W)
    N_mask = mask.sum(dim=(1, 2)).reshape(B, 1, 1).to(Sigma.dtype)

    return Sigma / N_mask               # (B, C, C)


def ncov_est_undersampled(x: Tensor, R:int,  mask:Tensor, f: Optional[Tensor] = None) -> Tensor:
    """
    Estimate the inter-channel noise covariance matrix.

    Parameters
    ----------
    x : (B, C, H, W)
    f : 1-D filter, defaults to cdf97

    Returns
    -------
    Tensor, shape (B, C, C)
    """
    if f is None:
        f = _default_filter(x)

    B, C, H, W = x.shape
    
    # Filter each channel independently via depthwise conv
    z = _separable_conv(x, f) / 2.0          # (B, C, H, W)

    # Send to kspace
    z = fftc(z) 

    # Compute NLE on masked kspace
    z = z[:, :, torch.squeeze(mask).bool()]

    # Flatten spatial dims: Z[b, c, n] = filtered pixel n of coil c in batch b
    N_mask = torch.sum(mask)
    Z = z.reshape(B, C, int(N_mask.item()))               # (B, C, N)

    # Sigma[b] = Z[b] @ Z[b]^H / N_mask      shape (B, C, C)
    Sigma = torch.bmm(Z, Z.conj().transpose(-2, -1))   # (B, C, C)
    # Normalise by number of non-zero pixels per batch element
    mask   = (x.abs().pow(2).sum(dim=1) > 0)           # (B, H, W)
    N_mask = mask.sum(dim=(1, 2)).reshape(B, 1, 1).to(Sigma.dtype)
    
    # Multiply by a compensating factor of acceleration
    return Sigma / N_mask * R              # (B, C, C)

# ---------------------------------------------------------------------------
# Covariance Matrix Square Root
# ---------------------------------------------------------------------------

def sqrt_covmat(Sigma: Tensor) -> Tensor:
    """
    Matrix square root of a PSD matrix (or batch thereof).
    Sigma = U S U^H  =>  sqrt(Sigma) = U sqrt(S) U^H

    Parameters
    ----------
    Sigma : (C, C) or (B, C, C)
    """
    if Sigma.dim() == 2:
        U, s, _ = torch.linalg.svd(Sigma)
        return U @ (s.sqrt().unsqueeze(0) * U.conj().T)

    # Batched (B, C, C)
    U, s, _ = torch.linalg.svd(Sigma)                  # U:(B,C,C), s:(B,C)
    return U @ (s.sqrt().unsqueeze(1) * U.conj().transpose(-2, -1))


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------

def _inv_sqrt_sigma(U: Tensor, s: Tensor, t: Tensor, eps: float) -> Tensor:
    """
    Apply Sigma^{-1/2} = U diag(1/sqrt(s)) U^H to t along the channel dim.

    Parameters
    ----------
    U : (B, C, C)
    s : (B, C)    singular values (real, >= 0)
    t : (B, C, H, W)
    """
    t1         = _mul_channel(U.conj().transpose(-2, -1), t)    # U^H t
    inv_sqrt_s = 1.0 / s.real.clamp(min=eps).sqrt()             # (B, C)
    t2         = t1 * inv_sqrt_s.unsqueeze(-1).unsqueeze(-1)    # (B, C, H, W)
    return _mul_channel(U, t2)


def _coil_combine(smaps: Tensor, data: Tensor) -> Tensor:
    """Sensitivity-weighted coil combination.  (B, C, H, W) -> (B, 1, H, W)"""
    return (smaps.conj() * data).sum(dim=1, keepdim=True)


def whiten(
    x:     Union[Tensor, List[Tensor]],
    smaps: Optional[Tensor] = None,
    Sigma: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    R: Optional[int] = None,
):
    """
    Whiten multicoil image-domain data.

    Calling conventions
    -------------------
    1. whiten(x)  or  whiten(x, Sigma=Sigma)
       No sensitivity maps -> returns whitened Tensor directly.

    2. whiten(x, smaps)  or  whiten(x, smaps, Sigma)
       Single data tensor with sensitivity maps.

    3. whiten([x1, x2, ...], smaps)
       List of data tensors sharing the same sensitivity maps.

    Parameters
    ----------
    x     : Tensor (B, C, H, W)  or list thereof
    smaps : Tensor (B, C, H, W), optional
    Sigma : Tensor (B, C, C),    optional – estimated via ncov_est if omitted

    Returns
    -------
    List:
        "data"  – whitened data, Tensor or list of Tensors (B, C, H, W)
        "smaps" – whitened & normalised sensitivity maps   (B, C, H, W)
        "sigma" – per-pixel scale factor                   (B, 1, H, W)
        "zinv"  – inverse normalisation map                (B, 1, H, W)
    """
    eps = _eps(x[0] if isinstance(x, (list, tuple)) else x)

    # Case 1: no sensitivity maps
    if smaps is None:
        assert isinstance(x, Tensor), "smaps=None requires a single Tensor x"
        if Sigma is None:
            if mask is None:
                Sigma = ncov_est(x)
            else:
                Sigma = ncov_est_undersampled(x, R, mask)
        U, s, _ = torch.linalg.svd(Sigma)     # (B,C,C), (B,C)
        return _inv_sqrt_sigma(U, s, x, eps)

    # Case 2 & 3: sensitivity maps present
    xs: List[Tensor] = list(x) if isinstance(x, (list, tuple)) else [x]

    if Sigma is None:
        if mask is None:
            Sigma = ncov_est(x)
        else:
            Sigma = ncov_est_undersampled(x, R, mask)

    U, s, _ = torch.linalg.svd(Sigma)         # (B,C,C), (B,C)

    def sq_inv(t: Tensor) -> Tensor:
        return _inv_sqrt_sigma(U, s, t, eps)

    # Whiten data and smaps
    xs_w    = [sq_inv(xi) for xi in xs]
    smaps_w = sq_inv(smaps)

    # Normalise whitened smaps: ||smap_w||_coil -> 1
    z       = smaps_w.abs().pow(2).sum(dim=1, keepdim=True).sqrt()  # (B,1,H,W)
    smaps_w = smaps_w / (z + eps)

    # Re-scale whitened data to match dynamic range of original coil-combined image
    if len(xs) == 1:
        beta  = _coil_combine(smaps,   xs[0]  ).abs().amax(dim=(2, 3), keepdim=True)
        delta = _coil_combine(smaps_w, xs_w[0]).abs().amax(dim=(2, 3), keepdim=True)
    else:
        beta  = torch.stack(
            [_coil_combine(smaps,   xi  ).abs().amax(dim=(2, 3)) for xi in xs],     dim=0
        ).mean(0).unsqueeze(-1).unsqueeze(-1)
        delta = torch.stack(
            [_coil_combine(smaps_w, xi_w).abs().amax(dim=(2, 3)) for xi_w in xs_w], dim=0
        ).mean(0).unsqueeze(-1).unsqueeze(-1)

    sigma = beta / delta.clamp(min=eps)        # (B, 1, 1, 1)
    xs_w  = [xi_w * sigma for xi_w in xs_w]

    # Renormalisation maps
    zinv  = (z > 0).to(sigma.dtype) / (sigma * z + eps)   # (B, 1, H, W)
    sigma = (z > 0).to(sigma.dtype) * sigma               # (B, 1, H, W)

    result_data = xs_w[0] if not isinstance(x, (list, tuple)) else xs_w
    return result_data, smaps_w, sigma, zinv


def whiten_5d(
    y:     Tensor,
    smaps: Tensor,
    Sigma: Optional[Tensor] = None,
):
    """
    Whiten a 5-D data tensor by slicing along the last dimension.

    Parameters
    ----------
    y     : (B, C, H, W, N)
    smaps : (B, C, H, W)
    Sigma : (B, C, C), optional

    Returns
    -------
    Same dict as whiten() but "data" is a 5-D Tensor (B, C, H, W, N).
    """
    slices = [y[..., ii] for ii in range(y.shape[-1])]
    result = whiten(slices, smaps, Sigma)
    result["data"] = torch.stack(result["data"], dim=-1)
    return result

def whiten_from_kspace(kspace, smaps, mask = None, R = None, gnd_truth_kspace = None):
    # Helper function to whiten from kspace. Our whitening function assumes multicoil image domain
    # Send kspace to mc image domain (works regardless of mask)
    x_mc_w, smaps_w, Sigma_n, Zinv = whiten(ifftc(kspace), smaps, mask = mask, R = R)
    sigma_n = Sigma_n.max()
    if mask is None:
        # if we don't pass a mask, then easy, whiten from fully sampled. 
        # We get multicoil image domain out, send to kspace and whiten.
        kspace_w = fftc(x_mc_w)
    else:
        # If we do have a mask, then we need to reapply our mask onto the whitened kspace since we did a matrix multiply
        kspace_w = mask*fftc(x_mc_w)
    if gnd_truth_kspace is None:
        # if I don't supply a ground truth, coil combine to get a whitened image
        image_w = torch.sum(smaps.conj()*x_mc_w, dim = 1, keepdim = True)
    else:
        image_w = torch.sum(smaps.conj()*ifftc(gnd_truth_kspace), dim = 1, keepdim = True)
    return kspace_w, image_w, smaps_w, sigma_n, smaps_w, Zinv

