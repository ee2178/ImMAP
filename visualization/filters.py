"""Filter-bank visualization for the unrolled models (CDLNet / GroupCDL / LPDSNet / ...).

Works for BOTH complex (`is_complex=True`) and real (`is_complex=False`) dictionaries. The
components expose a `.weight` property that is a complex tensor when the layer is complex and a
real tensor otherwise, so extraction is dtype-agnostic (the old `get_B_complex` path assumed
`conv_imag` existed and crashed on real models).

Rendering per input channel c of a filter tensor W (N, C, P, P):
  * complex weights -> R = real, G = 0, B = imag   (each mapped to [0,1] around 0.5)
  * real weights    -> white-centered diverging map (positive = red, negative = blue, 0 = white)
Multiple input channels are tiled horizontally (up to `max_channels`) into ONE image, so e.g.
the joint dict's D (C=4 = [x_t, FLAIR, T1, T2]) logs as a single 4-panel figure. Every grid is a
(3, H, W) float tensor in [0,1], the format `get_filter_grids` hands to `wandb.Image`.
"""

import os
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


# ============================================================
# (1) EXTRACTION LAYER — PURE TENSORS ONLY (real or complex)
# ============================================================

def _weight(layer):
    """Filter weight as a detached cpu tensor. Complex iff the layer is complex (the `.weight`
    property returns torch.complex(real, imag) for complex layers and the real weight otherwise)."""
    return layer.weight.detach().cpu()


def get_B_complex(B_layer):
    """Back-compat helper. Returns the (possibly real) weight of a Conv/ConvTranspose component;
    real when the layer has no imaginary branch (is_complex=False)."""
    if getattr(B_layer, "conv_imag", None) is None:
        return B_layer.conv_real.weight.detach()
    return torch.complex(B_layer.conv_real.weight.detach(),
                         B_layer.conv_imag.weight.detach())


def extract_lpds_filters(net):
    """Extract raw filters (A, B, D) as tensors (real or complex, per the model).

    Returns dict with A (list), B (list), D (tensor), global_max (float over A,B), K (int).
    """
    assert hasattr(net, "A") and hasattr(net, "B"), "net has no A/B filter banks"

    K = net.K
    A_list = [_weight(net.A[k]) for k in range(K)]
    B_list = [_weight(net.B[k]) for k in range(K)]
    D = _weight(net.D) if hasattr(net, "D") else B_list[0]

    global_max = 0.0
    for W in A_list + B_list:
        global_max = max(global_max, float(W.abs().max()))

    return {"A": A_list, "B": B_list, "D": D, "global_max": global_max, "K": K}


# ============================================================
# (2) RENDERING LAYER — PURE VISUALIZATION (NO I/O)
# ============================================================

def _complex_rgb(Wc, scale_each, global_max):
    """(N, 1, P, P) complex -> (N, 3, P, P) in [0,1]: R = real, G = 0, B = imag (centered 0.5)."""
    real, imag = torch.real(Wc), torch.imag(Wc)
    if scale_each:
        rmax = real.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        imax = imag.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:
        rmax = imax = global_max + 1e-8
    real = (real / rmax + 1) / 2
    imag = (imag / imax + 1) / 2
    green = torch.zeros_like(real)
    return torch.cat([real, green, imag], dim=1).clamp(0, 1)


def _real_rgb(Wc, scale_each, global_max):
    """(N, 1, P, P) real -> (N, 3, P, P) in [0,1]: white-centered diverging (+red, -blue, 0 white)."""
    if scale_each:
        m = Wc.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:
        m = global_max + 1e-8
    g = (Wc / m).clamp(-1, 1)               # (N,1,P,P) in [-1,1]
    p = g.clamp(min=0)                       # positive part
    n = (-g).clamp(min=0)                    # negative part
    rgb = torch.cat([1 - n, 1 - n - p, 1 - p], dim=1)     # R, G, B
    return rgb.clamp(0, 1)


def filter_to_grid(W, nrow=None, scale_each=False, global_max=None, max_channels=4):
    """Render one filter tensor W (N, C, P, P) — real or complex — to a (3, H, W) RGB grid in
    [0,1]. Up to `max_channels` input channels are tiled horizontally."""
    if W.dim() == 3:                                  # (N, P, P) -> (N, 1, P, P)
        W = W.unsqueeze(1)
    N, C, P, _ = W.shape
    is_cplx = W.is_complex()
    if global_max is None:
        global_max = float(W.abs().max())
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(N)))
    pad_val = 0.5 if is_cplx else 1.0                 # neutral (zero) color for the border

    panels = []
    n_ch = C if max_channels is None else min(C, max_channels)
    for c in range(n_ch):
        Wc = W[:, c:c + 1]
        rgb = _complex_rgb(Wc, scale_each, global_max) if is_cplx \
            else _real_rgb(Wc, scale_each, global_max)
        panels.append(make_grid(rgb, nrow=nrow, padding=2, pad_value=pad_val))   # (3, Hg, Wg)

    if len(panels) == 1:
        return panels[0]
    sep = torch.full((3, panels[0].shape[1], 2), pad_val)     # thin separator between channels
    row = []
    for i, p in enumerate(panels):
        row.append(p)
        if i < len(panels) - 1:
            row.append(sep)
    return torch.cat(row, dim=2)


# back-compat alias: old name assumed complex + single channel; the unified renderer covers it.
def complex_to_rgb_grid(W, nrow, scale_each=False, global_max=None):
    return filter_to_grid(W, nrow=nrow, scale_each=scale_each, global_max=global_max)


def render_lpds_filters(filters, scale_each=False, max_channels=4):
    """Render extracted filters into {name: (3, H, W)} grids (A/B per stage + D)."""
    A_list, B_list, D = filters["A"], filters["B"], filters["D"]
    global_max, K = filters["global_max"], filters["K"]
    nrow = int(np.ceil(np.sqrt(A_list[0].shape[0])))

    out = {}
    for k in range(K):
        out[f"A_stage_{k:02d}"] = filter_to_grid(A_list[k], nrow, scale_each, global_max, max_channels)
        out[f"B_stage_{k:02d}"] = filter_to_grid(B_list[k], nrow, scale_each, global_max, max_channels)
    out["D"] = filter_to_grid(D, nrow, scale_each, global_max, max_channels)
    return out


# ============================================================
# (3) IO / LOGGING LAYER — W&B OR DISK
# ============================================================

def get_filter_grids(net, scale_each=False, max_channels=4):
    """W&B-compatible logging dict {filters/A_stage_00, ..., filters/D}. Returns {} (never raises)
    for models without A/B filter banks, so unwrapped callers stay safe."""
    import wandb

    if not (hasattr(net, "A") and hasattr(net, "B")):
        return {}

    filters = extract_lpds_filters(net)
    grids = render_lpds_filters(filters, scale_each=scale_each, max_channels=max_channels)
    return {f"filters/{k}": wandb.Image(img.permute(1, 2, 0).numpy())
            for k, img in grids.items()}


def save_filters(net, save_dir, scale_each=False, max_channels=4):
    """Save filter visualizations to disk as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    filters = extract_lpds_filters(net)
    grids = render_lpds_filters(filters, scale_each=scale_each, max_channels=max_channels)
    for name, img in grids.items():
        save_image(img, os.path.join(save_dir, f"{name}.png"))
