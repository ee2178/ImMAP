import os
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


# ============================================================
# (1) EXTRACTION LAYER — PURE TENSORS ONLY
# ============================================================

def get_B_complex(B_layer):
    """Reconstruct complex weights from ComplexConvTranspose2d."""
    return torch.complex(
        B_layer.conv_real.weight.detach(),
        B_layer.conv_imag.weight.detach()
    )


def extract_lpds_filters(net):
    """
    Extract raw LPDS filters (A, B, D) as tensors.

    Returns
    -------
    dict with:
        A: list of tensors
        B: list of tensors (complex)
        D: tensor (complex)
        global_max: float
    """
    assert hasattr(net, "A") and hasattr(net, "B")

    K = net.K
    A_list, B_list = [], []
    global_max = 0.0

    for k in range(K):
        A = net.A[k].weight.detach().cpu()
        B = get_B_complex(net.B[k]).cpu()

        A_list.append(A)
        B_list.append(B)

        global_max = max(
            global_max,
            A.abs().max().item(),
            B.abs().max().item()
        )

    D = get_B_complex(net.D).cpu()

    return {
        "A": A_list,
        "B": B_list,
        "D": D,
        "global_max": global_max,
        "K": K,
    }


# ============================================================
# (2) RENDERING LAYER — PURE VISUALIZATION (NO I/O)
# ============================================================

def complex_to_rgb_grid(W, nrow, scale_each=False, global_max=None):
    """
    Convert complex filters to RGB grid:
        R = real, G = 0, B = imag
    """
    real = torch.real(W)
    imag = torch.imag(W)

    if scale_each:
        rmax = real.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        imax = imag.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    else:
        rmax = global_max + 1e-8
        imax = global_max + 1e-8

    real = (real / rmax + 1) / 2
    imag = (imag / imax + 1) / 2

    green = torch.zeros_like(real)

    rgb = torch.stack([real, green, imag], dim=2)
    rgb = rgb.squeeze(1)  # (N, 3, H, W)

    return make_grid(rgb, nrow=nrow, padding=2)


def render_lpds_filters(filters, scale_each=False):
    """
    Render LPDS filters into image grids.

    Returns:
        dict of torch.Tensors (grid images)
    """
    A_list = filters["A"]
    B_list = filters["B"]
    D = filters["D"]
    global_max = filters["global_max"]
    K = filters["K"]

    n = int(np.ceil(np.sqrt(A_list[0].shape[0])))

    out = {}

    for k in range(K):
        out[f"A_stage_{k:02d}"] = complex_to_rgb_grid(
            A_list[k],
            nrow=n,
            scale_each=scale_each,
            global_max=global_max
        )

        out[f"B_stage_{k:02d}"] = complex_to_rgb_grid(
            B_list[k],
            nrow=n,
            scale_each=scale_each,
            global_max=global_max
        )

    out["D"] = complex_to_rgb_grid(
        D,
        nrow=n,
        scale_each=scale_each,
        global_max=global_max
    )

    return out


# ============================================================
# (3) IO / LOGGING LAYER — W&B OR DISK
# ============================================================

def get_filter_grids(net, scale_each=False):
    """
    W&B-compatible logging dict:
        filters/A_stage_00
        filters/B_stage_00
        filters/D
    """
    import wandb

    filters = extract_lpds_filters(net)
    grids = render_lpds_filters(filters, scale_each=scale_each)

    logs = {}

    for k, img in grids.items():
        logs[f"filters/{k}"] = wandb.Image(
            img.permute(1, 2, 0).numpy()
        )

    return logs


def save_filters(net, save_dir, scale_each=False):
    """
    Save filter visualizations to disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    filters = extract_lpds_filters(net)
    grids = render_lpds_filters(filters, scale_each=scale_each)

    for name, img in grids.items():
        path = os.path.join(save_dir, f"{name}.png")
        save_image(img, path)
