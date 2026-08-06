import torch
import matplotlib.pyplot as plt


def contrast_enhance(x, thresh=1.0):
    """
    Clamp image intensities for visualization.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    thresh : float
        Maximum intensity value.

    Returns
    -------
    torch.Tensor
        Contrast-enhanced tensor.
    """
    return torch.clamp(x, max=thresh)


def prepare_image(x, magnitude=True, contrast=False, thresh=1.0):
    """
    Prepare a tensor for visualization.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor.
    magnitude : bool
        Take magnitude if tensor is complex.
    contrast : bool
        Apply contrast clipping.
    thresh : float
        Clipping threshold.

    Returns
    -------
    torch.Tensor
        Processed image tensor on CPU.
    """
    if magnitude:
        x = x.abs()

    x = torch.squeeze(x).detach().cpu()

    if contrast:
        x = contrast_enhance(x, thresh=thresh)

    return x


def plot_image(
    x,
    contrast=False,
    thresh=1.0,
    magnitude=True,
    cmap="gray",
    figsize=None,
    title=None,
    colorbar=False,
):
    """
    Display an image tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor.
    contrast : bool
        Apply contrast clipping.
    thresh : float
        Contrast threshold.
    magnitude : bool
        Take magnitude if complex-valued.
    cmap : str
        Matplotlib colormap.
    figsize : tuple or None
        Figure size.
    title : str or None
        Figure title.
    colorbar : bool
        Show colorbar.
    """
    x = prepare_image(
        x,
        magnitude=magnitude,
        contrast=contrast,
        thresh=thresh,
    )

    if figsize is not None:
        plt.figure(figsize=figsize)

    im = plt.imshow(x, cmap=cmap)

    if title is not None:
        plt.title(title)

    if colorbar:
        plt.colorbar(im)

    plt.axis("off")
    plt.show()


def save_image(
    x,
    path,
    contrast=False,
    thresh=1.0,
    magnitude=True,
    cmap="gray",
    dpi=300,
):
    """
    Save an image tensor to disk.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor.
    path : str
        Output filepath.
    contrast : bool
        Apply contrast clipping.
    thresh : float
        Contrast threshold.
    magnitude : bool
        Take magnitude if complex-valued.
    cmap : str
        Matplotlib colormap.
    dpi : int
        Output resolution.
    """
    x = prepare_image(
        x,
        magnitude=magnitude,
        contrast=contrast,
        thresh=thresh,
    )

    plt.figure()
    plt.imshow(x, cmap=cmap)
    plt.axis("off")

    plt.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
    )

    plt.close()

    print(f"Saved image to {path}.")


def show_kspace(
    kspace,
    log=True,
    cmap="gray",
    figsize=None,
):
    """
    Visualize k-space magnitude.

    Parameters
    ----------
    kspace : torch.Tensor
        Complex k-space tensor.
    log : bool
        Apply log transform.
    cmap : str
        Matplotlib colormap.
    figsize : tuple or None
        Figure size.
    """
    x = kspace.abs()

    if log:
        x = torch.log1p(x)

    plot_image(
        x,
        contrast=False,
        magnitude=False,
        cmap=cmap,
        figsize=figsize,
    )


# ===========================================================================
#  Reconstruction diagnostic panels
# ===========================================================================
def _to_mag(x):
    """(B,C,H,W) possibly-complex -> (1,1,H,W) real magnitude of the first slice."""
    x = x[:1]
    if x.shape[1] > 1:
        x = x[:, :1]
    return x.abs().detach().float().cpu()


def recon_panel(gt, recon, zero_filled=None, gain=4.0, eps=1e-12):
    """Side-by-side [zero-filled | recon | ground truth | |residual| x gain].

    The three IMAGE panels share one intensity scale (the ground truth's max),
    so they are directly comparable and an over/under-shooting reconstruction
    reads as such instead of being silently renormalised away. The residual is
    scaled by the SAME reference times `gain`, so its brightness means a fixed
    fraction of signal at every epoch -- an artifact that grows over training
    looks like it is growing.

    Returns `(panel, stats)`; `stats` carries the numbers worth putting in a
    caption (`res_max`, `res_rms`, both relative to the ground-truth max).
    """
    gt_m, rec_m = _to_mag(gt), _to_mag(recon)
    ref = gt_m.max().clamp_min(eps)

    res = (rec_m - gt_m).abs()
    stats = {"res_max": (res.max() / ref).item(),
             "res_rms": (res.pow(2).mean().sqrt() / ref).item(),
             "gain": float(gain)}

    panels = []
    if zero_filled is not None:
        panels.append((_to_mag(zero_filled) / ref).clamp(0, 1))
    panels += [(rec_m / ref).clamp(0, 1),
               (gt_m / ref).clamp(0, 1),
               (res * gain / ref).clamp(0, 1)]

    # Built with plain torch rather than make_grid: one strip, explicit
    # separators, and no renormalisation behind our back (make_grid's
    # `normalize` defaults differ across versions, and a silent rescale is
    # exactly what makes a residual panel lie).
    pad = torch.ones(1, 1, panels[0].shape[-2], 2)
    strip = [panels[0]]
    for q in panels[1:]:
        strip += [pad, q]
    return torch.cat(strip, dim=-1)[0], stats


def residual_kspace(gt, recon, eps=1e-12):
    """log-magnitude spectrum of the residual -- what kind of artifact is this?

    Reading it:
      * bright horizontal/vertical LINES at regular spacing -> undersampling
        fold-over. The spacing is the acceleration: replicas sit at multiples
        of N/R along the phase-encode axis. This is a reconstruction-quality
        problem (more data consistency / more iterations / stronger prior).
      * bright spots at the CORNERS and at the half/quarter band edges ->
        grid-locked structure at the Nyquist of a coarse level, i.e. the
        multigrid transfer operators or a strided ConvTranspose checkerboard.
        More depth will NOT remove this one.
      * a broadband floor -> plain noise.
    """
    d = _to_mag(recon) - _to_mag(gt)
    D = torch.fft.fftshift(torch.fft.fft2(d)).abs()
    D = torch.log10(D + eps)
    D = D - D.min()
    return (D / D.max().clamp_min(eps))[0]          # (1, H, W)
