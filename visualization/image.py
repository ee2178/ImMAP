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
