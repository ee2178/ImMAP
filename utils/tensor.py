import torch


def truncate(x, start, end, dim):
    """
    Truncate a tensor along a specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    start : int
        Starting index (inclusive).
    end : int
        Ending index (exclusive).
    dim : int
        Dimension along which to truncate.

    Returns
    -------
    torch.Tensor
        Truncated tensor.
    """
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, end)

    return x[tuple(slices)]


def subsample(x, factor, dim, offset=0):
    """
    Subsample a tensor along a specified dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    factor : int
        Subsampling factor. Keeps every `factor`-th element.
    dim : int
        Dimension along which to subsample.
    offset : int, optional
        Starting index for sampling phase.

    Returns
    -------
    torch.Tensor
        Subsampled tensor.
    """
    if factor <= 0:
        raise ValueError("factor must be a positive integer")

    slices = [slice(None)] * x.ndim
    slices[dim] = slice(offset, None, factor)

    return x[tuple(slices)]


def center_crop(x, shape):
    """
    Center crop the last dimensions of a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    shape : tuple
        Desired output spatial shape.

    Returns
    -------
    torch.Tensor
        Center-cropped tensor.
    """
    if len(shape) > x.ndim:
        raise ValueError("Crop shape has more dims than tensor")

    slices = [slice(None)] * x.ndim

    for i, target in enumerate(shape):
        dim = x.ndim - len(shape) + i
        size = x.shape[dim]

        if target > size:
            raise ValueError(
                f"Target size {target} exceeds tensor size {size}"
            )

        start = (size - target) // 2
        end = start + target

        slices[dim] = slice(start, end)

    return x[tuple(slices)]


def pad_to_shape(x, shape, value=0):
    """
    Symmetrically pad the last dimensions of a tensor to a target shape.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    shape : tuple
        Desired output spatial shape.
    value : float, optional
        Padding value.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    import torch.nn.functional as F

    if len(shape) > x.ndim:
        raise ValueError("Pad shape has more dims than tensor")

    pad = []

    for i, target in reversed(list(enumerate(shape))):
        dim = x.ndim - len(shape) + i
        size = x.shape[dim]

        if target < size:
            raise ValueError(
                f"Target size {target} smaller than tensor size {size}"
            )

        diff = target - size

        left = diff // 2
        right = diff - left

        pad.extend([left, right])

    return F.pad(x, pad, value=value)
