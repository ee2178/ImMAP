import torch


def uball_project(W, dim=(2, 3)):
    """
    Project tensor onto the unit ball along specified dimensions.

    Parameters
    ----------
    W : torch.Tensor
        Input tensor.
    dim : tuple
        Dimensions over which to compute the norm.

    Returns
    -------
    torch.Tensor
        Projected tensor.
    """
    normW = torch.linalg.norm(W, dim=dim, keepdim=True)

    return W * torch.clamp(1 / normW, max=1)
