import torch


def pca_pixelwise(x: torch.Tensor, n_components: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PCA on a B x C x H x W image tensor, treating C as the feature dimension.

    Args:
        x:            Input tensor of shape (B, C, H, W)
        n_components: Number of principal components to retain

    Returns:
        projected:    Shape (B, n_components, H, W) — pixel features in PC space
        components:   Shape (n_components, C)        — principal component vectors
        mean:         Shape (C,)                     — per-channel mean used for centering
    """
    B, C, H, W = x.shape

    # (B, C, H, W) -> (B*H*W, C): each pixel is one observation
    pixels = x.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)  where N = B*H*W

    # Center
    mean = pixels.mean(dim=0)                        # (C,)
    pixels_centered = pixels - mean                  # (N, C)

    # Covariance via SVD on the data matrix (numerically stabler than eig on cov)
    # torch.linalg.svd on (N, C) with N >> C: use full_matrices=False for economy SVD
    # V has shape (C, C); columns are right singular vectors = principal components
    _, S, Vh = torch.linalg.svd(pixels_centered, full_matrices=False)  # Vh: (C, C)

    components = Vh[:n_components]                   # (n_components, C)

    # Project every pixel onto the PC basis
    projected_flat = pixels_centered @ components.T  # (N, n_components)

    # Reshape back to (B, H, W, n_components) -> (B, n_components, H, W)
    projected = projected_flat.reshape(B, H, W, n_components).permute(0, 3, 1, 2)

    return projected, components, mean
