import torch
import torch.nn.functional as F

# Gaussian window
def gaussian_window(pixel_width, device=None, dtype=torch.float32):
    """
    pixel_width: radius of kernel
    kernel size = 2 * pixel_width + 1
    sigma = kernel_width / 5
    """
    kernel_width = 2 * pixel_width + 1
    sigma = kernel_width / 5.0

    coords = torch.arange(kernel_width, device=device, dtype=dtype) - pixel_width

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    w = torch.outer(g, g)
    w = w / w.sum()

    return w.view(1, 1, kernel_width, kernel_width)


def complex_conv2d(img, weight):
    """
    img: (B, C, H, W) complex
    weight: (C, 1, k, k) real-valued kernel (depthwise)
    """
    C = img.shape[1]
    k = weight.shape[-1]
    padding = k // 2

    real = F.conv2d(img.real, weight, padding=padding, groups=C)
    imag = F.conv2d(img.imag, weight, padding=padding, groups=C)

    return torch.complex(real, imag)


def gaussian_blur_complex(img, pixel_width):
    """
    img: (B, C, H, W) complex
    pixel_width: radius (NOT kernel size)
    """
    B, C, H, W = img.shape

    kernel = gaussian_window(
        pixel_width,
        device=img.device,
        dtype=img.real.dtype
    )

    # Depthwise expansion
    kernel = kernel.expand(C, 1, kernel.shape[-1], kernel.shape[-1])

    return complex_conv2d(img, kernel)
