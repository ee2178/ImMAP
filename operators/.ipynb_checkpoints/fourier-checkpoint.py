import torch
import torch.fft as fft
import torch.nn.functional as F
import math
from typing import Tuple
from operators.base import Operator

### General MRI Utilities

def fftc(x, dim = (-2, -1), mode = 'ortho', real = False):
    # If our input is real, use rfft and irfft
    if real is True:
        return fft.fftshift(fft.rfftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)
    else:
        return fft.fftshift(fft.fftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)

def ifftc(x, dim = (-2, -1), mode = 'ortho', real = False):
    if real is True:
        return fft.fftshift(fft.irfftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)
    else:
        return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim = dim), dim = dim, norm = mode), dim = dim)

def mri_encoding(x, mask, smaps):
    # x         B x 1 x H x W
    # smaps     B x C x H x W
    # mask      B x 1 x H x W
    x_coils = smaps * x         # B x C x H x W
    y_coils = fftc(x_coils)     # B x C x H x W
    y_mask = y_coils * mask     # B x C x H x W
    return y_mask

def mri_decoding(y, mask, smaps):
    # y         B x C x H x W
    # smaps     B x C x H x W
    # mask      B x 1 x H x W
    y_mask = mask * y           # B x C x H x W
    x_coils = ifftc(y_mask)     # B x C x H x W
    x = torch.sum(smaps.conj()*x_coils, dim = 1, keepdim = True) # B x 1 x H x W
    return x

def mri_awgn(x_coils, acceleration_map, smaps, noise_std):
    # Assume we take in a multicoil image
    if not isinstance(noise_std, (list, tuple)):
        sigma = noise_std
    elif isinstance(noise_std, (list, tuple)): # uniform sampling of sigma
        sigma = noise_std[0] + \
               (noise_std[1] - noise_std[0])*torch.rand(1, device=x.device)
    x_coils_noisy = x_coils + sigma*torch.randn_like(x_coils)
    y_coils = fftc(x_coils_noisy)
    y_mask = y_coils * acceleration_map
    # Always return masked kspace
    return y_mask, sigma

### Creating Operator Classes for Fourier and MRI Ops

class FFT2D(Operator):
    # Use our operator class and existing fftc functions to define a fourier operator class with a proper adjoint
    def forward(self, x):
        return fftc(x)
    
    def adjoint(self, x):
        return ifftc(x)


