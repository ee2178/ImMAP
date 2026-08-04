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

# NOTE: `mri_awgn` lives in operators/noise.py. A stale 4-argument copy used to
# sit here; it never applied the sensitivity maps and referenced an undefined
# name, and having two functions of the same name in sibling modules made the
# wrong one easy to import.

### Creating Operator Classes for Fourier and MRI Ops

class FFT2D(Operator):
    # Use our operator class and existing fftc functions to define a fourier operator class with a proper adjoint
    def forward(self, x):
        return fftc(x)
    
    def adjoint(self, x):
        return ifftc(x)


