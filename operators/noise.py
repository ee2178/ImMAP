import torch
import math

from operators.fourier import fftc
from math import log

def mri_awgn(x_coils, acceleration_map, smaps, noise_std, noise_dist):
    ### TO DO (not so important). Implement noise distributions

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

def awgn(input, noise_std, dist = 'uniform', k = 1, eps = 1e-8):
    """ Additive White Gaussian Noise
    y: clean input image
    noise_std: (tuple) noise_std of batch size N is uniformly sampled
                between noise_std[0] and noise_std[1]. Expected to be in interval
                [0,1]
    """
    # one sigma per batch element, singleton elsewhere -> broadcasts against
    # input whether it's (B, C, H, W), (B, H, W), etc.
    shape = (input.shape[0],) + (1,) * (input.dim() - 1)

    if not isinstance(noise_std, (list, tuple)):
        sigma = noise_std
    elif isinstance(noise_std, (list, tuple)) and dist == 'uniform': # uniform sampling of sigma
        sigma = noise_std[0] + \
               (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
    
    # Implement a power warp on the log distribution. 
    elif isinstance(noise_std, (list, tuple)) and dist == 'log':
        # log-uniform on [a, b] with a shape/"temperature" knob k
        #   k = 1 -> plain log-uniform
        #   k > 1 -> mass pushed toward low sigma (a)
        #   k < 1 -> mass pushed toward high sigma (b)
        a = noise_std[0] + eps
        b = noise_std[1]
        w = torch.rand(shape, device=input.device) ** k   # endpoints unchanged
        sigma = a * (b / a) ** w

    elif isinstance(noise_std, (list, tuple)) and dist == 'cosine':
        # \sigma = ((cos(X)+1)/2)^2, X ~ U([cos^-1(2*a^0.5-1), cos^-1(2*b^0.5-1)])
        # Draw uniform number [0, 1], map to our desired data range

        # The lower bound actually comes from b, not a, since arccos is monotonically decreasing

        x = math.acos(2*noise_std[1]**0.5-1) + \
        (math.acos(2*noise_std[0]**0.5-1)-math.acos(2*noise_std[1]**0.5-1))*torch.rand(len(input),1,1,1, device=input.device)

        sigma = ((torch.cos(x)+1)/2)**2
    return input + torch.randn_like(input) * (sigma), sigma
