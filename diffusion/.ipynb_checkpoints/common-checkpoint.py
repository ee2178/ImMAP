import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np

# Helper Function To Initialize Standard Diffusion and Warm Start Diffusion
def init_diff(y, sigma_max = 1.0, x_init = None):
    # Get a random image of variance sigma_max
    if x_init is None:
        x_t = sigma_max*torch.randn(1, 1, y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device)
    else:
        # If we have some specified intial condition, then we should add noise to that input
        x_t = sigma_max*torch.randn(1, 1, y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device) + x_init
    # Set initial conditions
    t = 1
    sigma_t = torch.Tensor([sigma_max]).to(y.device)
    sigma_t_prev = sigma_t

    return x_t, t, sigma_t, sigma_t_prev

# Helper Function to Initialize Noise Schedules if Needed
def prep_noise_schedule(self, mode = 'linear', sigma_max = 1., sigma_min = 0.01, nsteps = 101):
    if mode == 'linear':
        # Return a linear noise schedule sigma_max to sigma_min
        ns = torch.linspace(sigma_max, sigma_min, nsteps)
    if mode == 'eero':
        # Eero noise schedule doesn't have a controllable number of steps
        ns = [sigma_max]
        i=1
        while ns[-1] > sigma_min:
            ns.append((1-self.beta * self.h_0 * i/(1+self.h_0*(i-1)))*ns[i-1])
            i=i+1
        ns = torch.tensor(ns)
    return ns
