import torch 
import torch.fft as fft
import torch.nn as nn
import numpy as np
import os

from functorch import jacrev, jacfwd
from solvers.cg import cg
from solvers.rest import rest_prox as rest
from diffusion.common import init_diff, prep_noise_schedule
from operators import Identity
from models import NormUnet
from visualization.image import save_image as saveimg

def immap1(y, noise_level, denoiser, E, 
           sigma_max = 1.0, 
           sigma_min = 0.01, 
           # Base Diffusion Hyperparameters
           beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
           h_0 = 0.01,      # Initial step size
           save_dir = None, 
           verbose = True):
    # Set initial conditions
    x_t, t, sigma_t, sigma_t_prev, y = init_diff(y, sigma_max=sigma_max)

    # Refactor to take in a measurement operator E
    # Mean shifting
    EHy = E.H(y)
    x_t = x_t + torch.mean(EHy)

    with torch.no_grad():
        while sigma_t > sigma_min:
            def denoise(x, sigma, f = denoiser):
                x_hat, _ = f(x, Identity(), sigma=sigma)
                return x_hat
            x_hat_t = denoise(x_t, sigma_t)
            # Get noise level estimate
            sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
            sigma_t = sigma_t_sq**(0.5)
            
            # Tweedie's formula
            grad_prior = x_hat_t - x_t

            # PiGDM Laplace Approx (use * operator because the forward operator E starts with elementwise multiplication
            def S_t(x, noise_level=noise_level, sigma_t_sq = sigma_t_sq, E = E):
                # We do not actually want to explicitly compute Sigma_t, but rather have the ability to apply it to a matrix
                return noise_level**2 * x + sigma_t_sq/(1+sigma_t_sq)*E(E.H(x))
            # We want to solve sigma_t v_t = E x_hat - y
            # We may use CG since sigma_t is a covariance matrix + PSD symmetric matrix
            v_t, tol_reached = cg(S_t, E(x_hat_t) - y, max_iter = 500, tol=1e-3, verbose = False)
            EHv_t = E.H(v_t)
            # Compute vjp
            _, (grad_likelihood, _) = torch.autograd.functional.vjp(denoise, (x_t, sigma_t), EHv_t)
            grad_likelihood = -1*sigma_t_sq*grad_likelihood
            # Update step size
            h_t = h_0 * t/(1+h_0*(t-1))
            # Update noise injection
            gamma_t = sigma_t_sq**(0.5)*((1-beta*h_t)**2-(1-h_t)**2)**0.5
            noise = torch.randn_like(x_t)
            # Stochastic gradient ascent
            x_t = x_t + h_t * (grad_prior+grad_likelihood) + gamma_t*noise
            if t % 5 == 0 and save_dir:
                fname = os.path.join(save_dir, f"diffusion_sigma_{sigma_t.item():2f}.png")
                saveimg(x_t, fname)
            t = t + 1
            if verbose == True:
                print(f"Iteration {t} complete. Noise level: {sigma_t}. Tolerance Reached: {tol_reached}") 
            if sigma_t > sigma_t_prev:
                print("Noise is diverging...")
                continue
        if save_dir:
            fname = os.path.join(save_dir, "immap1_final.png")
            saveimg(x_t, fname)
    return x_t

def immap2(y, sigma_y, denoiser, E, 
           lam = 2.0, 
           sigma_max = 1.0, 
           sigma_min = 0.01, 
           # Base Diffusion Hyperparameters
           beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
           h_0 = 0.01,      # Initial step size
           save_dir = None, 
           verbose = True, 
           lam_min = 1e-9,
           use_rest=False, # Whether or not to use rest_prox
           ws = None):
    # Set initial conditions
    x_t, t, sigma_t, sigma_t_prev = init_diff(y, sigma_max=sigma_max, x_init=ws)

    # Precompute EHy for calculation
    EHy = E.H(y)

    # Mean shifting
    x_t = x_t + torch.mean(EHy)

    with torch.no_grad():
        while sigma_t > sigma_min:
            if isinstance(denoiser, NormUnet):
                x_hat_t = denoiser(torch.view_as_real(x_t))
                x_hat_t = torch.view_as_complex(x_hat_t.contiguous())
            else:
                x_hat_t, _ = denoiser(x_t, Identity(), sigma_t)
            # Get noise level estimate
            sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
            sigma_t = torch.sqrt(sigma_t_sq)
            
            # Compute proximal weighting
            p_t = lam*sigma_y**2 / (sigma_t_sq/(1+sigma_t_sq))

            # Try clamping to 0.01
            p_t = torch.clamp(p_t, min = lam_min)
            
            # Update step size
            h_t = h_0 * t/(1+h_0*(t-1))

            # Update noise injection
            gamma_t = sigma_t*((1-beta*h_t)**2-(1-h_t)**2)**0.5
            
            # Define operator for CG
            def A(x, E = E):
                return E.H(E(x)) + p_t*x
            if use_rest is True:
                v_t, tol_reached, _, _ = rest(E, E.H, v=x_hat_t, y=y, tau=1/p_t, x_est = x_hat_t, max_iter = 500, tol=1e-3, verbose = False)
            else:
                v_t, tol_reached = cg(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 500, tol=1e-3, verbose = False)
            
            # Perform update
            x_t = x_t +h_t*(v_t-x_t) + gamma_t*torch.randn_like(x_t)

            if t % 5 == 0 and save_dir:
                fname = os.path.join(save_dir, f"immap2_sigma_{sigma_t.item():.3f}.png")
                panel = torch.cat((x_hat_t, v_t, x_t), dim = 3)
                saveimg(panel, fname)
            if verbose == True:
                print(f"Iteration {t} complete. Noise level: {sigma_t}. p_t: {p_t} Tolerance Reached: {tol_reached}")
            t = t + 1
        # One final denoising step to be "noise free"
        if isinstance(denoiser, NormUnet):
            x_hat_t = denoiser(torch.view_as_real(x_t))
            x_hat_t = torch.view_as_complex(x_hat_t.contiguous())
        else:
            x_hat_t, _ = denoiser(x_t, Identity(), sigma_t)
        if save_dir:
            fname = os.path.join(save_dir, "immap2_final.png")
            saveimg(x_t, fname)
    return x_t

def immap2p5(y, sigma_y, net, E,    # Require usual inputs (measurent, nle, net, measurement op)
             sigma_max=1.0,             
             sigma_min=0.01,
             D = Identity(),            # Optional SSDU Operator for Image Domain masking
             organ_mask = None,         # Optional Organ Mask
             ws = None, 
             # Base Diffusion Hyperparameters
             beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
             h_0 = 0.01,      # Initial step size
             save_dir = None,           
             verbose = True, 
             ):
    # This implements a version of immap that conditions on an end to end reconstruction using a separate LPDSNet
    # Makes the approximation that E[x|x_t] = net(x_hat_t, 0, x_t, sigma_t)
    # Set initial conditions
    x_t, t, sigma_t, sigma_t_prev = init_diff(y, sigma_max=sigma_max, x_init=ws)
        
    # Precompute EHy for calculation
    EHy = E.H(y)
        
    # Optional organ masking, default to all ones if None
    if organ_mask is None:
        organ_mask = torch.ones_like(x_t[0,0]) == 1
        
    # # Mean shifting based on optional organ mask. 
    # x_t[:, :, organ_mask] = x_t[:, :, organ_mask] + torch.mean(EHy[:, :, organ_mask])
        
    with torch.no_grad():
        while sigma_t > sigma_min:
            if hasattr(D.ops[0], 'shuffle_mask'):
                D.ops[0].shuffle_mask(x_t)
            
            # Network forward pass (Assume JDR ImMAP2.5 net)
            v_t, _ = net(
                y, 
                E = E, 
                E_z = D,
                sigma = sigma_y,
                x_init=x_t,
                sigma_t=sigma_t,
            )
            
            # Apply organ mask    
            v_t = v_t * organ_mask
                
            # Noise Level Estimation
            sigma_t = torch.sqrt(torch.sum(((x_t-v_t)*organ_mask).abs()**2)/torch.sum(organ_mask))
                
            # update step size
            h_t = h_0 * t/(1+h_0*(t-1))
                
            # Update noise injection
            gamma_t = sigma_t*((1-beta*h_t)**2-(1-h_t)**2)**0.5
                
            # Update Eqn
            x_t = x_t * organ_mask + h_t * (v_t-x_t) + gamma_t * torch.randn_like(x_t)
                
            if t % 5 == 0 and save_dir:
                panel = torch.cat((x_t, v_t), dim = 3)
                fname = os.path.join(save_dir, f"immap2.5_sigma_{sigma_t.item():.3f}.png")
                saveimg(panel, fname, contrast=True)

            if verbose == True:
                print(f"Iteration {t} complete. Noise level: {sigma_t}.")
            t = t + 1

        if save_dir:
            fname = os.path.join(save_dir, "immap2.5_final.png")
            saveimg(v_t, fname, contrast=True)
                    
    return v_t
