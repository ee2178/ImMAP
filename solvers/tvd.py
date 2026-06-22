import torch
import numpy as np

from models.components import ST, CLIP
from operators import Identity
from solvers.cg import cg
from solvers.rest import img_2_norm
from solvers.fista import fista
from operators.difference import Diff2D, Diff2D_FFT

def tvd(y,      # Measurement
        E,      # Measurement Operator
        lam,    # Objective Function penalty weight
        rho,    # Augmented Lagrangian multiplier
        maxit = 100,    
        tol = 1e-3,
        verbose = False,
        ):

    # Here, we implement Total Variation Denoising for solving general linear inverse problems through a scaled-form ADMM style decomposition. 
    
    # First, initialize our difference operator. 
    D = Diff2D_FFT()
    # Second, initialize our forward operator for CG
    def A(x, E=E, D=D, rho = rho):
        return E.normal(x)+rho*D.normal(x)

    # Zero initialization for z and v
    z = torch.zeros_like(D(E.H(y)))
    v = torch.zeros_like(D(E.H(y)))

    # Precompute EHy
    EHy = E.H(y)

    tol_reached=False
    for i in range(maxit):
        # x-update. Here, we solve a symmetric linear system, so use CG
        x, cg_tol_reached = cg(A = A, b = EHy+rho*D.H(z-v), max_iter = maxit, tol = tol)
        # z-update, soft thresholding.
        z = ST(D(x)+v, lam/rho)
        # v-update, i.e. dual ascent
        v = v + (D(x)-z)

        # Early exit check
        r = img_2_norm(D(x)-z)
        if verbose == True:
            print(f"Iteration {i}: r = {r:.3f}, CG Tolerance Reached: {cg_tol_reached}")
        if r < tol:
            print(f"Reached tolerance at Iteration {i}")
            tol_reached=True
            return x, z, v, tol_reached

    return x, z, v, tol_reached

# Implement isotropic TV Denoising using FISTA on the dual problem
def tvd_fista(y,      # Measurement
              lam,    # Objective Function penalty weight
              eta = 4.0,    # Approximate Lipschitz constant
              isotropic = False,    # Isotropic or Not, changes the prox operator. 
              maxit = 100,
              tol = 1e-3,
              verbose = False,
            ):
    # We need the transposed D operator. 
    D = Diff2D_FFT()
    DH = D.H # The .T operator returns an actual operator that is transposed. 

    # Define a small epsilon to avoid division by 0:
    eps = 1e-12

    # defining prox operator. But, since we assume that the data is complex valued, isotropic and anisotropic projections are more or less the same, i.e. both forms of clipping.
    if isotropic is False:
        # Anisotropic case: component wise clipping
        def prox(z):
            return CLIP(z, lam)
    if isotropic is True:
        # Isotropic case: clipping on blocks:
        def prox(z):
            # I return 2 x 1 x H x W for my D.H operator
            norm = z.abs().pow(2).sum(dim=0, keepdim=True).sqrt()
            return z * CLIP(norm, lam) / norm.clamp_min(eps)

    # Now, call FISTA.
    p_star, q, t, tol_reached = fista(y, DH, prox, eta, max_iter = maxit, tol = tol, verbose = verbose)

    # To get the optimal primal solution, we can make use of the substitution x = y - D^Hp. However, this only works because we consider strictly a denoising problem. We would need some CG step to handle using an arbitrary measurement operator. 
    return y - DH(p_star), p_star, tol_reached

