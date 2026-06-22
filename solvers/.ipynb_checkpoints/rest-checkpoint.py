import torch
import numpy as np
from solvers.cg import cg as conj_grad

def img_2_norm(x):
    return torch.sqrt(torch.sum(x.conj()*x).real/torch.numel(x))
    
def rest_prox(  A,                  # Linear Operator from standard solve
                AH,                 # Hermitian Transpose of A
                v,                  # Input to proximal operator
                y,                  # Measurement
                tau,                # Proximal Weight
                x_est = None,       # Approximation to solution
                max_iter = 10,      # Number of outer loop iterations for alpha/beta estimation        
                tol = 1e-6,         # CG Settings
                max_iter_cg = 100,  # CG Settings
                verbose = False     # CG Settings
            ):
    # This function will implement the proximal operator of the REST objective
    
    # x0 initialization
    if x_est is None:
        x0 = torch.zeros_like(v)
    else:
        x0 = x_est
    x = x0
    tol_reached = False
    with torch.no_grad():
        for k in range(max_iter):
            # The CG here requires two auxiliary scalars alpha and beta, both functions of the solution. 
            # In a diffusion context, we can warm start this with an estimate of x, which we get at every iteration.
            alpha = 1/(1+img_2_norm(x)**2)
            # Now, we can run a CG with a modified operator:
            # Compute safe beta
            raw_beta = (img_2_norm(y - A(x))**2) * alpha**2
            beta = torch.clamp(raw_beta, max=(1.0 - 1e-4) / (2 * tau))
            
            def rest_op(z, A=A, AH=AH, alpha=alpha, beta=beta):
                diag_coeff = 1 - 2 * tau * beta   # now guaranteed > 0
                return diag_coeff * z + 2 * tau * alpha * AH(A(z))
        
            # Form new b
            b = v + 2*tau*alpha * AH(y)
    
            # Now run cg with our new b
            x_next, tol_reached_cg = conj_grad(rest_op, b, tol = tol, x0 = None, max_iter = max_iter_cg, verbose = verbose)
            # Here, we should define some stopping condition. Observe that as we get a better solution, beta should go to zero and 
            if torch.max((x_next - x).abs()) < tol:
            # if beta <= tol:
                tol_reached = True
                return x_next, tol_reached, alpha, beta
            x = x_next
    return x, tol_reached, alpha, beta
