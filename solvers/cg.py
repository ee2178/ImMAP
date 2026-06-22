import torch
from tqdm import tqdm
import numpy as np

def cg(A, b, x0 = None, tol = 1e-6, max_iter = 100, verbose = False):
    '''
    Let A be a Pytorch operator, necessarily symmetric, positive semi-definite
    We solve the system Ax = b
    '''
    if x0 is None:
        x0 = torch.zeros_like(b)
    # Compute first residual
    r = b - A(x0)
    p = r.clone()
    x = x0
    tol_reached = False
    for k in range(int(max_iter)):
        # Apply operator to p
        Ap = A(p)
        # Implement inner products as elementwise sums
        rsold = torch.sum(r.conj() * r).real
        alpha = rsold / (torch.sum(p.conj() * Ap) + 1e-8)
        x_next = x + alpha * p
        r_next = r - alpha * Ap
        rsnew = torch.sum(r_next.conj() * r_next).real
        beta = rsnew/(rsold + 1e-8)
        if (torch.sqrt(beta.real) <= tol) or torch.max((x_next - x).abs()) < tol:
            tol_reached = True
            return x_next, tol_reached
        p = r_next + beta * p
        r = r_next
        x = x_next
        if verbose is True:
            print(f"Iteration: {k}, Residual: {rsnew}")
    return x, tol_reached
