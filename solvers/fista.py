import torch

def fista(y,            # Measurement
          E,            # Measurement Operator
          prox,         # Proximal operator 
          L,            # Lipschitz constant of gradF (approximate)
          p_0=None,     # Initial estimate of x (needed for shape estimates)
          max_iter = 100, 
          tol = 1e-3,
          verbose = False,
          ):
    # This implementation of FISTA solves an optimization problem
    # argmin 1/2 ||y-x||_2^2 + lam * g(x), where g is our regularizer. 
    # We need 2 main ingredients: the proximal operator of g, and L, the Lipschitz constant of gradF
    
    if p_0 is None:
        p_0 = torch.zeros_like(E.H(y))

    # Returns x, t, y, tolerance_reached. 
    p_prev = p_0
    q = torch.clone(p_0)
    t_prev = 1

    for i in range(max_iter):
        # FISTA iterations
        p = prox(q - 1/L*E.H(E(q)-y))
        # Calculating acceleration term 
        t = (1+(1+4*t_prev**2)**.5)/2
        # Adjusting momentum
        q = p + (t_prev-1)/t*(p-p_prev)
        
        res = torch.max((p_prev - p).abs())

        if verbose:
            print(f"Iteration {i}: Maximum Residual is {res:.3f}")

        # Checking early exit 
        if res < tol:
            print(f"Completed FISTA at Iteration {i}")
            return p, t, q, True

        # Updating x and t
        p_prev = p
        t_prev = t

    return p, q, t, False
