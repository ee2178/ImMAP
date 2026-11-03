import torch
from tqdm import tqdm
import numpy as np


# ===========================================================================
#  Batched CG and the Tikhonov solve  (lam I + C) x = b
#  Port of `cg` / `tcg` (+ its rrule) from Sljiva's src/solver.jl.
# ===========================================================================
def _bsum(x, keepdim=True):
    """Sum over everything but the batch axis -- one inner product per image."""
    return x.sum(dim=tuple(range(1, x.dim())), keepdim=keepdim)


STOP_RULES = ("ratio", "rhs")


def batched_cg(A, b, x0=None, tol=1e-6, max_iter=100, verbose=False, stop="ratio"):
    """Conjugate gradients on a BATCH of independent Hermitian PSD systems.

    Unlike `cg` above, the inner products are reduced per batch element, so
    every image gets its own step sizes.  With one global alpha (a single
    scalar for the whole batch) a well-conditioned slice is throttled by the
    worst-conditioned one and the batch converges at the speed of its slowest
    member -- which matters here because the unrolled networks call this
    inside every layer.

    `stop` picks the stopping rule:
      "ratio"  sqrt(|r_k|^2 / |r_{k-1}|^2) < tol for the worst image, or no entry
               of x moved by tol (Sljiva's rule; the default, what every caller
               before 2026-11-03 used)
      "rhs"    |r_k| < tol |b| for EVERY image, checked after the update and nothing
               else: deepinv 0.4.2's `conjugate_gradient`, iteration for iteration.
               Its every-100-iterations residual refresh is not reproduced, so
               the two agree up to max_iter = 101.
    """
    if stop not in STOP_RULES:
        raise ValueError(f"stop must be one of {STOP_RULES}, got {stop!r}")
    x = torch.zeros_like(b) if x0 is None else x0
    r = b - A(x)
    p = r.clone()
    rs_old = _bsum((r.conj() * r).real)
    if stop == "rhs":
        b_sq = _bsum((b.conj() * b).real)
        rs_tol = torch.where(b_sq > 0, b_sq, torch.ones_like(b_sq)) * tol ** 2

    for k in range(int(max_iter)):
        Ap = A(p)
        # A is Hermitian, so <p, Ap> is real; taking .real also keeps the
        # recursion stable when A is only approximately self-adjoint.
        alpha = rs_old / (_bsum((p.conj() * Ap).real) + 1e-8)
        x_old = x
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _bsum((r.conj() * r).real)
        if stop == "rhs":
            if verbose:
                print(f"k: {k:3d}, |r|/|b| = {(rs_new / rs_tol).sqrt().max().item() * tol:.3e}")
            if bool(torch.all(rs_new < rs_tol)):
                break
            beta = rs_new / (rs_old + 1e-8)
            p = r + beta * p
            rs_old = rs_new
            continue
        beta = rs_new / (rs_old + 1e-8)
        relres = torch.sqrt(beta).max()
        if verbose:
            print(f"k: {k:3d}, relres= {relres.item():.3e}")
        if relres < tol or (x - x_old).abs().max() < tol:
            break
        p = r + beta * p
        rs_old = rs_new
    return x


def _reduce_to(t, shape):
    """Sum `t` down to `shape`, undoing broadcasting."""
    while t.dim() > len(shape):
        t = t.sum(dim=0)
    for i, s in enumerate(shape):
        if s == 1 and t.shape[i] != 1:
            t = t.sum(dim=i, keepdim=True)
    return t.reshape(shape)


class _ImplicitTCG(torch.autograd.Function):
    """`(lam I + C)^{-1} b` differentiated by the implicit function theorem.

    Backprop through an unrolled CG stores every iterate; here the forward runs
    under `no_grad` and the backward recovers the exact same gradients from one
    extra solve, because for `A x = b` with A Hermitian:

        dL/db     = A^{-1} dL/dx                       (A^{-H} = A^{-1})
        dL/dlam   = -Re< x, A^{-1} dL/dx >
        dL/dtheta = VJP of C(x) at cotangent -A^{-1} dL/dx

    Memory is O(1) in the iteration count and the gradient is that of the
    *converged* solution rather than of the truncated recursion.  Tensors that
    C depends on must be passed explicitly via `params` so autograd can see
    them (Julia gets them for free from `rrule_via_ad`).
    """

    @staticmethod
    def forward(ctx, C, kw, x0, lam, b, *params):
        # x0 only warm-starts the forward solve: the solution does not depend on
        # it, so it gets no gradient, and the backward solve starts from zero.
        with torch.no_grad():
            x = batched_cg(lambda v: lam * v + C(v), b, x0=x0, **kw)
        ctx.C, ctx.kw = C, kw
        ctx.save_for_backward(x, lam, *params)
        return x

    @staticmethod
    def backward(ctx, gx):
        x, lam = ctx.saved_tensors[0], ctx.saved_tensors[1]
        params = ctx.saved_tensors[2:]
        with torch.no_grad():
            g = batched_cg(lambda v: lam * v + ctx.C(v), gx, **ctx.kw)

        gb = g if ctx.needs_input_grad[4] else None
        glam = None
        if ctx.needs_input_grad[3]:
            glam = _reduce_to(-(x.conj() * g).real, lam.shape)

        gparams = [None] * len(params)
        if params and any(ctx.needs_input_grad[5:]):
            with torch.enable_grad():
                out = ctx.C(x.detach())
                grads = torch.autograd.grad(out, params, grad_outputs=-g,
                                            allow_unused=True, retain_graph=False)
            gparams = list(grads)
        return (None, None, None, glam, gb, *gparams)


def tcg(C, lam, b, params=(), implicit=False, tol=1e-3, max_iter=10,
        verbose=False, x0=None, stop="ratio"):
    """Solve `(lam I + C) x = b`.

    `C` is a callable normal operator (e.g. `E.gram`).  `lam > 0` may be a
    scalar or a per-image tensor.  Set `implicit=True` and list the tensors C
    depends on in `params` to get the O(1)-memory implicit gradient instead of
    backprop through the iterations. `x0` warm-starts the forward solve either
    way (the implicit backward solve starts from zero), and `stop` is
    `batched_cg`'s stopping rule.
    """
    kw = dict(tol=tol, max_iter=max_iter, verbose=verbose, stop=stop)
    if not implicit:
        return batched_cg(lambda v: lam * v + C(v), b, x0=x0, **kw)
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(float(lam), device=b.device,
                              dtype=b.real.dtype if b.is_complex() else b.dtype)
    x0 = None if x0 is None else x0.detach()
    return _ImplicitTCG.apply(C, kw, x0, lam, b, *tuple(params))


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
