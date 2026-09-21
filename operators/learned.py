"""
Learned (NONLINEAR) forward operators, for unrolled nets that were written for linear ones.

A linear operator's adjoint does not depend on where it is applied, which is what lets CDLNet
precompute E^H y once and write its gradient as E^H E x - E^H y. A learned E has no such adjoint:
its "adjoint" is the transposed Jacobian J_E(x)^T, a VJP that only exists AT a point x. So these
operators do not offer a free-standing `adjoint`. They expose the gradient of the data term as ONE
call that carries its linearisation point:

    data_grad(x, y) = grad_x 1/2 || y - E(x) ||^2 = J_E(x)^T (E(x) - y)

and set `nonlinear = True`, which is the switch a model checks to take its gradient-form branch
(see models/cdlnet.py). Calling `adjoint` raises, so a linear-only code path cannot silently mix
Jacobians taken at different points.

    LearnedOperator      E(x; c): a frozen models.forward_ops.ForwardOp with its side information
                         (e.g. T2, FLAIR) bound for one batch.
    LinearizedOperator   its Jacobian at a fixed point: a true linear operator with an exact
                         adjoint, for Gauss-Newton / CG (sb/immap_sb.py).
    BridgeDCOperator     the I2SB regression problem at one bridge step: the debiased bridge state
                         AND a learned-operator measurement, as one weighted data term.
"""

import torch

from operators.base import Operator


class LearnedOperator(Operator):
    """E(x) = net(x, c) with `c` fixed for this batch. `net` should be frozen and in eval mode."""

    nonlinear = True

    def __init__(self, net, cond=None):
        self.net = net
        self.cond = cond

    def forward(self, x):
        return self.net(x, self.cond)

    def adjoint(self, x):
        raise TypeError(
            "LearnedOperator is nonlinear: its adjoint is J_E(x)^T and needs the point x. Use "
            "data_grad(x, y), or adjoint_at(x, r) for J_E(x)^T r.")

    def adjoint_at(self, x, r):
        """J_E(x)^T r. Differentiable w.r.t. x (and r) when grad is enabled."""
        create = torch.is_grad_enabled()
        with torch.enable_grad():
            xin = x if (create and x.requires_grad) else x.detach().requires_grad_(True)
            out = self.net(xin, self.cond)
            (g,) = torch.autograd.grad(out, xin, grad_outputs=r, create_graph=create,
                                       allow_unused=True)
        return torch.zeros_like(x) if g is None else g

    def data_grad(self, x, y):
        """J_E(x)^T (E(x) - y) in one forward + one VJP.

        With grad enabled (training an unrolled net THROUGH this step) the graph is kept, so the
        loss differentiates through the VJP -- second order in E, first order in its frozen
        weights. Under torch.no_grad (validation, sampling) the result is a plain tensor."""
        return self.net.data_grad(x, y, self.cond, create_graph=torch.is_grad_enabled())


class LinearizedOperator(Operator):
    """The Jacobian J of a learned operator at a FIXED point x_bar -- a genuine LINEAR Operator.

        forward(v)  = J v        adjoint(u) = J^T u        gram(v) = J^T J v

    Because the JVP and VJP are taken at the same x_bar, J^T J is exactly symmetric and PSD
    (<u, J^T J v> = <J u, J v>), so it can go straight into CG. It passes the dot-product test,
    unlike the nonlinear operator it came from, whose adjoint only exists at a point.

    Built with the double-VJP trick rather than torch.func, so it runs on any torch: ONE forward of
    the net at x_bar, and then every J v / J^T u reuses that graph -- no further forwards. `value`
    holds A(x_bar), which the Gauss-Newton residual needs anyway.

    Inference only: results are detached. (Training THROUGH the solve would differentiate
    implicitly around the CG call, not through these graphs.)
    """

    def __init__(self, net, x_bar, cond=None):
        with torch.enable_grad():
            self._x = x_bar.detach().requires_grad_(True)
            self._out = net(self._x, cond)
            self._u = torch.zeros_like(self._out, requires_grad=True)
            # J^T u as a function of u, kept differentiable: grad of <J^T u, v> w.r.t. u is J v
            (self._jtu,) = torch.autograd.grad(self._out, self._x, grad_outputs=self._u,
                                               create_graph=True)
        self.value = self._out.detach()

    def forward(self, v):
        with torch.enable_grad():
            (jv,) = torch.autograd.grad(self._jtu, self._u, grad_outputs=v, retain_graph=True)
        return jv.detach()

    def adjoint(self, u):
        with torch.enable_grad():
            (jtu,) = torch.autograd.grad(self._out, self._x, grad_outputs=u, retain_graph=True)
        return jtu.detach()

    def gram(self, v):
        return self.adjoint(self.forward(v))


class BridgeDCOperator(Operator):
    """The data term of an unrolled I2SB regressor with a learned measurement operator.

    At bridge step k the regressor sees x_t and must return x0 (CT1). Two things are known about x0:

        r  = x_t - mu1 * x1  =  mu0 * x0 + sigma_sb * eps     (the bridge state, debiased)
        y  = T1              =  E(x0; T2, FLAIR) + e,  e ~ sigma_E

    The MAP data term is  mu0 * (mu0 x - r) / sigma_sb^2  +  J_E^T (E(x) - T1) / sigma_E^2.
    Normalising by the total precision  mu0^2 / sigma_sb^2 + 1 / sigma_E^2  gives

        data_grad(x) = [ sigma_E^2 * mu0 * (mu0 x - r)  +  sigma_sb^2 * J_E^T (E(x) - T1) ]
                       / ( sigma_E^2 * mu0^2 + sigma_sb^2 )

    which never divides by mu0 (it -> 0 at the prior end) and has Lipschitz constant
    <= max(1, L_E^2), so the step size is stable across the whole bridge:
        t -> 0  (mu0 -> 1, sigma_sb -> 0):  the bridge term; x_t is nearly x0
        t -> 1  (mu0 -> 0):                 pure data consistency with T1 through E

    `init(x_t)` is the same precision-weighted combination with E replaced by the identity -- the
    image the unrolled net starts from, in place of E^H y. `noise_level(x_t)` is that estimate's
    std, sigma_sb * sigma_E / sqrt(sigma_E^2 mu0^2 + sigma_sb^2): sigma_eff at t -> 0, sigma_E at
    t -> 1. It is what a noise-adaptive threshold should see.

    All bridge quantities are (B, 1, 1, 1) tensors for THIS batch at THIS step; `y` passed to
    data_grad / init is x_t itself.
    """

    nonlinear = True

    def __init__(self, E, x1, t1, mu0, mu1, std_sb, sigma_E):
        if not isinstance(E, LearnedOperator):
            raise TypeError("E must be a LearnedOperator (a frozen ForwardOp with its cond bound)")
        self.E, self.x1, self.t1 = E, x1, t1
        self.mu0, self.mu1, self.std_sb = mu0, mu1, std_sb
        self.var_E = float(sigma_E) ** 2
        self.den = self.var_E * mu0 ** 2 + std_sb ** 2

    def forward(self, x):
        """The stacked measurement model (mu0 * x, E(x)). Mostly for inspection."""
        return self.mu0 * x, self.E(x)

    def adjoint(self, x):
        raise TypeError("BridgeDCOperator is nonlinear: use data_grad(x, x_t).")

    def _r(self, x_t):
        return x_t - self.mu1 * self.x1

    def init(self, x_t):
        return (self.var_E * self.mu0 * self._r(x_t) + self.std_sb ** 2 * self.t1) / self.den

    def noise_level(self, x_t=None):
        return self.std_sb * self.var_E ** 0.5 / self.den.sqrt()

    def data_grad(self, x, x_t):
        g_bridge = self.mu0 * (self.mu0 * x - self._r(x_t))
        g_E = self.E.data_grad(x, self.t1)
        return (self.var_E * g_bridge + self.std_sb ** 2 * g_E) / self.den
