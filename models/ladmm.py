"""
Unrolled linearized ADMM for MRI reconstruction, with optional joint
coil-sensitivity estimation.

PyTorch port of Sljiva's `src/networks/ladmm.jl`.

One LADMM layer is

    x-update :  (E^H E + rho I) x = y~ + rho (Dz - u)      (Tikhonov CG)
    z-update :  Dz = denoiser(x + u ; sigma)               (a learned prox)
    u-update :  u  = u + x - Dz                            (dual ascent)

`rho(sigma)` is a learned, noise-adaptive per-channel penalty.  The denoiser is
any network from this repo -- CDLNet, GroupCDL, or a multigrid `MGCDLNet` --
plugged in as the proximal step; with `reuse_latent=True` its sparse code is
carried between layers so each z-update warm-starts from the previous one
instead of re-solving from z = 0.

Joint coil-map estimation (`smap_update=True`)
---------------------------------------------
After each layer the maps are refined by a regularized CG solve with a learned
high-pass penalty, and written back into the encoding operator so the *next*
x-update sees them.  With the image frozen:

    min_s  1/2 ||M F (x . s) - y||^2 + (mu/2) <s, W s> + (gamma/2) ||s - s_prev||^2
    <=>    (E_x^H E_x + mu W + gamma I) s = E_x^H y + gamma s_prev,
           E_x = M . F . diag(x)

which is one more Tikhonov CG, with gamma slotted into the Tikhonov argument.
The operator is threaded through the unroll functionally -- never mutated --
so the autograd graph of earlier layers stays valid.
"""

from __future__ import annotations

import copy
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prox import Polynomial
from operators import FFT2D, Identity, Mask
from operators.accessors import get_mask, get_sensitivity_map, set_sensitivity_map
from preprocessing.image import post_process, pre_process
from solvers.cg import tcg


# ===========================================================================
#  bounded reparameterisations
# ===========================================================================
def softplus_inv(t):
    """`raw` such that `softplus(raw) == t`, for t > 0."""
    return math.log(math.expm1(float(t)))


def clamp_stepsize(raw):
    return F.softplus(raw) + 1e-4          # > 0, so the system stays SPD


def clamp_mu(raw):
    return F.softplus(raw)


def clamp_sigma(raw, sigma_min):
    return F.softplus(raw) + sigma_min


# ===========================================================================
#  Learned isotropic Gaussian high-pass penalty
# ===========================================================================
class LearnedHighpass(nn.Module):
    """`W = (I - G_s)^H (I - G_s)` with a single learnable Gaussian width.

    The applied kernel is `delta - 2 G_s + G_{s sqrt 2}`, which is exactly
    `(I - G)^2` because a Gaussian convolved with itself widens by sqrt(2).  So
    `<s, W s> = ||(I - G) s||^2` is PSD by construction -- no extra machinery
    needed to keep the coil-map system solvable by CG.

    The kernel *size* is fixed at construction from `sigma_max` (the 5-sigma
    rule), not from the current sigma, so the shape never changes under
    autograd as the width trains; only the values are rebuilt when sigma moves.
    `||I - G|| <= 1` always, since a normalised Gaussian's DFT lies in (0, 1].

    Separable application
    ---------------------
    An isotropic Gaussian is an outer product of two 1-D Gaussians, so each blur
    is two 1-D passes rather than one `ks x ks` pass: 2*(2*ks) multiplies per
    pixel instead of `ks^2`, i.e. 60 vs 225 at ks=15 -- 3.75x fewer.  The
    `delta` term needs no convolution at all; it is the input.

    This is EXACT, not an approximation.  With zero padding, convolving rows
    then columns equals one 2-D convolution with the outer-product kernel: the
    intermediate is zero wherever it would be padded, because the input already
    was.  `tests/test_ladmm.py` checks it against the explicit 2-D kernel.

    Profiling motivated it: this layer was 57% of AltSplitCDLNet's forward pass,
    running 11 times per coil-map CG solve.
    """

    def __init__(self, sigma_g=1.5, sigma_max=3.0, sigma_min=0.3):
        super().__init__()
        assert sigma_min <= sigma_g <= sigma_max
        ks = int(round(5 * sigma_max - 1))
        self.ks = ks if ks % 2 == 1 else ks + 1
        self.sigma_min = float(sigma_min)
        self.sigma_raw = nn.Parameter(
            torch.tensor([softplus_inv(sigma_g - sigma_min)]))

        m = (self.ks - 1) // 2
        grid = torch.arange(-m, m + 1, dtype=torch.float32)
        # gx / gy are kept (rather than replaced by a 1-D buffer) so the
        # state_dict is unchanged; the separable path reads `gx[0]`.
        self.register_buffer("gy", grid.view(-1, 1).expand(self.ks, self.ks).clone())
        self.register_buffer("gx", grid.view(1, -1).expand(self.ks, self.ks).clone())
        delta = torch.zeros(1, 1, self.ks, self.ks)
        delta[0, 0, m, m] = 1.0
        self.register_buffer("impulse", delta)

        self._k_key = None                     # cache: (sigma version, grad mode)
        self._k_val = None

    @property
    def sigma_g(self):
        return clamp_sigma(self.sigma_raw, self.sigma_min)

    # -- kernels ------------------------------------------------------------
    def _gauss_1d(self, sigma):
        g = torch.exp(-(self.gx[0] ** 2) / (2.0 * sigma * sigma))
        return g / g.sum()

    def kernels_1d(self):
        """The two 1-D Gaussians, rebuilt only when sigma_g actually moves.

        Cached across the ~11 applies of a CG solve.  The key carries the
        parameter's version counter (bumped in place by `optimizer.step()`) AND
        the grad mode: a kernel built under `no_grad` has no graph, so handing
        it back inside an `enable_grad` region would silently drop sigma_g's
        gradient.
        """
        key = (int(self.sigma_raw._version), torch.is_grad_enabled())
        if self._k_key != key:
            s = self.sigma_g
            self._k_val = (self._gauss_1d(s), self._gauss_1d(s * math.sqrt(2.0)))
            self._k_key = key
        return self._k_val

    def kernel(self):
        """The explicit 2-D kernel. Not used by `forward` -- kept for
        inspection and as the reference the separability test compares to."""
        g1, g2 = self.kernels_1d()
        k = (self.impulse
             - 2.0 * torch.outer(g1, g1).reshape(1, 1, self.ks, self.ks)
             + torch.outer(g2, g2).reshape(1, 1, self.ks, self.ks))
        return k

    def gaussian(self, sigma):
        """The 2-D isotropic Gaussian, normalised. Outer product of the 1-D one,
        which is already unit-sum, so the 2-D one is too."""
        g = self._gauss_1d(sigma)
        return torch.outer(g, g).reshape(1, 1, self.ks, self.ks)

    # -- application --------------------------------------------------------
    def _blur(self, x, g):
        """Separable Gaussian blur: columns then rows, both zero-padded."""
        pad = (self.ks - 1) // 2
        x = F.conv2d(x, g.reshape(1, 1, -1, 1), padding=(pad, 0))
        return F.conv2d(x, g.reshape(1, 1, 1, -1), padding=(0, pad))

    def _apply_real(self, x, g1, g2):
        return x - 2.0 * self._blur(x, g1) + self._blur(x, g2)

    def forward(self, x):
        """Apply W channel-wise; coils are folded into the batch axis."""
        shape = x.shape
        xr = x.reshape(-1, 1, shape[-2], shape[-1])
        g1, g2 = self.kernels_1d()
        if torch.is_complex(xr):
            out = torch.complex(self._apply_real(xr.real, g1, g2),
                                self._apply_real(xr.imag, g1, g2))
        else:
            out = self._apply_real(xr, g1, g2)
        return out.reshape(shape)

    @torch.no_grad()
    def project_(self):
        pass                               # softplus already bounds sigma_g


class SmapNormalOp:
    """`s -> E_x^H E_x s + mu W s`; the CG matvec for the coil subproblem."""

    def __init__(self, Ex, W, mu):
        self.Ex, self.W, self.mu = Ex, W, mu

    def __call__(self, s):
        return self.Ex.adjoint(self.Ex(s)) + self.mu * self.W(s)


# ===========================================================================
#  Coil-map update layer
# ===========================================================================
class LSmapUpdate(nn.Module):
    """One regularized CG solve for the sensitivity maps, image held fixed."""

    def __init__(self, sigma_g=1.5, sigma_max=3.0, sigma_min=0.3, mu0=1.0,
                 gamma0=1e-2, cg_maxit=10, cg_tol=1e-3, cg_verbose=False,
                 implicit=False):
        super().__init__()
        self.W = LearnedHighpass(sigma_g=sigma_g, sigma_max=sigma_max,
                                 sigma_min=sigma_min)
        self.mu_raw = nn.Parameter(torch.tensor([softplus_inv(mu0)]))
        self.gamma_raw = nn.Parameter(torch.tensor([softplus_inv(gamma0)]))
        self.cg_kws = dict(max_iter=cg_maxit, tol=cg_tol, verbose=cg_verbose)
        self.implicit = bool(implicit)

    def forward(self, x, y, A, s_prev):
        mu = clamp_mu(self.mu_raw).view(1, 1, 1, 1)
        gamma = clamp_stepsize(self.gamma_raw).view(1, 1, 1, 1)
        Ex = Mask(get_mask(A)) @ FFT2D() @ Mask(x)
        rhs = Ex.adjoint(y) + gamma * s_prev
        G = SmapNormalOp(Ex, self.W, mu)
        params = (x, self.W.sigma_raw, self.mu_raw) if self.implicit else ()
        return tcg(G, gamma, rhs, params=params, implicit=self.implicit,
                   **self.cg_kws)


# ===========================================================================
#  denoiser plumbing
# ===========================================================================
def _accepts_z0(net):
    try:
        return "z0" in inspect.signature(net.forward).parameters
    except (TypeError, ValueError):
        return False


def denoise(net, x, sigma, z0=None):
    """Call any of this repo's denoisers uniformly; returns (x_hat, latent)."""
    if z0 is not None and _accepts_z0(net):
        out = net(x, E=Identity(), sigma=sigma, z0=z0)
    else:
        out = net(x, E=Identity(), sigma=sigma)
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


# ===========================================================================
#  LADMM layer / unroll
# ===========================================================================
class LADMMLayer(nn.Module):
    """One linearized-ADMM iteration: x-solve, learned prox, dual ascent."""

    def __init__(self, prox, reuse_latent=True, smap_update=False, chs=1,
                 rho0=1.0, degrees_rho=0, cg_maxit=10, cg_tol=1e-3,
                 cg_verbose=False, implicit_cg=False, smap_kws=None):
        super().__init__()
        self.prox = prox
        self.reuse_latent = bool(reuse_latent)
        self.smap_update = bool(smap_update)
        self.implicit_cg = bool(implicit_cg)
        self.rho = Polynomial(chs, degrees=degrees_rho, tau0=rho0)
        self.smap = (LSmapUpdate(cg_maxit=cg_maxit, cg_tol=cg_tol,
                                 cg_verbose=cg_verbose, implicit=implicit_cg,
                                 **(smap_kws or {}))
                     if smap_update else None)
        self.cg_kws = dict(max_iter=cg_maxit, tol=cg_tol, verbose=cg_verbose)

    # -- x-update -----------------------------------------------------------
    def _solve_x(self, A, rho, b):
        params = ()
        if self.implicit_cg:
            # the implicit backward needs every tensor A's Gram depends on
            try:
                smaps = get_sensitivity_map(A)
            except (ValueError, AttributeError):
                smaps = None
            if torch.is_tensor(smaps) and smaps.requires_grad:
                params = (smaps,)
        return tcg(A.gram, rho, b, params=params, implicit=self.implicit_cg,
                   **self.cg_kws)

    # -- forward ------------------------------------------------------------
    def forward(self, state, y, A, sigma=None, latent=None):
        """`state` is None on a cold start, else the previous `(x, Dz, u)`.

        `y` is the adjoint measurement `E^H y` when `smap_update=False`, and
        the raw whitened k-space when it is True -- in that case `E^H y` is
        recomputed each layer against the *current* coil maps.
        """
        rho = self.rho(sigma, ref=y)

        if state is None:
            b = A.adjoint(y) if self.smap_update else y
            x = self._solve_x(A, rho, b)
            Dz, latent = denoise(self.prox, x, sigma)
            u = x - Dz
        else:
            x_prev, Dz, u = state
            rhs = (A.adjoint(y) if self.smap_update else y) + rho * (Dz - u)
            x = self._solve_x(A, rho, rhs)
            Dz, latent = denoise(self.prox, x + u, sigma,
                                 z0=latent if self.reuse_latent else None)
            u = u + x - Dz

        if self.smap_update:
            A = set_sensitivity_map(A, self.smap(Dz, y, A, get_sensitivity_map(A)))
        return (x, Dz, u), A, latent

    @torch.no_grad()
    def project_(self):
        self.rho.project_(lo=0.0)


class LADMM(nn.Module):
    """K untied LADMM layers (the `prox` denoiser is untied per layer too)."""

    def __init__(self, K, layer_factory):
        super().__init__()
        self.K = int(K)
        proto = layer_factory()
        self.layers = nn.ModuleList([proto] + [copy.deepcopy(proto)
                                               for _ in range(self.K - 1)])

    def forward(self, y, A, sigma=None):
        state, latent = None, None
        for layer in self.layers:
            state, A, latent = layer(state, y, A, sigma=sigma, latent=latent)
        return state, A, latent


# ===========================================================================
#  AltSplitCDLNet wrapper
# ===========================================================================
class AltSplitCDLNet(nn.Module):
    """LADMM with a learned CDL denoiser as its proximal step.

    `forward(y, E, sigma) -> (x_hat, extras)` matching the repo's model
    interface; `extras` carries the latent code, the x / u iterates and the
    final coil maps.
    """

    def __init__(self, admm_iters=5, preproc="identity", reuse_latent=True,
                 smap_update=False, rho0=1.0, degrees_rho=0,
                 denoiser_type="mgcdlnet", denoiser_kws=None,
                 cg_maxit=10, cg_tol=1e-3, cg_verbose=False,
                 implicit_cg=False, smap_kws=None):
        super().__init__()
        self.K = int(admm_iters)
        self.preproc = preproc
        self.smap_update = bool(smap_update)

        def factory():
            denoiser = build_denoiser(denoiser_type, **(denoiser_kws or {}))
            return LADMMLayer(denoiser, reuse_latent=reuse_latent,
                              smap_update=smap_update, rho0=rho0,
                              degrees_rho=degrees_rho, cg_maxit=cg_maxit,
                              cg_tol=cg_tol, cg_verbose=cg_verbose,
                              implicit_cg=implicit_cg, smap_kws=smap_kws)

        self.net = LADMM(self.K, factory)

    def forward(self, y, E=None, sigma=None):
        if E is None:
            E = Identity()

        if self.smap_update:
            # the unroll needs raw k-space: E^H y is re-formed per layer
            y_in, params = y, None
        elif self.preproc == "identity":
            y_in, params = E.adjoint(y), None
        else:
            y_in, params = pre_process(E.adjoint(y), 1)

        (x, Dz, u), A, latent = self.net(y_in, E, sigma=sigma)
        x_hat = Dz if params is None else post_process(Dz, list(params))
        extras = dict(z=latent, x=x, u=u)
        if self.smap_update:
            extras["smaps"] = get_sensitivity_map(A)
        return x_hat, extras

    @torch.no_grad()
    def project(self):
        for m in self.modules():
            if hasattr(m, "project_"):
                m.project_()

    def extra_repr(self):
        return f"K={self.K}, smap_update={self.smap_update}"


def build_denoiser(name, **kws):
    """Denoiser factory for the LADMM prox slot (Julia's `select_network`)."""
    name = str(name).lower()
    if name in ("mgcdlnet", "mgcdl", "multigrid", "mggroupcdl"):
        from models.multigrid import MGCDLNet
        kws.setdefault("preproc", "image")
        return MGCDLNet(**kws)
    if name == "cdlnet":
        from models.cdlnet import CDLNet
        return CDLNet(**kws)
    if name == "groupcdl":
        from models.groupcdl import GroupCDL
        return GroupCDL(**kws)
    raise ValueError(f"unknown denoiser type '{name}'")
