"""
Multilevel Learned Primal-Dual Splitting (ML-LPDS).

The ANALYSIS-form multilevel prior, solved by unrolled Condat-Vu:

    min_x  1/2 ||E x - y||^2  +  sum_{l=1..L} lambda_l || A_(1,l) x ||_1,
    A_(1,l) := A_l A_{l-1} ... A_1

`A_l` is a strided convolution from level l-1 to level l; level 0 is the image
(C channels), level l carries M_l channels on a grid coarsened by s_l.  Same
level schedule as `models/ml_cdlnet.py` (`level_channels`, `level_strides`).

How this differs from `models/ml_cdlnet.py`
-------------------------------------------
There the dictionaries GENERATE the image (x = D_1 ... D_L g_L), every level
sits on the reconstruction path, and the codes are coupled to one another --
which is where the sweep ordering, the read-out choice and the deepest-code
bottleneck all come from.  Here they PENALISE the image.  The penalty is
separable across levels, so its conjugate is too, and the saddle-point form

    min_x max_{z_1..z_L}  f(x) + sum_l [ <A_(1,l) x, z_l> - g_l*(z_l) ]

has no term coupling z_l to z_l'.  The duals interact only through x, so all L
of them update IN PARALLEL from the same x -- no ordering to choose.  And the
primal is the image itself: there is no read-out dictionary and no read-out
choice.

One layer (Algorithm 2 of the ML-LPDS record)
---------------------------------------------
    r_L = z_L ;  r_l = z_l + B_{l+1} r_{l+1}          up:   A^H z, Horner form
    x+  = x - tau (E^H E x - y~ + B_1 r_1)            primal gradient step
    xb  = x+ + theta (x+ - x)                         over-relaxation
    t_0 = xb ;  t_l = A_l t_{l-1}                     down: every tap of A xb
    z_l = clip_{lambda_l(sigma)}(z_l + t_l)           L independent clips

`B_l` is a `ConvTranspose2d` initialised at `A_l^H` and freed by training.  Each
of the two stacked operators costs ONE cascade -- L convolutions down, L up --
and E^H E is applied once, at the image grid.

With L = 1 this is `models/lpds.py::LPDSLayer` exactly, and the net is exactly
`MGLPDSNet` with an integer K; `tests/test_ml_lpds.py` asserts both.

Indexing of the cold start
--------------------------
`LPDSStack`'s convention is kept: layer 0 IS the cold start (x = y~, z_l =
clip(A_(1,l) y~)) and layers 1..K-1 are sweeps.  The record's Algorithm 2 writes
the cold start with A^(0) and then runs K sweeps starting from A^(0) again.
Keeping LPDSStack's indexing is what makes L = 1 reduce to the existing net
exactly, with the same parameter count, so the L = 1 cell is a true baseline.

Like `LPDSStack`, two sets of parameters get no gradient.  Layer 0's B_l, tau
and theta are never used, because the cold start makes no primal step.  The
last layer's theta, A_l and prox only feed a dual the output never reads.

Step sizes
----------
Every level is spectrally normalised at init (||B_l A_l|| = 1, B_l = A_l^H, so
||A_l|| = 1), and the per-level dual step is absorbed into the filters the way
the repo already does.  Condat-Vu then converges when

    tau (1/2 + ||A||^2) <= 1,       ||A||^2 <= sum_l ||A_(1,l)||^2 <= L

under ||E|| <= 1.  That is the one real cost of depth: the admissible primal
step shrinks like 1/L.  `MLLPDSLayer.op_norm2` measures ||A||^2 exactly with
the two cascades, and `step_bound` turns it into the bound.  tau is NOT
clamped to it: once B_l != A_l^H the iteration no longer solves any saddle-point
problem and the bound stops being a guarantee.  It is a good initialisation and
a sensible target, and nothing more.

Noise
-----
sigma is passed unchanged to every level, as in `models/ml_cdlnet.py`, and a
spatial noise map is rejected there for the same reason (`_MLIO._check_sigma`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.base import set_weight
from models.lista import LISTALayer, gram
from models.lpds import LPDSStack
from models.ml_cdlnet import _MLIO, _init_size, level_channels, level_strides
from models.prox import Polynomial
from operators.identity import Identity
from solvers.eigen import power_method


# ===========================================================================
#  one resolution level
# ===========================================================================
class MLLPDSLevel(LISTALayer):
    """`A_l`, `B_l` (initialised at `A_l^H`) and the dual prox of level l.

    Subclasses `LISTALayer` only for how it builds the analysis, synthesis and
    prox, plus its `init_filters`, `spectral_normalize` and `project_`, the
    same way `MLSplitLevel` does.  `forward` is this level's dual step and
    nothing else.  The primal step belongs to `MLLPDSLayer`, because x lives
    on the image grid that all the levels share.
    """

    def __init__(self, C, M, **kws):
        kws.pop("multigrid", None)              # no FAS correction here
        kws.pop("dual", None)                   # always the Fenchel prox
        super().__init__(C, M, multigrid=False, dual=True, **kws)

    def forward(self, z, t, sigma=None, cache=None):
        """`z_l <- prox_{g_l*}(z_l + t_l)`, a clip; `z = None` is the cold start."""
        return self.prox(t if z is None else z + t, sigma, cache)


# ===========================================================================
#  one unrolled Condat-Vu iteration over L levels
# ===========================================================================
class MLLPDSLayer(nn.Module):
    """One ML-LPDS sweep.  With `L = 1` this is `LPDSLayer` exactly.

    Signature-compatible with `LPDSLayer` (`(state, y_tilde, E, sigma, pi,
    cache) -> ((x, z), cache)`), so `LPDSStack` holds these unchanged.  `z` is a
    1-indexed list `[None, z_1, ..., z_L]`, as in `models/ml_cdlnet.py`.
    """

    def __init__(self, C, M, L, widen=1, P=7, s=1, Mh=None, window=1,
                 lam0=1e-2, tau0=1e-1, theta0=1e-1, degrees=0, is_complex=True,
                 prox_kws=None, spectral_init=True, init_norm="cascade",
                 proj_mode="slice"):
        super().__init__()
        self.C, self.L = int(C), int(L)
        self.is_complex = bool(is_complex)
        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = level_strides(s, self.L)

        levels = []
        for l in range(1, self.L + 1):
            # The group prox's attention width widens with the level, as in
            # `MLSweep`.
            Mh_l = None if Mh is None else max(1, int(round(Mh * widen ** (l - 1))))
            lev = MLLPDSLevel(self.Mch[l - 1], self.Mch[l], P=P,
                              stride=self.strides[l], tau0=lam0,
                              degrees=degrees, is_complex=is_complex,
                              window=window, Mh=Mh_l, prox_kws=prox_kws,
                              proj_mode=proj_mode)
            lev.init_filters()
            if spectral_init:
                lev.spectral_normalize(size=_init_size(l))
            levels.append(lev)
        self.levels = nn.ModuleList(levels)

        if init_norm not in ("level", "cascade"):
            raise ValueError(
                "init_norm must be 'level' or 'cascade'; got %r" % (init_norm,))
        self.init_norm = init_norm
        if spectral_init and init_norm == "cascade":
            self.normalize_cascade()

        # Per-PRIMAL-channel and noise-adaptive, exactly as in `LPDSLayer`.
        # There is one primal, on the image grid, so they are not per level.
        self.tau = Polynomial(C, degrees=degrees, tau0=tau0)
        self.theta = Polynomial(C, degrees=degrees, tau0=theta0)

    # -- the two cascades ----------------------------------------------------
    def analyse(self, x):
        """Down: `t_0 = x`, `t_l = A_l t_{l-1}`.  Returns `[None, t_1..t_L]`.

        `t_l = A_(1,l) x` by construction, so one cascade produces every tap
        of the stacked operator.
        """
        t, h = [None] * (self.L + 1), x
        for l in range(1, self.L + 1):
            h = self.levels[l - 1].analysis(h)
            t[l] = h
        return t

    def adjoint(self, z):
        """Up: `B_1 (z_1 + B_2 (z_2 + ... + B_L z_L))`, each dual injected at
        its own level.

        Equals `sum_l A_(1,l)^H z_l` while `B_l = A_l^H`, which is Horner's
        factorisation of the stacked adjoint.
        """
        r = z[self.L]
        for l in range(self.L - 1, 0, -1):
            r = z[l] + self.levels[l].synthesis(r)
        return self.levels[0].synthesis(r)

    # -- forward -------------------------------------------------------------
    def forward(self, state, y_tilde, E=None, sigma=None, pi=None, cache=None):
        if pi is not None:
            raise ValueError(
                "MLLPDSLayer takes no FAS correction: its levels are a "
                "multilevel PRIOR, not a V-cycle, so there is nothing to "
                "correct. Use models/mg_lpds.py for a multigrid solver.")
        if cache is None:
            cache = {}

        if state is None:
            # Cold start: x = y~, and the down cascade already fills every dual.
            x = y_tilde
            return (x, self._dual(None, self.analyse(x), sigma, cache)), cache

        x, z = state
        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)

        x_new = x - tau * (gram(E, x) - y_tilde + self.adjoint(z))
        x_bar = x_new + theta * (x_new - x)             # over-relaxation

        return (x_new, self._dual(z, self.analyse(x_bar), sigma, cache)), cache

    def _dual(self, z, t, sigma, cache):
        """All L clips, from the same x.  No level reads another's dual.

        Each level gets its own cache, threaded across layers.  A group prox's
        adjacency belongs to one grid and one channel count, so level 2 must
        not reuse level 1's Gamma.
        """
        out = [None] * (self.L + 1)
        for l in range(1, self.L + 1):
            key = "level%d" % l
            out[l], cache[key] = self.levels[l - 1](
                None if z is None else z[l], t[l], sigma=sigma,
                cache=cache.setdefault(key, {}))
        return out

    # -- constraints ---------------------------------------------------------
    @torch.no_grad()
    def project_(self):
        """tau >= 0, theta in [0, 1].

        Filters and proxes are projected by the levels' own `project_`, which
        the net's `project()` module walk reaches.
        """
        self.tau.project_(lo=0.0)
        self.theta.project_(lo=0.0, hi=1.0)

    # -- initialisation ------------------------------------------------------
    def _grid(self, size):
        """A grid every level divides exactly, at least `size` across."""
        m = self.strides[1] * 2 ** (self.L - 1)
        size = max(int(size), 4 * m)
        return size - size % m

    @torch.no_grad()
    def cascade_norm(self, l, size=64, num_iter=100):
        """`||A_(1,l)||_2` -- the gain of the cascade the algorithm applies.

        Exact while `B_j = A_j^H` (i.e. at init), since the power iteration
        uses the synthesis as the adjoint.  Afterwards it is the spectral
        radius of the learned pair, which is a diagnostic and nothing more.
        """
        w = self.levels[0].analysis.weight
        x0 = torch.rand(1, self.C, self._grid(size), self._grid(size),
                        dtype=w.dtype, device=w.device)

        def AtA(v):
            for j in range(1, l + 1):
                v = self.levels[j - 1].analysis(v)
            for j in range(l, 0, -1):
                v = self.levels[j - 1].synthesis(v)
            return v

        lam = power_method(AtA, x0, num_iter=num_iter, verbose=False)[0]
        return float(abs(lam)) ** 0.5

    @torch.no_grad()
    def normalize_cascade(self, size=64, num_iter=100):
        """Rescale each level so that `||A_(1,l)||_2 = 1` for every l.

        `spectral_normalize` sets `||A_l|| = 1` measured on GENERIC level-(l-1)
        input.  The operator this algorithm actually applies is the cascade
        `A_(1,l) = A_l ... A_1`, and for independently normalised factors that
        product is far BELOW 1, for two compounding reasons: the top singular
        directions of independently drawn operators do not align, and `A_l` is
        normalised against inputs it never receives (it only ever sees
        `A_(1,l-1) x`, which lives in the range of the previous cascade).

        Measured at init on the `mllpdsw2` shape (48/96/192, P=7, s=2), with
        every `||A_l|| ~ 0.99`:

            ||A_(1,1)|| = 0.99      ||A_(1,2)|| = 0.48      ...

        so level 2's dual already receives half the gain level 1's does, while
        `lam0` clips both against the same threshold -- and the level's push
        back into the primal carries `||A_(1,l)||` a second time.  The deep
        levels are near-inert before training starts.

        This walks the levels in order and divides level l by the cascade norm
        measured with levels 1..l-1 already fixed, so one pass makes every
        cascade unit gain.  Both `A_l` and `B_l` are scaled, which preserves
        `B_l = A_l^H`.

        The cost is the depth tax: `||A||^2` rises from ~1.1 to ~L, so the
        admissible Condat-Vu step falls from `1/(1/2 + 1.1)` to `1/(1/2 + L)`.
        Halve `tau0` when switching this on -- the levels are now doing
        something, and the step size has to pay for it.

        THIS REQUIRES proj_mode="slice", and that is why "slice" is the default.
        Scaling level l up by 1/g scales its filter norms by 1/g too, and the
        per-ATOM ball is tight enough to claw that straight back. Measured on
        the 48/96/192 config, cascade norms before -> after one `project()`:

            proj_mode="slice"   0.999 1.001 1.005  ->  0.998 1.001 1.006
            proj_mode="atom"    0.999 1.001 1.005  ->  0.998 0.599 0.377

        So the two options are not independent: per-atom undoes the cascade
        rescale, and the looseness of the per-slice ball is exactly the headroom
        this needs. Pair them only if you have re-measured.
        """
        for l in range(1, self.L + 1):
            g = self.cascade_norm(l, size=size, num_iter=num_iter)
            if not (g > 0) or g != g:                      # 0, inf or nan
                continue
            lev = self.levels[l - 1]
            set_weight(lev.analysis, lev.analysis.weight / g)
            set_weight(lev.synthesis, lev.synthesis.weight / g)

    # -- step-size diagnostics ----------------------------------------------
    @torch.no_grad()
    def op_norm2(self, size=64, num_iter=100):
        """`||A||^2` for the stacked analysis operator, by power iteration on
        `adjoint(analyse(.))`.

        Exact while `B_l = A_l^H`, i.e. at init.  After training it is the
        spectral radius of the learned cascade pair, which is a diagnostic
        and nothing more.
        """
        m = self.strides[1] * 2 ** (self.L - 1)
        size = max(int(size), 4 * m)
        size -= size % m
        w = self.levels[0].analysis.weight
        x0 = torch.rand(1, self.C, size, size, dtype=w.dtype, device=w.device)
        return abs(power_method(lambda x: self.adjoint(self.analyse(x)), x0,
                                num_iter=num_iter, verbose=False)[0])

    def step_bound(self, **kws):
        """The largest Condat-Vu primal step: `1 / (1/2 + ||A||^2)`, for
        `||E|| <= 1` and the dual step absorbed into the filters."""
        return 1.0 / (0.5 + self.op_norm2(**kws))

    def extra_repr(self):
        return "L=%d, channels=%s, strides=%s" % (
            self.L, self.Mch, self.strides[1:])


# ===========================================================================
#  the net
# ===========================================================================
class MLLPDSNet(_MLIO):
    """`preprocess -> K ML-LPDS layers -> postprocess`.

    Same interface as `MGLPDSNet`: `forward(y, E, sigma, state)` returns
    `(x_hat, (x, z))`, and `state` warm-starts from such a pair.  `K` counts
    layers the way `LPDSStack` does (layer 0 is the cold start), so
    `MLLPDSNet(K=K, L=1)` and `MGLPDSNet(K=K)` are the same network.
    """

    def __init__(self, K=30, L=2, M=169, C=1, P=7, s=2, widen=1,
                 lam0=1e-2, tau0=1e-1, theta0=1e-1, degrees=0,
                 is_complex=True, window=1, Mh=None, dK=1, sim_fun="distance",
                 nheads=1, rho0=1.0, gamma0=0.8, init_strategy="spectral_norm",
                 subgrad_mode="rigorous", attn_backend="gather",
                 flex_block_size=128, preproc="kspace", spectral_init=True,
                 init_norm="cascade", proj_mode="slice"):
        super().__init__()
        self.K, self.L = int(K), int(L)
        self.M, self.C, self.P, self.s = int(M), int(C), int(P), int(s)
        self.widen = widen
        self.is_complex = bool(is_complex)
        if self.L < 1:
            raise ValueError("L must be >= 1; got %r" % (L,))
        if self.K < 1:
            raise ValueError("K must be >= 1; got %r" % (K,))
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                "preproc must be 'image' (denoising), 'kspace' (reconstruction) "
                "or 'identity'; got %r" % (preproc,))
        self.preproc = preproc
        self.attn_backend = attn_backend
        self.init_norm, self.proj_mode = str(init_norm), str(proj_mode)

        self.Mch = level_channels(C, M, self.L, widen)
        self.strides = level_strides(s, self.L)
        # Every level must halve exactly, on the image AND the strided latent
        # grid. Same contract as MLCDLNet / MGLPDSNet.
        self.pad_stride = self.s * (2 ** (self.L - 1))

        prox_kws = dict(tau0=lam0, degrees=degrees, nheads=nheads, dK=dK,
                        sim_fun=sim_fun, rho0=rho0, gamma0=gamma0,
                        init_strategy=init_strategy, subgrad_mode=subgrad_mode,
                        attn_backend=attn_backend, flex_block_size=flex_block_size)

        # One prototype, spectrally normalised once and deep-copied K times,
        # so layer k starts as the same classical Condat-Vu step.
        self.net = LPDSStack(self.K, lambda: MLLPDSLayer(
            C, M, self.L, widen=widen, P=P, s=s, Mh=Mh, window=window,
            lam0=lam0, tau0=tau0, theta0=theta0, degrees=degrees,
            is_complex=is_complex, prox_kws=prox_kws,
            spectral_init=spectral_init, init_norm=init_norm,
            proj_mode=proj_mode))

    # -- forward -------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()
        self._check_sigma(sigma)
        y_tilde, E, params, post = self._pre(y, E)

        (x, z), _ = self.net(state, y_tilde, E=E, sigma=sigma, cache={})
        x_hat = x if params is None else post(x, params)
        return x_hat, (x, z)

    # -- diagnostics ---------------------------------------------------------
    def layer(self, k=-1):
        return self.net.layers[k]

    def step_bound(self, k=-1, **kws):
        """Condat-Vu's primal step bound for layer k.  See `MLLPDSLayer.step_bound`."""
        return self.layer(k).step_bound(**kws)

    @torch.no_grad()
    def cascade_norms(self, k=-1, **kws):
        """`[||A_(1,1)||, ..., ||A_(1,L)||]` for layer k.

        Under `init_norm='cascade'` these are all 1 at init; under `'level'`
        they decay with depth, which is the collapse `normalize_cascade`
        exists to fix.  After training they are a diagnostic only (`B` is no
        longer `A^H`).
        """
        lay = self.layer(k)
        return [lay.cascade_norm(l, **kws) for l in range(1, self.L + 1)]

    @torch.no_grad()
    def level_contributions(self, z, k=-1):
        """`[B_1 ... B_l z_l]` for l = 1..L: what each level pushes into the
        primal gradient, on the image grid.

        Their sum is `adjoint(z)`.  Their norms show which levels the trained
        prior actually uses, which is the evidence the channel-widening
        question needs.  Pass the `z` from `forward`'s `(x, z)`.
        """
        lay, out = self.layer(k), []
        for l in range(1, self.L + 1):
            h = z[l]
            for j in range(l, 0, -1):
                h = lay.levels[j - 1].synthesis(h)
            out.append(h)
        return out

    def extra_repr(self):
        return ("K=%d, L=%d, channels=%s, strides=%s, preproc=%r"
                % (self.K, self.L, self.Mch, self.strides[1:], self.preproc))
