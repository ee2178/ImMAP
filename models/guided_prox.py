"""
Guided group thresholding -- the prox at the heart of LGGS.

PyTorch port of Sljiva's `src/networks/guided_gt.jl` (`GuidedGroupThreshold`),
built on top of `models/prox.py::GroupThreshold` so that every piece the two
share -- the `Polynomial` fields tau / gamma / rho, the four pixel-wise
transforms W_theta / W_phi / W_alpha / W_beta and their per-head init, the
head <-> batch folding, the constraint projection -- is literally the same code.

What "guided" changes
---------------------
`GroupThreshold` builds ONE adjacency, `Gamma = row-sm(sim(W_theta z, W_phi z))`,
from the latent to itself, so the group energy at pixel j pools |W_alpha z|^2
over j's own neighbourhood.  The guided prox adds one adjacency PER GUIDE,

    Phi     = sim( W_theta z , W_phi z   ; window       )      self  branch
    Omega_g = sim( W_theta z , W_phi v_g ; guide_window )      guide branch, g = 1..G

normalises them (see below), and pools both branches into one energy

    xi_a = sqrt( Phi (W_alpha z)^2  +  sum_g Omega_g (W_alpha v_g)^2 )
    xi   = W_beta xi_a
    GT(z, v) = z * relu(1 - tau / xi)

The guides `v_g` are latent-domain maps -- in LGGS they are the layer's own
analysis operator applied to a fully-sampled prior image (`guided_lpds.jl`:
`v = l.analysis(w)`), so the same dictionary sees the target and the guide.

Because the guide branch is what selects WHICH pixels group together, a
fully-sampled prior contributes its (undegraded) self-similarity structure to
the estimate without ever being added to it -- the estimate only inherits the
grouping, never the prior's intensities.  That is the property the longitudinal
setting wants: anatomy is shared across timepoints, contrast/pathology is not.

Normalisation: joint vs independent
-----------------------------------
`joint_softmax=True` (what every LGGS config uses) softmaxes across the
CONCATENATED neighbour axis of Phi and all Omega_g at once, so self and guide
weights compete inside a single simplex and their relative strength is learned
implicitly through the similarity.  `joint_softmax=False` gives each branch its
own row-softmax and blends them with a learned, noise-adaptive scalar
`omega in [0.05, 0.95]`:

    xi_a^2 = omega * Phi (W_alpha z)^2 + |1 - omega| * sum_g Omega_g (W_alpha v_g)^2

Note `window` and `guide_window` are independent.  The LGGS configs run
`windowsize=1` with `guide_windowsize=15`: a 1x1 self window is a single
neighbour, so under a joint softmax the self branch reduces to "this pixel's own
energy, competing against a 15x15 guide neighbourhood".  That is a legitimate
and deliberate configuration, not a degenerate one, so `window=1` is allowed
here (unlike `build_prox`, where `window=1` selects soft-thresholding).

Backends
--------
`gather` materialises the (B, Q, K) window values as a `Circulant`. It is the
only backend that carries the Alg.-4 adjacency BLEND (a convex combination of
materialised values) and complex features on every similarity.

`flex` / `triton` are fused: nothing of size (B, Q, K) is allocated and there is
no per-offset loop. Each branch is one windowed attention. They cover BOTH
normalisations:

  joint_softmax=False   independent branches, learned blend omega.
  joint_softmax=True    the SAME joint simplex as gather, without concatenating
      anything. A fused kernel normalises one attention at a time, but it also
      returns that attention's log-sum-exp, and a softmax over a union of
      windows factorises exactly:

          L_b   = log sum_{k in window b} exp(s_bk)                 (returned)
          pi_b  = softmax_b(L_b)                                    per query
          joint weight of (b, k) = pi_b * softmax_k(s_b)_k

      so the pooled energy is `sum_b pi_b * (Gamma_b v_b)` -- `merge_joint`.
      Every branch shares the query, so whatever per-query constant a backend
      leaves out of its scores cancels in pi.

Neither fused backend blends adjacencies across layers: on a refresh the new one
simply replaces the old (as in `GroupThreshold`). With dK = 1 that is the gather
model exactly; with dK > 1 it is the gather model minus the gamma blend, so a
checkpoint trained on gather does not transfer bit-for-bit.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.circulant_attention import Circulant, _abs2
from models.circulant_similarity import circulant_similarity_window
from models.circulant_flex import FlexAdjacency
from models.circulant_triton import TritonAdjacency
from models.prox import GroupThreshold, Polynomial


def merge_joint(outs, lses, nheads):
    """The joint-softmax energy from per-branch attention outputs and their log-sum-exps.

    outs : list of (B, C, H, W)  -- `Gamma_b v_b`, each branch softmaxed on its own
    lses : list, each (B, heads, S) or (B * heads, S) with S = H * W, row-major
    Returns `sum_b softmax_b(lse)_b * outs_b`, the head's weight spread over its channels.
    """
    B, C, H, W = outs[0].shape
    L = torch.stack([l.reshape(B, nheads, H, W) for l in lses], dim=0)     # (nb, B, h, H, W)
    pi = torch.softmax(L, dim=0)
    if C != nheads:
        pi = pi.repeat_interleave(C // nheads, dim=2)                      # heads are contiguous
    e = pi[0] * outs[0]
    for b in range(1, len(outs)):
        e = e + pi[b] * outs[b]
    return e


# One comparison against the gather backend per (backend, similarity, heads) per process:
# the fused joint softmax cannot be run on the CPU dev box, so its first use on a GPU checks
# itself. IMMAP_SKIP_FUSED_CHECK=1 skips it.
_FUSED_JOINT_CHECKED = set()


def as_guide_list(v):
    """Normalise the guide argument to a list of (B, C, H, W) tensors.

    Accepts `None` (no guide), a single tensor, a stacked `(B, G, C, H, W)`
    tensor, or a list/tuple of tensors.  Julia stacks guides along the batch
    axis and reshapes them out again (`vg = reshape(v, ..., :, batchsize)`);
    an explicit G axis says the same thing without the reshape convention.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if v.dim() == 5:
        return [v[:, g] for g in range(v.shape[1])]
    return [v]


class GuidedGroupThreshold(GroupThreshold):
    """`GroupThreshold` with one extra adjacency per guide.  See module docstring.

    Extra arguments over `GroupThreshold`:

    guide_window : int, odd
        Window side of the guide branches.  Defaults to `window`.
    joint_softmax : bool
        Normalise self + guide branches in ONE simplex (True, the LGGS setting)
        or independently with a learned blend `omega` (False).

    `forward` takes the guide explicitly: `prox(z, guide, sigma, cache)`.  With
    `guide=None` it falls back to `GroupThreshold.forward` -- the self-only view
    that shares every weight, which is what `guided_lpds.jl` reaches for via
    `GroupThreshold(ggt)` on the no-guide code path.
    """

    def __init__(self, M, guide_window=None, joint_softmax=True, **kws):
        super().__init__(M, **kws)
        self.guide_window = int(self.window if guide_window is None
                                else guide_window)
        assert self.guide_window % 2 == 1, "guide window side must be odd"
        self.joint_softmax = bool(joint_softmax)

        # joint softmax on flex / triton: exact, through the branches' log-sum-exps
        # (module docstring, `merge_joint`)
        self.joint_fused = self.joint_softmax and self.attn_backend != "gather"

        # Only the non-joint path has a blend to learn; the joint softmax makes
        # the branches compete directly, so `omega` would be redundant there
        # (`guided_gt.jl` installs a NoOpLayer in exactly that case).
        self.omega = None if self.joint_softmax else \
            Polynomial(self.nheads, degrees=0, tau0=0.5)

    # -- projections ---------------------------------------------------------
    def _rho_scale(self, x, sq):
        return x / sq if self.rho_inv else x * sq

    def _project_qk(self, z, guides, sigma):
        """Query from `z`, keys from `z` and each guide, all sharing one rho.

        `scaled_qk` in `group.jl` computes rho once per call; the guided
        forwards then reuse the SAME rho for every guide key (`_guided_key` in
        the flash variant makes this explicit).  Sharing it is what keeps the
        self and guide similarities on a common scale, which a joint softmax
        needs to mean anything.
        """
        rho = self.rho(sigma, ref=z)
        sq = torch.sqrt(rho + self.eps)
        q = self._rho_scale(self.Wtheta(z) if self.grouped else z, sq)
        k_self = self._rho_scale(self.Wphi(z) if self.grouped else z, sq)
        k_guides = [self._rho_scale(self.Wphi(g) if self.grouped else g, sq)
                    for g in guides]
        return q, k_self, k_guides

    # -- adjacencies ---------------------------------------------------------
    def _fused_branch(self, q, k, win):
        """One independently-normalised branch on the flex / triton backend."""
        if self.attn_backend == "triton":
            return TritonAdjacency(q, k, win, sim=self.sim_fun,
                                   heads=self.nheads,
                                   block_m=self.triton_block_m)
        q, k = self._stack_ri(q), self._stack_ri(k)
        # The mask must be THIS branch's window. It used to be the self window for every
        # branch, so with window != guide_window (LGGS: 1 vs 15) a flex guide branch attended
        # over the self window only.
        return FlexAdjacency(q, k, win, sim=self.sim_fun, heads=self.nheads,
                             block_mask=self._flex_block_mask(q, win),
                             compiled=self._flex_fn)

    def _build_adjacencies(self, z, guides, sigma):
        """`(Phi, [Omega_g])`, normalised jointly or independently."""
        q, k_self, k_guides = self._project_qk(z, guides, sigma)

        if self.attn_backend != "gather":
            # One independent windowed attention per branch. Under joint_softmax
            # they are merged afterwards through their log-sum-exps (forward).
            return (self._fused_branch(q, k_self, self.window),
                    [self._fused_branch(q, kg, self.guide_window)
                     for kg in k_guides])

        return self._gather_adjacencies(q, k_self, k_guides, tuple(z.shape[-2:]))

    def _gather_adjacencies(self, q, k_self, k_guides, spatial):
        qh = self._to_heads(q)
        s_phi, col_phi, crow_phi = circulant_similarity_window(
            self.sim_fun, qh, self._to_heads(k_self), self.window)
        branches = [circulant_similarity_window(
            self.sim_fun, qh, self._to_heads(kg), self.guide_window)
            for kg in k_guides]

        if s_phi.is_complex():
            raise ValueError(
                f"sim_fun={self.sim_fun!r} is complex-valued; the softmax needs "
                f"a real similarity (distance / realdot / pidot / pidistance).")

        if self.joint_softmax:
            # `CircAtt.joint_softmax(S_Phi, S_Omega_1, ...)`: one simplex over
            # the concatenated neighbour axis, split back afterwards.
            sizes = [s_phi.shape[-1]] + [b[0].shape[-1] for b in branches]
            cat = torch.cat([s_phi] + [b[0] for b in branches], dim=-1)
            parts = torch.split(F.softmax(cat, dim=-1), sizes, dim=-1)
            v_phi, v_omegas = parts[0], parts[1:]
        else:
            v_phi = F.softmax(s_phi, dim=-1)
            v_omegas = [F.softmax(b[0], dim=-1) for b in branches]

        Phi = Circulant(v_phi, col_phi, crow_phi, spatial, self.window)
        Omegas = [Circulant(v, b[1], b[2], spatial, self.guide_window)
                  for v, b in zip(v_omegas, branches)]
        return Phi, Omegas

    def adjacencies_of(self, z, guides, sigma, cache):
        """Fetch / build / blend `(Phi, [Omega_g])`, mirroring `gamma_of`.

        Same Alg.-4 schedule as the unguided prox -- built on the first call,
        rebuilt every `dK` layers as a convex blend with the previous pair,
        reused in between -- with the blend applied to the self and guide
        adjacencies alike (`guided_gt.jl` blends `Phi` and every `Omega_g` with
        the same gamma).  The fused backends have no materialised values to
        blend and re-cache outright.
        """
        if cache is None:
            cache = {}
        prev = cache.get("Phi")
        if prev is None:
            cache["Phi"], cache["Omega"] = self._build_adjacencies(
                z, guides, sigma)
            cache["gdupdate"] = 1
        elif cache.get("gdupdate", 0) % self.dK == 0:
            Phi_new, Om_new = self._build_adjacencies(z, guides, sigma)
            if isinstance(Phi_new, Circulant):
                g = self.gamma(sigma, ref=z).reshape(-1)          # (nheads,)
                g = g.repeat(z.shape[0]).view(-1, 1, 1)           # (B*h, 1, 1)

                def blend(old, new):
                    return old._like(old.values
                                     + g * (new.values - old.values))

                cache["Phi"] = blend(prev, Phi_new)
                cache["Omega"] = [blend(o, n)
                                  for o, n in zip(cache["Omega"], Om_new)]
            else:
                cache["Phi"], cache["Omega"] = Phi_new, Om_new
        cache["gdupdate"] = (cache.get("gdupdate", 0) + 1) % self.dK
        return cache["Phi"], cache["Omega"], cache

    # -- cross branch --------------------------------------------------------
    def _cross_adjacency(self, q_lat, k_lat, sigma):
        """One attention whose QUERY is not `z`: sim(W_theta q_lat, W_phi k_lat) over the guide
        window, with its OWN row-softmax. Same projections and rho as every other branch, so it
        adds no parameters; it stays out of the joint simplex because its query differs and the
        scores are not comparable with the self / guide ones."""
        rho = self.rho(sigma, ref=q_lat)
        sq = torch.sqrt(rho + self.eps)
        q = self._rho_scale(self.Wtheta(q_lat) if self.grouped else q_lat, sq)
        k = self._rho_scale(self.Wphi(k_lat) if self.grouped else k_lat, sq)
        if self.attn_backend != "gather":
            return self._fused_branch(q, k, self.guide_window)
        s, col, crow = circulant_similarity_window(
            self.sim_fun, self._to_heads(q), self._to_heads(k), self.guide_window)
        return Circulant(F.softmax(s, dim=-1), col, crow, tuple(q_lat.shape[-2:]),
                         self.guide_window)

    # -- blend weight --------------------------------------------------------
    def _omega_map(self, z, sigma):
        """`omega` broadcast from per-head to the Mh channels of the energy."""
        w = self.omega(sigma, ref=z)                     # (1, nheads, 1, 1)
        if self.nheads > 1:
            width = (self.Mh if self.grouped else self.M) // self.nheads
            w = w.repeat_interleave(width, dim=1)
        return w

    # -- prox ----------------------------------------------------------------
    def forward(self, z, guide=None, sigma=None, cache=None, cross=None):
        """`cross` (optional) adds one branch that attends between two OTHER latents:

            cross = {"query": fn() -> (B, M, H, W),   evaluated only when the adjacency refreshes
                     "guide": (B, M, H, W),           the keys, and the values whose energy is pooled
                     "weight": broadcastable to (B, M, 1, 1), >= 0}

            Xi   = row-softmax( sim( W_theta query , W_phi guide ; guide_window ) )
            xi^2 <- xi^2 + weight * ( W_beta sqrt( Xi (W_alpha guide)^2 ) )^2

        i.e. its group energy is ADDED per atom, scaled by `weight`, so it can only raise xi and
        therefore only relax the shrinkage of the atoms it is weighted onto. Like every guide it
        enters through the adjacency alone. Refreshed on the same dK schedule as Phi (rebuilt,
        not blended). cross=None is the prox exactly as before.
        """
        guides = as_guide_list(guide)
        if not guides and cross is None:
            # Self-only view sharing every weight -- `GroupThreshold(ggt)`.
            return super().forward(z, sigma, cache)

        refresh = (not cache) or cache.get("Phi") is None \
            or cache.get("gdupdate", 0) % self.dK == 0
        Phi, Omegas, cache = self.adjacencies_of(z, guides, sigma, cache)

        za = self.Walpha(z) if self.grouped else z
        e_self = self.apply_gamma(Phi, _abs2(za))
        terms = []
        for Om, g in zip(Omegas, guides):
            ga = self.Walpha(g) if self.grouped else g
            terms.append(self.apply_gamma(Om, _abs2(ga)))

        if self.joint_fused and terms:
            # the joint simplex, from each branch's own softmax and its log-sum-exp
            self._check_fused_joint(z)
            energy = merge_joint([e_self] + terms,
                                 [Phi._lse] + [Om._lse for Om in Omegas], self.nheads)
        else:
            e_guide = None
            for term in terms:
                e_guide = term if e_guide is None else e_guide + term
            if e_guide is None:                   # a cross branch with no ordinary guide
                e_guide = torch.zeros_like(e_self)
            elif not self.joint_softmax:
                w = self._omega_map(z, sigma)
                e_self = w * e_self
                e_guide = (1.0 - w).abs() * e_guide
            energy = e_self + e_guide

        xi_a = torch.sqrt(energy + self.eps)
        xi = self.beta_apply(xi_a) if self.grouped else xi_a
        if cross is not None:
            if refresh or "Xi" not in cache:
                cache["Xi"] = self._cross_adjacency(cross["query"](), cross["guide"], sigma)
            ga = self.Walpha(cross["guide"]) if self.grouped else cross["guide"]
            xa = torch.sqrt(self.apply_gamma(cache["Xi"], _abs2(ga)) + self.eps)
            xi_x = self.beta_apply(xa) if self.grouped else xa
            xi = torch.sqrt(xi ** 2 + cross["weight"] * xi_x ** 2)
        tau = self.tau(sigma, ref=z)
        return z * F.relu(1.0 - tau / (xi + self.eps)), cache

    # -- fused joint softmax: one self-check per process ----------------------
    @torch.no_grad()
    def _check_fused_joint(self, ref, size=None, tol=1e-3):
        """Compare the fused joint energy with the gather one on a small random problem.

        Runs the first time a (backend, similarity, heads, windows) combination is used in a
        process and raises if they disagree: a silently different attention would train
        without complaint. Returns the relative error (None when skipped)."""
        key = (self.attn_backend, self.sim_fun, self.nheads, self.window, self.guide_window,
               ref.is_complex())
        if key in _FUSED_JOINT_CHECKED or os.environ.get("IMMAP_SKIP_FUSED_CHECK") == "1":
            return None
        _FUSED_JOINT_CHECKED.add(key)
        n = size or (max(self.window, self.guide_window) + 5)
        gen = torch.Generator(device="cpu").manual_seed(0)

        def rnd():
            t = torch.randn(1, self.M, n, n, generator=gen)
            if ref.is_complex():
                t = torch.complex(t, torch.randn(1, self.M, n, n, generator=gen))
            return t.to(device=ref.device, dtype=ref.dtype)

        z, g = rnd(), rnd()
        q, k_self, k_guides = self._project_qk(z, [g], None)
        vals = [_abs2(self.Walpha(t) if self.grouped else t) for t in (z, g)]

        branches = [self._fused_branch(q, k_self, self.window),
                    self._fused_branch(q, k_guides[0], self.guide_window)]
        outs = [self.apply_gamma(A, v) for A, v in zip(branches, vals)]
        fused = merge_joint(outs, [A._lse for A in branches], self.nheads)

        Phi, Omegas = self._gather_adjacencies(q, k_self, k_guides, (n, n))
        want = self.apply_gamma(Phi, vals[0]) + self.apply_gamma(Omegas[0], vals[1])
        err = float((fused - want).abs().max() / want.abs().max().clamp_min(1e-30))
        if not err < tol:
            raise RuntimeError(
                f"fused joint softmax ({self.attn_backend}, sim={self.sim_fun}, "
                f"heads={self.nheads}, windows {self.window}/{self.guide_window}) disagrees "
                f"with the gather backend: relative error {err:.2e} on a {n}x{n} check. "
                f"Use attn_backend='gather', or set IMMAP_SKIP_FUSED_CHECK=1 to run anyway.")
        return err

    # -- subgradient ---------------------------------------------------------
    def subgradient(self, z, guide=None, sigma=None, cache=None, mode=None):
        """Moreau envelope `z - prox(z)`, always.

        `GroupThreshold`'s rigorous chain rule differentiates the group energy
        through Gamma; with a joint softmax the guide branch's weights depend on
        `z` through the SHARED denominator, so the same derivation picks up a
        cross term that has no counterpart in `mg_group.jl`.  Nothing in the
        LGGS family needs a subgradient (it is the V-cycle's FAS correction that
        does), so this stays the exact-and-cheap envelope rather than a
        half-derived formula that would silently be wrong inside a V-cycle.
        """
        zt, cache = self.forward(z, guide, sigma, cache)
        return z - zt, cache

    @torch.no_grad()
    def project_(self):
        super().project_()
        if self.omega is not None:
            self.omega.project_(lo=0.05, hi=0.95)

    def extra_repr(self):
        return (f"{super().extra_repr()}, guide_window={self.guide_window}, "
                f"joint_softmax={self.joint_softmax}")


class GuidedFenchelProx(nn.Module):
    """`prox_{g*}(z, v) = z - prox_g(z, v)`, the guided `FenchelProx`.

    `models/prox.py::FenchelProx` cannot be reused directly: it calls
    `self.prox(z, sigma, cache)`, and the guided prox needs the guide in that
    slot.  The identity is the same one -- Moreau's -- and it is what turns
    guided group thresholding into guided group CLIPPING, which is the map the
    LPDS dual step applies (`fenchel(l.prox, (z + Ax, v, sigma), ...)` in
    `guided_lpds.jl`).
    """

    def __init__(self, prox):
        super().__init__()
        self.prox = prox

    def forward(self, z, guide=None, sigma=None, cache=None):
        zt, cache = self.prox(z, guide, sigma, cache)
        return z - zt, cache

    def subgradient(self, z, guide=None, sigma=None, cache=None):
        # d g* telescopes to the inner prox, exactly as in `FenchelProx`.
        return self.prox(z, guide, sigma, cache)

    @torch.no_grad()
    def project_(self):
        self.prox.project_()


def build_guided_prox(M, Mh=None, window=1, guide_window=None,
                      joint_softmax=True, dual=True, **kws):
    """The guided prox, wrapped in its Fenchel conjugate for LPDS by default.

    Unlike `models/prox.py::build_prox` there is no `window > 1` switch: a
    guided prox is always a group prox, and `window=1` is the LGGS setting
    (a single self neighbour competing against the guide window), not a request
    for soft-thresholding.
    """
    prox = GuidedGroupThreshold(M, Mh=Mh, window=window,
                                guide_window=guide_window,
                                joint_softmax=joint_softmax, **kws)
    return GuidedFenchelProx(prox) if dual else prox
