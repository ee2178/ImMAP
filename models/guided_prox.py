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
`gather` materialises the (B, Q, K) window values as a `Circulant` and is the
only backend that can do a joint softmax (a fused kernel normalises one
attention at a time; a joint simplex spans two differently-windowed ones) or
carry complex features.  `flex` / `triton` are available for
`joint_softmax=False`, where each branch is an ordinary independent attention;
they skip the Alg.-4 adjacency blend for the same reason `GroupThreshold` does.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.circulant_attention import Circulant, _abs2
from models.circulant_similarity import circulant_similarity_window
from models.circulant_flex import FlexAdjacency
from models.circulant_triton import TritonAdjacency
from models.prox import GroupThreshold, Polynomial


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

        if self.joint_softmax and self.attn_backend != "gather":
            raise ValueError(
                f"joint_softmax=True needs attn_backend='gather': the joint "
                f"simplex spans the self window ({self.window}) AND the guide "
                f"window ({self.guide_window}) together, and a fused attention "
                f"kernel normalises exactly one of them. Use "
                f"attn_backend='gather', or joint_softmax=False if you want the "
                f"fused backends (each branch then gets its own softmax and the "
                f"learned blend omega).")

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
        return FlexAdjacency(q, k, win, sim=self.sim_fun, heads=self.nheads,
                             block_mask=self._flex_block_mask(q),
                             compiled=self._flex_fn)

    def _build_adjacencies(self, z, guides, sigma):
        """`(Phi, [Omega_g])`, normalised jointly or independently."""
        q, k_self, k_guides = self._project_qk(z, guides, sigma)

        if self.attn_backend != "gather":
            # Reached only when joint_softmax is False (the constructor rejects
            # the other combination), so every branch is an ordinary
            # independent attention.
            return (self._fused_branch(q, k_self, self.window),
                    [self._fused_branch(q, kg, self.guide_window)
                     for kg in k_guides])

        qh = self._to_heads(q)
        spatial = tuple(z.shape[-2:])
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

    # -- blend weight --------------------------------------------------------
    def _omega_map(self, z, sigma):
        """`omega` broadcast from per-head to the Mh channels of the energy."""
        w = self.omega(sigma, ref=z)                     # (1, nheads, 1, 1)
        if self.nheads > 1:
            width = (self.Mh if self.grouped else self.M) // self.nheads
            w = w.repeat_interleave(width, dim=1)
        return w

    # -- prox ----------------------------------------------------------------
    def forward(self, z, guide=None, sigma=None, cache=None):
        guides = as_guide_list(guide)
        if not guides:
            # Self-only view sharing every weight -- `GroupThreshold(ggt)`.
            return super().forward(z, sigma, cache)

        Phi, Omegas, cache = self.adjacencies_of(z, guides, sigma, cache)

        za = self.Walpha(z) if self.grouped else z
        e_self = self.apply_gamma(Phi, _abs2(za))
        e_guide = None
        for Om, g in zip(Omegas, guides):
            ga = self.Walpha(g) if self.grouped else g
            term = self.apply_gamma(Om, _abs2(ga))
            e_guide = term if e_guide is None else e_guide + term

        if not self.joint_softmax:
            w = self._omega_map(z, sigma)
            e_self = w * e_self
            e_guide = (1.0 - w).abs() * e_guide

        xi_a = torch.sqrt(e_self + e_guide + self.eps)
        xi = self.beta_apply(xi_a) if self.grouped else xi_a
        tau = self.tau(sigma, ref=z)
        return z * F.relu(1.0 - tau / (xi + self.eps)), cache

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
