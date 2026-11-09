"""
LGGS -- Longitudinally-Guided Group-Sparse reconstruction.

PyTorch port of Sljiva's `src/networks/guided_lpds.jl` (`GuidedGroupLPDSLayer`,
`GuidedLPDSNet`) on top of `models/lpds.py::LPDSLayer` and
`models/guided_prox.py::GuidedGroupThreshold`.

"LGGS" is not a class in the Julia source -- it is the NAME the experiments give
to one configuration of `guided_lpdsnet`, and `scripts/makeconfigs_longi.jl`
pins it down:

    network        = config/guided_lpdsnet.yaml     (network_type: lpdsnet)
    network.guided = true , multiguide = true , joint_softmax = true
    network.windowsize = 1 , guide_windowsize = 15
    similarity     = pidistance
    K = 30 , M = 169 , p = 7 , stride = 2 , init_strategy = semi_orthogonal
    data.guided    = true , num_guides in {1, 3, 5}

so: a learned primal-dual splitting network whose DUAL step is a guided group
threshold, with a trivial (1x1) self window and a 15x15 guide window normalised
in one joint softmax.  `lggs_defaults()` below returns exactly those keys.

The algorithm
-------------
One sweep, `w` the fully-sampled guide image(s):

    Bz  = B z
    x+  = x - tau (E^H E x - y~ + Bz)                primal gradient step
    xb  = x+ + theta (x+ - x)                        over-relaxation
    v   = A w                                        guide, THIS layer's dictionary
    z+  = prox_{g*}( z + A xb , v )                  guided group clipping

and the cold start is `x = y~`, `z = prox_{g*}(A y~, A w)`.

Two details that are easy to get wrong and are deliberate here:

* `v = A w` is recomputed every layer.  The layers are UNTIED (`LPDSNet`'s
  parameters are a `deepcopy` per layer in `lpds.jl`), so layer k's guide latent
  is layer k's dictionary applied to the guide -- not a value hoisted out of the
  loop.  Hoisting it would silently tie the guide to layer 0's dictionary.
* the guide is preprocessed with its OWN mean subtraction, on the same padded
  grid as `y~` (`preproc.jl`: `w_tilde, _ = preprocess(ImagePreprocess(stride,
  false), w)`).  It is never DC-matched to the measurement -- the whole premise
  is that the guide's intensities are a different contrast / timepoint and only
  its geometry transfers.

Why this is the network for longitudinally-guided contrast synthesis
--------------------------------------------------------------------
The guide enters ONLY through the adjacency: it decides which pixels are pooled
together in the group-sparsity penalty, and its own intensities are never added
to the estimate (see `models/guided_prox.py`).  A prior fully-sampled CT1 study
therefore contributes anatomy and lesion geometry while leaving enhancement to
be recovered from the measured contrasts -- which is exactly the failure mode a
naive "concatenate the prior as an extra input channel" model has, where a
model can copy the prior's enhancement forward.

`E` is optional and defaults to `Identity`, so the same class covers the
reconstruction setting (`E` = SENSE/Fourier) and the synthesis setting
(`E = Identity`, `y` the observed contrast stack).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.guided_prox import (GuidedGroupThreshold, ProjectedGuide,
                                as_guide_list, build_guided_prox)
from models.base import batched_projection
from models import compile_glue
from models.components import to_complex, to_planar
from models.lpds import LPDSLayer, _per_channel, gram
from models.sb_schedule import BridgeScheduleMixin
from operators.identity import Identity
from operators.padding import calc_pad_2d, unpad
from preprocessing.image import post_process, pre_process
from preprocessing.kspace import (_pad_complex, kspace_post_process,
                                  kspace_pre_process)


def _primal_step(x, data, Bz, tau, theta):
    """`x+ = x - tau (data + B z)`, `xb = x+ + theta (x+ - x)` -- pure, for `compile_glue`."""
    x_new = x - tau * (data + Bz)
    return x_new, x_new + theta * (x_new - x)


# ---------------------------------------------------------------------------
class GuidedLPDSLayer(LPDSLayer):
    """One primal-dual sweep whose dual prox is guided.

    Identical to `LPDSLayer` except for the prox slot and the extra `guide`
    argument threaded to it.  `guide` is in the IMAGE domain; this layer applies
    its own `analysis` to it, matching `guided_lpds.jl`.
    """

    def __init__(self, C, M, P=7, stride=1, lam0=1e-2, tau0=1e-1, theta0=1e-1,
                 degrees=0, is_complex=True, window=1, guide_window=None,
                 joint_softmax=True, Mh=None, prox_kws=None):
        # Build the plain layer first (convs, tau, theta, and a prox we replace)
        # so the analysis/synthesis construction and init stay in one place.
        super().__init__(C, M, P=P, stride=stride, lam0=lam0, tau0=tau0,
                         theta0=theta0, degrees=degrees, is_complex=is_complex,
                         window=1, Mh=None, prox_kws=None)

        prox_kws = dict(prox_kws or {})
        prox_kws.setdefault("tau0", lam0)
        prox_kws.setdefault("degrees", degrees)
        self.prox = build_guided_prox(M, Mh=Mh, window=window,
                                      guide_window=guide_window,
                                      joint_softmax=joint_softmax, dual=True,
                                      **prox_kws)

    # -- guide latents ------------------------------------------------------
    def analyse_guides(self, guides):
        """`[A w_g]` -- this layer's dictionary applied to each guide image."""
        return [self.analysis(w) for w in guides]

    def guide_latents(self, guides, planar=False):
        """What the prox gets for each guide: `A w`, as a `ProjectedGuide` where the composition
        with the prox's 1x1 transforms is expressible (always, for LGGS), else as the tensor.
        Same values either way; the projected form never builds the M-channel latent.
        `planar`: in the planar state's [re; im] form, to match a planar code."""
        gt = self.prox.prox
        if ProjectedGuide.supported(guides[0], self.analysis, gt):
            return [ProjectedGuide(w, self.analysis, gt, planar=planar) for w in guides]
        v = self.analyse_guides(guides)
        return [to_planar(t) for t in v] if planar else v

    def _zero_code(self, x):
        """The all-zero latent the analysis conv would produce from `x`: M channels (2M planar),
        on the strided grid, complex for a complex code."""
        B, Hz, Wz = _code_shape(self, x)
        planar = self.is_complex and not x.is_complex() and x.shape[1] == 2 * self.C
        ch = 2 * self.M if planar else self.M
        dtype = x.dtype
        if self.is_complex and not planar and not x.is_complex():
            dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        return torch.zeros(B, ch, Hz, Wz, dtype=dtype, device=x.device)

    # -- forward ------------------------------------------------------------
    def forward(self, state, y_tilde, guide=None, E=None, sigma=None, pi=None,
                cache=None):
        """One sweep.  `state` is `(x, z)`, or None for the cold start.

        `guide` is a list / stacked tensor of image-domain guides (or None, in
        which case the prox falls back to its unguided self-only form and this
        layer is an ordinary `LPDSLayer`).
        """
        if cache is None:
            cache = {}
        guides = as_guide_list(guide)

        hint = cache.pop("_gram_x", None)
        # state = (x, None): start the iteration at x instead of y~ (a bridge state, BridgeLGGS)
        x_init = None
        if state is not None and state[1] is None:
            x_init, state = state[0], None
        x_start = y_tilde if x_init is None else x_init
        x0 = x_start if state is None else state[0]
        # planar STATE (LGGSNet.planar_state_active): x, z arrive real, [re; im] stacked. The
        # convs and the prox take that form as it is; only the data term and the guides adapt.
        # Told apart by the channel count, not the dtype: a complex net is also fed REAL images
        # (a magnitude estimate, the learned-DC path), and those have C channels, not 2C.
        planar = (self.is_complex and not x0.is_complex()
                  and x0.shape[1] == 2 * self.C)
        v = self.guide_latents(guides, planar=planar) if guides else None

        if state is None and getattr(self, "z_init", "prox") == "prox":
            # cold start (lpds.jl): x = y~, z = GGT*(A y~) -- this layer makes no primal step
            x = x_start
            z, cache = self.prox(self.analysis(x), v, sigma, cache)
            return (x, z), cache

        if state is None:
            # z(0) = 0, x(0) = y~ (paper Eq. 13): this layer is a full sweep like every other
            x, z = x_start, self._zero_code(x_start)
        else:
            x, z = state
        pi_x, pi_z = (None, None) if pi is None else pi

        tau = self.tau(sigma, ref=x)
        theta = self.theta(sigma, ref=x)
        if planar:
            tau, theta = _per_channel(tau, x), _per_channel(theta, x)

        if getattr(E, "nonlinear", False):
            # learned operator: the data gradient is ONE call at the current iterate,
            # J_E(x)^H (E(x) - y). `E^H E x - E^H y` would need a point-free adjoint it lacks.
            data = to_planar(E.grad(to_complex(x))) if planar else E.grad(x)
        else:
            Ex = hint[1] if (hint is not None and hint[0] is x) else gram(E, x, planar)
            data = Ex - y_tilde
        if pi_x is None and compile_glue.enabled() and not x.is_complex():
            # the primal step and its over-relaxation as one compiled piece (models/compile_glue);
            # real tensors only (planar state, or a real net): inductor's complex support is thin
            x_new, x_bar = compile_glue.compiled(_primal_step)(x, data, self.synthesis(z),
                                                               tau, theta)
        else:
            residual = data + self.synthesis(z)
            if pi_x is not None:
                residual = residual - pi_x
            x_new = x - tau * residual

            x_bar = x_new + theta * (x_new - x)

        u = z + self.analysis(x_bar)
        if pi_z is not None:
            u = u - pi_z
        z_new, cache = self.prox(u, v, sigma, cache)

        return (x_new, z_new), cache


def _code_shape(layer, x):
    B, _, H, W = x.shape
    P, s = layer.P, layer.stride
    p = (P - 1) // 2
    return B, (H + 2 * p - P) // s + 1, (W + 2 * p - P) // s + 1


def make_guided_lpds_layer(C, M, spectral_init=True, **kws):
    """One `GuidedLPDSLayer`, filters tied (`B = A^H`) and spectrally scaled."""
    layer = GuidedLPDSLayer(C, M, **kws)
    layer.init_filters()
    if spectral_init:
        layer.spectral_normalize()
    return layer


def tie_transforms_(layers):
    """Point every layer's W_theta, W_phi, W_alpha, W_beta and gamma at layer 0's (in place).

    What Sljiva's LGGS runs share (config/guided_lpdsnet.yaml `share: {αβ: true, θϕ: true,
    γ: true, ρ: false}`, the April-2026 runs behind the paper's results) and what the paper
    states for the W's (arXiv 2610.22643, Sec. IV-A). eta, theta, A, B, tau and rho stay per
    layer. Assigning the submodule makes each parameter one object: the optimiser updates it
    once and `parameters()` counts it once (state_dict still lists it under every layer).
    """
    src = layers[0].prox.prox
    if not src.grouped:
        return layers
    for layer in layers[1:]:
        dst = layer.prox.prox
        dst.Wtheta, dst.Wphi, dst.Walpha, dst.Wbeta = src.Wtheta, src.Wphi, src.Walpha, src.Wbeta
        dst.gamma = src.gamma
    return layers


_TIED = ("Wtheta", "Wphi", "Walpha", "Wbeta", "gamma")


def _refuse_untied(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                   error_msgs):
    """Load pre-hook of LGGSNet: refuse a state dict whose layers hold DIFFERENT copies of a tied
    transform (an untied run) -- loading it would write every layer's copy into the one shared
    tensor, last wins. `prefix` is this net's position in the model being loaded ("" alone,
    "lggs." inside SBUnetLGGS, ...)."""
    head = prefix + "layers."
    ref = {}
    for k, v in state_dict.items():
        if not k.startswith(head):
            continue
        parts = k[len(head):].split(".")
        if len(parts) < 2 or not any(w in parts for w in _TIED):
            continue
        tail = ".".join(parts[1:])                       # drop the layer index
        if tail in ref and not torch.equal(ref[tail], v):
            raise ValueError(f"{prefix or 'LGGSNet.'}layers.*.{tail} differs between layers: "
                             f"this checkpoint was trained with untied transforms, which LGGS "
                             f"no longer has (retrain, or evaluate it from a pre-2026-11-09 "
                             f"commit)")
        ref.setdefault(tail, v)


class _PaddedDataGrad:
    """A learned operator's data gradient, seen from the PREPROCESSED grid.

    The layers iterate on x~ = pad(x - mean). E was trained on x itself, so the gradient is
    evaluated at unpad(x~) + mean and the pad band gets zeros (no data term lives there).
    """

    nonlinear = True

    def __init__(self, E, y, xmean, pad):
        self.E, self.y, self.xmean, self.pad = E, y, xmean, tuple(int(p) for p in pad)

    def grad(self, x):
        xr = (unpad(x, self.pad) if any(self.pad) else x) + self.xmean
        g = self.E.data_grad(xr, self.y)
        if not any(self.pad):
            return g
        if torch.is_complex(g):
            return torch.complex(F.pad(g.real, self.pad), F.pad(g.imag, self.pad))
        return F.pad(g, self.pad)


# ---------------------------------------------------------------------------
class LGGSNet(nn.Module):
    """K untied guided primal-dual sweeps -- the network the LGGS runs use.

    Returns `(x_hat, (x, z))`, the same pair `MGLPDSNet` returns: for a
    primal-dual method the state IS the pair, and handing back `z` alone would
    make a warm start half a warm start.

    Parameters follow `config/guided_lpdsnet.yaml`, renamed to this repo's
    spelling (`p -> P`, `stride -> s`, `lambda0 -> lam0`, `windowsize -> window`,
    `guide_windowsize -> guide_window`, `Delta K -> dK`).
    """

    def __init__(self, K=30, M=169, C=1, P=7, s=2, Mh=64, window=1,
                 guide_window=15, joint_softmax=True, sim_fun="pidistance",
                 lam0=1e-3, tau0=0.5, theta0=0.0, degrees=0, nheads=1, dK=5,
                 rho0=1.0, gamma0=0.8, rho_inv=True,
                 init_strategy="semi_orthogonal", attn_backend="gather",
                 flex_block_size=128, is_complex=True, preproc="kspace",
                 spectral_init=True, guide_blend=None, guide_gain0=1.0,
                 triton_materialize=False, z_init="prox",
                 guide_mean="per_guide", self_sim_fun=None, adj_blend=True,
                 similarity_scale="learned", prox_degrees=None):
        """W_theta, W_phi, W_alpha, W_beta and gamma are ONE set shared by the K layers (as in
        Sljiva's LGGS runs, `share:` in config/guided_lpdsnet.yaml, and the paper, Sec. IV-A);
        eta, theta, A, B, tau and rho stay per layer. There is no untied variant. A checkpoint
        whose layers hold different W's (trained untied, before 2026-11-09) is refused on load,
        also when this net is nested in another model (SBUnetLGGS): `_refuse_untied`.

        Paper-alignment switches (notes/lggs_paper_alignment.md; arXiv 2610.22643); every
        default is the port's behaviour:

        z_init            "prox": layer 0 is a cold start, z = GGT*(A y~) -- K - 1 sweeps.
                          "zero": z(0) = 0 and every layer is a full sweep (paper Eq. 13).
        guide_mean        "per_guide": each guide its own mean. "joint": one mean over all
                          N_g guide slices (paper Appendix B, mu_w = mean(w)).
        self_sim_fun, adj_blend, similarity_scale -> GuidedGroupThreshold (models/guided_prox).
        prox_degrees      polynomial degree in sigma of the prox THRESHOLD tau alone; None =
                          `degrees`, which otherwise sets the step sizes eta, theta too. Sljiva's
                          LGGS runs: a noise-adaptive threshold (`degrees: 1` reaches only the
                          GT's tau in src/networks/lpds.jl) with CONSTANT steps (degree 0) --
                          `degrees=0, prox_degrees=1`.
        """
        super().__init__()
        if z_init not in ("prox", "zero"):
            raise ValueError(f"z_init must be 'prox' or 'zero'; got {z_init!r}")
        if guide_mean not in ("per_guide", "joint"):
            raise ValueError(f"guide_mean must be 'per_guide' or 'joint'; got {guide_mean!r}")
        self.z_init = z_init
        self.guide_mean = guide_mean
        self.K, self.M, self.C, self.P, self.s = int(K), int(M), int(C), int(P), int(s)
        self.is_complex = bool(is_complex)
        self.attn_backend = attn_backend
        if preproc not in ("image", "kspace", "identity"):
            raise ValueError(
                f"preproc must be 'image' (denoising / synthesis), 'kspace' "
                f"(reconstruction) or 'identity'; got {preproc!r}")
        self.preproc = preproc
        self.pad_stride = self.s

        self.prox_degrees = int(degrees if prox_degrees is None else prox_degrees)
        prox_kws = dict(nheads=nheads, dK=dK, sim_fun=sim_fun, rho0=rho0,
                        degrees=self.prox_degrees,
                        gamma0=gamma0, rho_inv=rho_inv,
                        init_strategy=init_strategy, attn_backend=attn_backend,
                        flex_block_size=flex_block_size,
                        triton_materialize=triton_materialize, self_sim_fun=self_sim_fun,
                        adj_blend=adj_blend, similarity_scale=similarity_scale)
        if guide_blend is not None:            # "additive": models/guided_prox.py
            prox_kws.update(guide_blend=guide_blend, guide_gain0=guide_gain0)
        layer_kws = dict(P=P, stride=s, lam0=lam0, tau0=tau0, theta0=theta0,
                         degrees=degrees, is_complex=is_complex, window=window,
                         guide_window=guide_window, joint_softmax=joint_softmax,
                         Mh=Mh, prox_kws=prox_kws)

        proto = make_guided_lpds_layer(C, M, spectral_init=spectral_init,
                                       **layer_kws)
        self.layers = nn.ModuleList(
            [proto] + [copy.deepcopy(proto) for _ in range(self.K - 1)])
        for layer in self.layers:
            layer.z_init = z_init
        tie_transforms_(self.layers)
        # a pre-hook, not a load_state_dict override: it runs at any nesting depth (an
        # override is skipped when a parent model's load_state_dict recurses into this one)
        self._register_load_state_dict_pre_hook(_refuse_untied)

    # -- guide preprocessing -------------------------------------------------
    def _prep_guides(self, guide, ref):
        """Mean-subtract each guide and pad it onto `y~`'s grid.

        `preproc.jl` runs the guide through `ImagePreprocess` regardless of the
        network's own preprocessing mode, with `resize_noise=false` and its OWN
        mean -- the guide is a different contrast, so borrowing the
        measurement's DC would be wrong.  The pad is computed from the guide's
        shape, which equals the target's, so both land on the same grid; we
        assert that rather than trusting it.
        """
        guides = as_guide_list(guide)
        if not guides:
            return []
        out = []
        pad = calc_pad_2d(*guides[0].shape[-2:], self.pad_stride)
        joint = None
        if self.guide_mean == "joint":
            # one mean over all N_g guide slices (paper Appendix B: mu_w = mean(w))
            joint = torch.stack([w.to(ref.dtype) for w in guides], dim=1).mean(
                dim=(-4, -3, -2, -1), keepdim=True)[:, 0]
        for w in guides:
            w = w.to(ref.dtype)
            w = w - (w.mean(dim=(-3, -2, -1), keepdim=True) if joint is None else joint)
            if any(pad):
                w = _pad_complex(w, pad)
            if w.shape[-2:] != ref.shape[-2:]:
                raise ValueError(
                    f"guide grid {tuple(w.shape[-2:])} does not match the "
                    f"preprocessed measurement grid {tuple(ref.shape[-2:])}; "
                    f"guides must be registered to the target and share its "
                    f"field of view.")
            out.append(w)
        return out

    # -- forward -------------------------------------------------------------
    def forward(self, y, guide=None, E=None, sigma=None, state=None):
        if E is None:
            E = Identity()
        if getattr(E, "nonlinear", False):
            return self._forward_nonlinear(y, guide, E, sigma, state)

        if self.preproc == "kspace":
            y_tilde, E, params = kspace_pre_process(y, E, self.pad_stride)
            post = kspace_post_process
        else:
            x_adj = E.adjoint(y) if not isinstance(E, Identity) else y
            if self.preproc == "identity":
                y_tilde, params, post = x_adj, None, None
                self._check_grid(x_adj.shape[-2:])
            else:
                y_tilde, params = pre_process(x_adj, self.pad_stride)
                post = lambda x, p: post_process(x, list(p))    # noqa: E731

        guides = self._prep_guides(guide, y_tilde)
        x, z = self._sweeps(state, y_tilde, guides, E, sigma)
        x_hat = x if params is None else post(x, params)
        return x_hat, (x, z)

    def _sweeps(self, state, y_tilde, guides, E, sigma):
        """The K sweeps -> (x, z), complex in and out. Under the planar state the iterates are
        carried as real [re; im] tensors in between (the guides stay complex images: their
        projections come out planar, models/guided_prox.py::ProjectedGuide)."""
        planar = self.planar_state_active() and torch.is_complex(y_tilde)
        if planar:
            y_tilde = to_planar(y_tilde)
            if state is not None:
                state = tuple(None if t is None else to_planar(t) for t in state)
        cache = {}
        for layer in self.layers:
            state, cache = layer(state, y_tilde, guide=guides, E=E,
                                 sigma=sigma, cache=cache)
        x, z = state
        if planar:
            x, z = to_complex(x), to_complex(z)
        return x, z

    def planar_state_active(self):
        """Carry the iterates planar? The process-wide switch `MGLPDSNet.PLANAR_STATE`
        (`training.planar_state`, `IMMAP_PLANAR_STATE`; models/mg_lpds.py::set_planar_state),
        for a complex net. Changes no parameter and, to fp roundoff, no output or gradient."""
        from models.mg_lpds import MGLPDSNet
        return bool(MGLPDSNet.PLANAR_STATE and self.is_complex)

    def _forward_nonlinear(self, y, guide, E, sigma, state):
        """The same K guided sweeps with a LEARNED (nonlinear) data operator.

        Mirrors `CDLNet._forward_nonlinear`. Three things differ from the linear path, all
        forced by E having no point-free adjoint (operators/learned.py):

          * the start is `E.init(y)`, the operator's own image-domain estimate, not E^H y;
          * each sweep's data gradient is one `E.data_grad(x, y)` at the current iterate;
          * the image preprocessing (mean removal, stride padding) does not commute with a
            learned E, which was trained on real intensities at the true frame size. So the
            gradient is taken at unpad(x) + mean and zero-padded back -- the exact gradient
            w.r.t. the padded iterate, since the pad band carries no data term.

        `sigma` for the noise-adaptive prox is E.noise_level(y) when the operator defines one
        (BridgeDCOperator does), else the caller's.
        """
        if self.preproc == "kspace":
            raise ValueError("a learned operator acts in the image domain; use preproc='image' "
                             "or 'identity', not 'kspace'")
        y0 = E.init(y) if hasattr(E, "init") else y
        if self.preproc == "identity":
            self._check_grid(y0.shape[-2:])
            y_tilde, params = y0, None
            xmean, pad = 0.0, (0, 0, 0, 0)
        else:
            y_tilde, params = pre_process(y0, self.pad_stride)
            xmean, pad = params
        if hasattr(E, "noise_level"):
            sigma = E.noise_level(y)

        dc = _PaddedDataGrad(E, y, xmean, pad)
        guides = self._prep_guides(guide, y_tilde)
        x, z = self._sweeps(state, y_tilde, guides, dc, sigma)
        x_hat = x if params is None else post_process(x, list(params))
        return x_hat, (x, z)

    def _check_grid(self, hw):
        H, W = int(hw[0]), int(hw[1])
        if H % self.pad_stride or W % self.pad_stride:
            raise ValueError(
                f"{type(self).__name__} got a {H}x{W} input, not a multiple of "
                f"the dictionary stride s={self.s}. With preproc='identity' "
                f"nothing pads it, so the strided analysis / synthesis pair "
                f"would not land back on the input grid.")

    # -- hooks ---------------------------------------------------------------
    def compile_flex(self):
        """torch.compile every guided prox's fused kernel (GPU; call once)."""
        for m in self.modules():
            if isinstance(m, GuidedGroupThreshold) and m.attn_backend == "flex":
                m.compile_flex()
        return self

    @torch.no_grad()
    def project(self):
        """Constraint projection, each constrained module ONCE (as MGLPDSNet.project).

        Top-down: a module that defines `project_` is called and NOT descended into -- a layer
        projects its filters, its steps and its prox. Walking `self.modules()` instead called
        every layer, then the Fenchel wrapper, then the prox inside it, then each Polynomial
        again with default bounds: four passes over the same parameters every step. Clamps
        are idempotent, so the result is identical (tests/test_guided_speedups.py). Inside
        `batched_projection` the filters are projected together and shared attention
        transforms once.
        """
        def walk(mod):
            for child in mod.children():
                if hasattr(child, "project_"):
                    child.project_()
                else:
                    walk(child)

        with batched_projection():
            walk(self)

    def extra_repr(self):
        return (f"K={self.K}, M={self.M}, C={self.C}, s={self.s}, "
                f"preproc={self.preproc}, "
                f"z_init={self.z_init}, guide_mean={self.guide_mean}, "
                f"prox_degrees={self.prox_degrees}")


class BridgeLGGS(BridgeScheduleMixin, LGGSNet):
    """LGGS as an I2SB x0-regressor for reconstruction (task "i2sb_recon"; sb/i2sb_recon.py):

        x0_hat, _ = net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n, guide=guide_image)

    The same K guided primal-dual sweeps as LGGSNet on the measurement y (k-space preprocessing,
    the same data term), but STARTED AT THE BRIDGE STATE: the primal iterate begins at x_t (in
    the DC-removed frame of kspace_pre_process, x_t - mu) and layer 0 is the usual cold start
    z = GGT*(A x_t) from it. The noise level the thresholds see is

        sigma = sqrt(std_fwd^2 + sigma_n^2)     per sample

    the bridge's own noise on top of the measurement's -- SBUnetLGGS's convention -- so the
    noise-adaptive threshold (prox_degrees) widens toward t = 1. The guide is the fully
    sampled side image, as for LGGS; `guided=False` runs the sweeps self-only.

    Known maps and a complex output, like BridgeTGVN: the bridge state shares its phase
    reference with the operator (x0_source "operator"). The schedule params MUST match
    cfg["i2sb"] (assert_schedule_matches runs at startup)."""

    def __init__(self, kind="brownian", tau=0.1, n_points=1000, beta_max=0.3, guided=True,
                 **lggs):
        if lggs.get("preproc", "kspace") != "kspace":
            raise ValueError("BridgeLGGS reconstructs from k-space: preproc must be 'kspace'")
        if not lggs.get("is_complex", True):
            raise ValueError("BridgeLGGS needs is_complex=True: the bridge state is complex")
        super().__init__(**dict(lggs, preproc="kspace"))
        self.guided = bool(guided)
        self._init_bridge_tables(kind=kind, tau=tau, n_points=n_points, beta_max=beta_max)

    def forward(self, x_t, sigma=None, x1=None, y=None, E=None, sigma_n=None, guide=None,
                step=None):
        if y is None or E is None:
            raise ValueError("BridgeLGGS needs the measurement y and its operator E")
        if not torch.is_complex(x_t) or x_t.shape[1] != self.C:
            raise ValueError(f"expected a complex (B, {self.C}, H, W) bridge state, got "
                             f"{tuple(x_t.shape)} {x_t.dtype}")
        B = y.shape[0]
        if sigma is None:
            if step is None:
                raise ValueError("BridgeLGGS needs the bridge position: sigma (std_fwd) or step")
            st = torch.as_tensor(step, device=self.std_fwd.device).reshape(-1).long()
            sigma = self.std_fwd[st]
        s_b = torch.as_tensor(sigma, dtype=torch.float32, device=y.device).reshape(-1)
        s_b = s_b.expand(B) if s_b.numel() == 1 else s_b
        s_n = torch.zeros(B, device=y.device) if sigma_n is None else torch.as_tensor(
            sigma_n, dtype=torch.float32, device=y.device).reshape(-1).expand(B)
        sig = torch.sqrt(s_b ** 2 + s_n ** 2).reshape(B, 1, 1, 1)

        y_tilde, Ep, params = kspace_pre_process(y, E, self.pad_stride)
        mu, pad = params
        x0 = _pad_complex(x_t, pad) if any(pad) else x_t
        x0 = x0.to(y_tilde.dtype) - mu
        if x0.shape != y_tilde.shape:
            raise ValueError(f"bridge state {tuple(x0.shape)} is not on the measurement grid "
                             f"{tuple(y_tilde.shape)}")
        guides = self._prep_guides(guide if self.guided else None, y_tilde)
        x, _ = self._sweeps((x0, None), y_tilde, guides, Ep, sig)
        return kspace_post_process(x, list(params)), None


# Alias: the class is the port of `guided_lpds.jl`'s `GuidedLPDSNet`; `LGGSNet`
# is the name the experiments use for it.
GuidedLPDSNet = LGGSNet


def lggs_defaults(num_guides=1, **over):
    """The exact LGGS grid point from `scripts/makeconfigs_longi.jl`.

    `num_guides` is a DATA-side setting (how many prior studies the loader
    stacks), not a network hyper-parameter -- the same weights handle any
    number of guides, since every guide contributes one adjacency built from
    the shared W_phi.  It is returned here only so a config can carry it.
    """
    cfg = dict(K=30, M=169, C=1, P=7, s=2, Mh=64, window=1, guide_window=15,
               joint_softmax=True, sim_fun="pidistance", lam0=1e-3, tau0=0.5,
               theta0=0.0, degrees=0, prox_degrees=1, dK=5, nheads=1,
               init_strategy="semi_orthogonal",
               is_complex=True, preproc="kspace")
    cfg.update(over)
    return cfg, int(num_guides)
