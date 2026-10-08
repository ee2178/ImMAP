# -*- coding: utf-8 -*-
"""
The enhancement map S, decoded from the SAME sparse code as the target.

The additive model is  y = x - S  with x = CT1, y = T1 and S >= 0 the (sparse) enhancement. In a
synthesis-dictionary net the target is x = D z, so tying S to the same code z makes the T1
measurement LINEAR in z:

    free    S = D_S z                 y = (D - D_S) z
    gate    S = D (m . z)             y = D ((1 - m) . z)        m in [0, 1], one number per ATOM

`free` gives S its own synthesis dictionary: it is exactly a coupled dictionary for the pair
(T1, CT1), written so that D_S = 0 is "no enhancement" and the net starts at x = y.
`gate` builds S from the target's own atoms: m selects the subset of textures in the code that
are enhancement. Atoms with m ~ 0 are anatomy, pinned by T1; atoms with m ~ 1 are invisible to T1.

Either way the measurement fidelity 1/2 ||y - D_y z||^2 has an exact gradient -- a conv and its
transpose -- so this is a learned data-consistency term with no autograd inside the unrolling.

Untied form, as everywhere in the CDLNet family (A_k analysis, B_k synthesis of the target pair):

    free    g_y = (A_k - A_S,k) ( (B_k - B_S,k) z - y )
    gate    g_y = (1 - m) . A_k ( B_k ((1 - m) . z) - y )

and the readout is  S_hat = B_S,0 z  (free)  or  B_0 (m . z)  (gate).

Two collapses, both visible in the logs this module emits:
    D_S -> 0 / m -> 0     S = 0 and the T1 term forces x ~ T1 (no enhancement predicted)
    D_S -> D / m -> 1     D_y = 0: T1 constrains nothing and the net is a free synthesiser
"""

import copy

import torch
import torch.nn as nn

S_MODES = ("free", "gate")


class EnhancementCoupling(nn.Module):
    """Parameters and operators of the S model for a K-layer net.

    A_proto / B_proto are ONE analysis / synthesis module of the target pair; `free` deep-copies
    them (so geometry -- stride, padding, output_padding -- matches by construction) and zeroes
    the copies. `gate` holds a single per-atom vector shared by every layer.
    """

    def __init__(self, mode, K, M, A_proto, B_proto, gate_init=0.0):
        super().__init__()
        if mode not in S_MODES:
            raise ValueError(f"s_mode must be one of {S_MODES} (or None), got {mode!r}")
        self.mode, self.K, self.M = mode, int(K), int(M)
        if mode == "free":
            self.A_S = nn.ModuleList([copy.deepcopy(A_proto) for _ in range(self.K)])
            self.B_S = nn.ModuleList([copy.deepcopy(B_proto) for _ in range(self.K)])
            with torch.no_grad():
                for p in list(self.A_S.parameters()) + list(self.B_S.parameters()):
                    p.zero_()                     # D_S = 0: start at "no enhancement", x = y
        else:
            if not 0.0 <= float(gate_init) <= 1.0:
                raise ValueError(f"gate_init must be in [0, 1], got {gate_init}")
            self.m = nn.Parameter(torch.full((self.M,), float(gate_init)))

    def gate(self):
        return self.m.view(1, -1, 1, 1)

    def meas_grad(self, k, A, B, z, y):
        """Gradient of 1/2 ||y - D_y z||^2 w.r.t. z, with layer k's untied target pair (A, B)."""
        if self.mode == "free":
            res = B(z) - self.B_S[k](z) - y
            return A(res) - self.A_S[k](res)
        keep = 1.0 - self.gate()
        return keep * A(B(keep * z) - y)

    def decode_layer(self, k, B, z):
        """S from a code through layer k's pair -- the in-loop estimate (decode is the readout)."""
        return self.B_S[k](z) if self.mode == "free" else B(self.gate() * z)

    def decode(self, B0, z):
        """S_hat from the final code, through the readout synthesis B0 of the target pair."""
        return self.B_S[0](z) if self.mode == "free" else B0(self.gate() * z)

    @torch.no_grad()
    def project(self):
        # D_S is deliberately NOT projected: it is a learned difference of dictionaries, not a
        # dictionary that sets an ISTA step size (same reasoning as DTCDLNet's E_k).
        if self.mode == "gate":
            self.m.clamp_(0.0, 1.0)

    @torch.no_grad()
    def logs(self, Bs):
        """Parameter-side collapse indicators. `Bs` = the target pair's synthesis modules, one
        per layer (they are what D_S is compared against).

        gate    gate_closed -> 1 (or gate_max -> 0)   S = 0: nothing is enhancement
                gate_open -> 1                        T1 constrains nothing
        free    DS_over_D -> 0                        S = 0  (readout layer, and mean over layers)
                Dy_over_D -> 0                        D_y = D - D_S vanishes: T1 constrains nothing
                                                      (min over layers: ONE dead layer shows)
        """
        if self.mode == "gate":
            m = self.m
            return {"gate_mean": float(m.mean()), "gate_max": float(m.max()),
                    "gate_open": float((m > 0.5).float().mean()),
                    "gate_closed": float((m < 0.05).float().mean())}

        def norm(mod, other=None):
            ps = list(mod.parameters())
            qs = list(other.parameters()) if other is not None else [None] * len(ps)
            return float(sum(((p_ if q_ is None else p_ - q_) ** 2).sum()
                             for p_, q_ in zip(ps, qs)).sqrt())

        ds, dy = [], []
        for k, B in enumerate(Bs):
            n = max(norm(B), 1e-12)
            ds.append(norm(self.B_S[k]) / n)
            dy.append(norm(B, self.B_S[k]) / n)
        return {"DS_over_D_readout": ds[0], "DS_over_D_mean": sum(ds) / len(ds),
                "Dy_over_D_min": min(dy), "Dy_over_D_mean": sum(dy) / len(dy)}


def step_logit(n_fidelities):
    """Pre-sigmoid value that starts each of n fidelity steps at 1/n, so the COMBINED step is one
    ISTA step whatever n is (each pair is spectrally normalised to a unit step). n = 2 gives 0,
    the existing two-fidelity init; n = 1 is capped at 0.9."""
    p = min(1.0 / max(int(n_fidelities), 1), 0.9)
    return float(torch.logit(torch.tensor(p)))


def enhancement_target(ct1, t1):
    """What S is supervised against: the one-sided difference (CT1 - T1)_+."""
    return (ct1 - t1).clamp_min(0.0)


def enhancement_loss(net, ct1, t1, mask=None, use_mask=False):
    """MSE between the S map of the net's LAST forward and (CT1 - T1)_+.

    The net stores `last_S` on every forward (with its graph), so the trainers need no change to
    how they call the model. Raises if the net has no S model -- a weight on this term with a
    plain net would otherwise be silently ignored."""
    S = getattr(net, "last_S", None)
    if S is None:
        raise ValueError(f"s_weight > 0 needs a model with an enhancement map (s_mode 'free' or "
                         f"'gate'); {type(net).__name__} produced none.")
    S = S.real if torch.is_complex(S) else S
    err = (S - enhancement_target(ct1, t1)) ** 2
    if use_mask and mask is not None:
        w = (mask > 0.5).to(err.dtype)
        return (err * w).sum() / w.sum().clamp_min(1.0)
    return err.mean()


def enhancement_panel(S_hat, ct1, t1, mask=None):
    """The first sample's S, for a validation panel: (rgb, caption).

    rgb is (3, 3, H, W) -- target (CT1 - T1)_+ | S_hat | S_hat - target -- on ONE diverging
    blue-white-red scale set by the TARGET's peak (white = 0, red = positive). Tying the scale to
    the target means an S that is too weak looks pale and one that overshoots saturates; a
    per-panel scale would make both look right. `mask` blanks the panels outside it.
    """
    S = S_hat[:1, :1]
    S = (S.real if torch.is_complex(S) else S).float()
    tgt = enhancement_target(ct1[:1, :1], t1[:1, :1]).float()
    if mask is not None:
        w = (mask[:1, :1] > 0.5).float()
        S, tgt = S * w, tgt * w
    vmax = float(tgt.abs().amax().clamp_min(1e-8))

    def bwr(x):
        t = (x / vmax).clamp(-1.0, 1.0)
        pos, neg = t.clamp(min=0.0), (-t).clamp(min=0.0)
        return torch.cat([1.0 - neg, 1.0 - neg - pos, 1.0 - pos], dim=1)

    rms = lambda x: float(x.pow(2).mean().sqrt())                          # noqa: E731
    cap = ("enhancement map: (CT1 - T1)+ target | S_hat | S_hat - target  "
           f"[bwr, white=0, +-{vmax:.3f}]  rms(S_hat)/rms(target)={rms(S) / max(rms(tgt), 1e-12):.3f}  "
           f"min(S_hat)={float(S.min()):.3f}")
    return torch.cat([bwr(tgt), bwr(S), bwr(S - tgt)], dim=0), cap


def target_synthesis(net):
    """The target pair's synthesis modules, one per layer, for either S-map net."""
    return list(net.B_D) if hasattr(net, "B_D") else [layer.synthesis for layer in net.layers]


@torch.no_grad()
def collapse_param_logs(net):
    """Parameter-side collapse indicators of a net with an S model -> {name: float}.

    Beside the coupling's own (EnhancementCoupling.logs), the fidelity STEPS: a step at 0 means
    that term has been switched off, which is a collapse the dictionaries cannot show.
        step_xi    learned data consistency        step_eta   bridge target term
        step_nu    side contrasts
    Each is sigmoid of the constant term (mid-bridge), mean and min over the unrolled layers.
    """
    if getattr(net, "coupling", None) is None:
        return {}
    out = dict(net.coupling.logs(target_synthesis(net)))
    steps = [("xi", net.a_xi)]
    if net.bridge_fidelity:
        steps.append(("eta", net.a_eta))
    if net.n_p:
        steps.append(("nu", net.a_nu))
    for name, a in steps:
        v = torch.sigmoid(a[:, 0])
        out[f"step_{name}_mean"], out[f"step_{name}_min"] = float(v.mean()), float(v.min())
    g = getattr(net, "cross_gain", None)
    if g is not None:                        # -> 0: the enhancement cross-attention is switched off
        out["cross_gain_mean"], out["cross_gain_min"] = float(g.mean()), float(g.min())
    return out


class CollapseMeter:
    """Data-side collapse indicators, pooled over a validation pass.

        S_rms_ratio    rms(S_hat) / rms((CT1 - T1)_+)       -> 0: S collapsed to zero (1 = right size)
        S_neg_frac     share of S_hat's energy below zero   -> large: S is not an enhancement map
        T1_resid       rms(x_hat - S_hat - T1) / rms(T1 - mean)
                                                            -> 1: the model's T1 no longer explains T1
        code_density   fraction of the code that is nonzero -> 0: the code died (output = DC);
                                                               -> 1: no sparsity left
    `result(net)` adds the parameter-side indicators, so one dict holds everything. Ratios are of
    POOLED sums, not means of per-batch ratios, which empty-enhancement slices would dominate.
    """

    def __init__(self):
        self.s2 = self.t2 = self.neg2 = self.res2 = self.y2 = self.dens = 0.0
        self.n = 0

    @torch.no_grad()
    def add(self, net, x_hat, ct1, t1, mask=None):
        S = getattr(net, "last_S", None)
        if S is None:
            return
        S = (S.real if torch.is_complex(S) else S).float()
        x_hat = (x_hat.real if torch.is_complex(x_hat) else x_hat).float()
        w = (mask > 0.5).float() if mask is not None else torch.ones_like(S)
        tgt = enhancement_target(ct1, t1).float()
        t1c = t1 - t1.mean(dim=(1, 2, 3), keepdim=True)       # the DC the net carries separately
        self.s2 += float((S ** 2 * w).sum())
        self.t2 += float((tgt ** 2 * w).sum())
        self.neg2 += float((S.clamp(max=0.0) ** 2 * w).sum())
        self.res2 += float(((x_hat - S - t1) ** 2 * w).sum())
        self.y2 += float((t1c ** 2 * w).sum())
        d = getattr(net, "last_density", None)
        b = S.shape[0]
        if d is not None:
            self.dens += float(d) * b
        self.n += b

    def result(self, net=None):
        if not self.n:
            return {}
        eps = 1e-30
        out = {"S_rms_ratio": (self.s2 / max(self.t2, eps)) ** 0.5,
               "S_neg_frac": self.neg2 / max(self.s2, eps),
               "T1_resid": (self.res2 / max(self.y2, eps)) ** 0.5,
               "code_density": self.dens / self.n}
        if net is not None:
            out.update(collapse_param_logs(net))
        return out
