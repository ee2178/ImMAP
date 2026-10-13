# -*- coding: utf-8 -*-
"""
The fidelity step of the shared-code enhancement-map nets as TWO convolutions per layer.

A layer of SBCDLNet / SBGuidedGroupCDL with an S model (models/enhancement.py) takes

    u = z - xi g_y - eta g_D - nu g_P

    g_y = A_y ( B_y z - y )                  measurement (T1);  A_y, B_y the pair of D_y = D - D_S
    g_D = mu0 A ( mu0 B z - r )              bridge target term
    g_P = A_P ( B_P z - c_side )             side contrasts

Written module by module that is eight convolutions -- B z twice, B_S z, B_P z, then A and A_S
on the measurement residual, A on the bridge residual, A_P on the side residual -- and about a
dozen elementwise passes over the M-channel code to scale and combine them. Every one of those
convolutions has M channels on one side and one or two on the other, so its cost is reading or
writing the code, not arithmetic.

All of it is linear in the weights, so it is the same map as

    [B_y z, B z, B_P z] = conv_transpose( z, cat([W_By, W_B, W_BP], dim=1) )      ONE synthesis
    u = z - conv( cat([xi (B_y z - y), eta mu0 (mu0 B z - r), nu (B_P z - c_side)]),
                  cat([W_Ay, W_A, W_AP], dim=1) )                                 ONE analysis

with W_Ay = W_A - W_AS, W_By = W_B - W_BS (free) or keep . W_A, keep . W_B (gate) formed on the
WEIGHTS, which are (M, 1, P, P). The residuals and the step sizes live on the few image
channels. The code is read once and written once per layer.

Nothing about the model changes: the same parameters, in the same modules, with the same
gradients (to floating-point reassociation) -- a checkpoint loads either way. Complex, grouped
or biased convolutions are not handled and keep the module-by-module path.

IMMAP_FUSED_FIDELITY=0 in the environment (or `fused_fidelity.ENABLED = False`) restores the old
path, for an A/B timing or to rule this out.
"""

import os

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from models.components import Conv2d, ConvTranspose2d

ENABLED = os.environ.get("IMMAP_FUSED_FIDELITY", "1") != "0"


def plain_weight(mod):
    """The raw weight of a real, ungrouped, bias-free repo conv -- or None if it is not one."""
    if not isinstance(mod, (Conv2d, ConvTranspose2d)):
        return None
    if mod.complex or mod.groups != 1 or mod.conv_real.bias is not None:
        return None
    return mod.conv_real.weight


def fusable(A, B, *others):
    """Can these convs be stacked? `A` / `B` are an analysis / synthesis pair; every other module
    must be a plain conv with the geometry of whichever of the two it is the same kind as."""
    if not ENABLED or plain_weight(A) is None or plain_weight(B) is None:
        return False
    if not isinstance(A, Conv2d) or not isinstance(B, ConvTranspose2d):
        return False
    for m in others:
        ref = A if isinstance(m, Conv2d) else B
        if plain_weight(m) is None or (m.stride, m.padding) != (ref.stride, ref.padding):
            return False
        if plain_weight(m).shape[-2:] != plain_weight(ref).shape[-2:]:
            return False
    return True


class _Shrink(torch.autograd.Function):
    """`u - clamp(u, -tau, tau)` with components.ST's derivative written out.

    Autograd's own derivative of that expression is NOT ST's at tau = 0 -- clamp with min == max
    sends the whole gradient to `max` -- and tau = 0 is where these nets start (t0 = 0), so the
    threshold would get a biased gradient exactly when it matters. ST's is

        d out / d u   = 1{|u| > tau}
        d out / d tau = -sgn(u) 1{|u| > tau}

    and both are functions of the OUTPUT alone (out != 0 iff |u| > tau, and sgn(out) = sgn(u)
    there), so nothing but the output -- the next layer's code, alive anyway -- is saved. ST
    keeps |u|, the shifted modulus, the relu and the sign for its backward.
    """

    @staticmethod
    def forward(ctx, u, tau):
        out = torch.clamp(u, min=-tau, max=tau).neg_().add_(u)
        ctx.save_for_backward(out)
        ctx.tau_shape = tau.shape
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (out,) = ctx.saved_tensors
        gu = g * (out != 0) if ctx.needs_input_grad[0] else None
        gt = None
        if ctx.needs_input_grad[1]:
            gt = -(g * out.sign()).sum_to_size(ctx.tau_shape)
        return gu, gt


def shrink(u, tau):
    """Soft threshold of a REAL code, for tau >= 0 broadcastable against it.

    The same map and the same (sub)gradients as components.ST's `sgn(u) relu(|u| - tau)`,
    including at tau = 0 and at |u| = tau, in two passes over the code instead of five and
    with no intermediate kept for the backward (see _Shrink)."""
    if not torch.is_tensor(tau):
        tau = torch.as_tensor(tau, dtype=u.dtype, device=u.device)
    return _Shrink.apply(u, tau.to(u.dtype))


def s_gradient(z, k, A, B, coupling, y_meas, xi, bridge=None, side=None):
    """`xi g_y + eta g_D + nu g_P` for layer k, through one synthesis and one analysis.

    A, B       layer k's target pair (Conv2d, ConvTranspose2d)
    coupling   the EnhancementCoupling (its `meas_weights` gives the pair of D_y)
    bridge     None, or (mu0, r, eta)
    side       None, or (A_P[k], B_P[k], c_side, nu)
    xi, eta, nu, mu0 broadcast against (B, 1, H, W).
    """
    wa, wb = A.conv_real.weight, B.conv_real.weight
    wa_y, wb_y = coupling.meas_weights(k, wa, wb)

    syn = [wb_y]
    if bridge is not None:
        syn.append(wb)
    if side is not None:
        syn.append(side[1].conv_real.weight)
    out = F.conv_transpose2d(z, torch.cat(syn, dim=1) if len(syn) > 1 else syn[0],
                             stride=B.stride, padding=B.padding,
                             output_padding=B.output_padding)
    parts = torch.split(out, [w.shape[1] for w in syn], dim=1)

    res, ana = [xi * (parts[0] - y_meas)], [wa_y]
    if bridge is not None:
        mu0, r, eta = bridge
        res.append((eta * mu0) * (mu0 * parts[1] - r))
        ana.append(wa)
    if side is not None:
        A_P, _, c_side, nu = side
        res.append(nu * (parts[-1] - c_side))
        ana.append(A_P.conv_real.weight)
    if len(res) == 1:
        return F.conv2d(res[0], ana[0], stride=A.stride, padding=A.padding)
    return F.conv2d(torch.cat(res, dim=1), torch.cat(ana, dim=1),
                    stride=A.stride, padding=A.padding)
