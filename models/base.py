import contextlib

import torch
import torch.nn as nn
import numpy as np

from models.components import ComplexConvTranspose2d
from operators.projections import uball_project
from solvers.eigen import power_method


def _weight_is_property(module):
    """True when `.weight` is a derived view rather than the stored Parameter."""
    return isinstance(getattr(type(module), "weight", None), property)


def set_weight(module, W):
    """Write a conv weight, whichever way the module exposes it.

    The complex convs in `models/components.py` expose `.weight` as a PROPERTY
    that materialises `torch.complex(conv_real.weight.data, conv_imag.weight.data)`
    -- a fresh tensor, not a view of the storage.  So the natural-looking

        module.weight.data /= scale          # or .copy_(...), .mul_(...)

    silently modifies a temporary and is discarded.  Only the property SETTER
    reaches conv_real / conv_imag.  Raw `nn.Conv2d` is the opposite case: its
    `.weight` IS the Parameter, and assigning a plain tensor to it raises
    ("cannot assign 'torch.Tensor' as parameter 'weight'").  This helper picks
    the right path for either.
    """
    if _weight_is_property(module):
        module.weight = W                      # setter copies into real/imag
        return
    p = module.weight
    if not p.is_complex() and torch.is_complex(W):
        W = W.real
    p.data.copy_(W.to(p.dtype))


# ---------------------------------------------------------------------------
#  Unit-ball projection of conv filters, batched across a network
# ---------------------------------------------------------------------------
# One conv's projection is ~10 small kernels: materialise the complex weight,
# abs, square, sum, sqrt, reciprocal, clamp, multiply, and two copies back into
# conv_real / conv_imag. A [6,[4,4,6]] V-cycle has 216 convs, all with the same
# weight shape, and `project()` runs after every optimizer step: ~2,200 kernel
# launches, 37 ms per step on an A100 node against a 118 ms forward
# (scripts/time_net.py --step, 2026-10-08). The arithmetic is trivial; the
# launches are the cost.
#
# So an owner calls `project_conv(m)` instead of projecting on the spot. Inside
# `batched_projection()` the request is only recorded, and on exit every conv
# with the same weight shape is handled together: ONE stacked norm, then one
# in-place multiply per parameter. Outside the context `project_conv` projects
# immediately, exactly as before -- so a `project_()` called on its own (the
# older models do) is unaffected.
_BATCH = None


def _foreach_list_ok():
    """Does this torch take `_foreach_mul_(tensors, tensors)` with broadcasting?"""
    try:
        t = [torch.ones(2, 3)]
        torch._foreach_mul_(t, [torch.full((2, 1), 2.0)])
        return bool((t[0] == 2).all())
    except Exception:
        return False


_FOREACH_LIST = _foreach_list_ok()


def project_conv(module, dim=(2, 3)):
    """Project a conv's filters onto the unit ball (`uball_project` over `dim`).

    Deferred to the end of an enclosing `batched_projection()`; immediate
    otherwise. Only the complex/real pair convs of `models/components.py`
    (a `.weight` property over conv_real / conv_imag) are batched.
    """
    if (_BATCH is not None and _weight_is_property(module)
            and hasattr(module, "conv_real")):
        _BATCH.append((module, tuple(dim)))
        return
    set_weight(module, uball_project(module.weight, dim))


@contextlib.contextmanager
def batched_projection():
    """Collect `project_conv` requests and apply them together on exit."""
    global _BATCH
    if _BATCH is not None:                 # nested: the outermost one flushes
        yield
        return
    _BATCH = []
    try:
        yield
        pending, _BATCH = _BATCH, None     # flush with batching OFF
        _flush_projection(pending)
    finally:
        _BATCH = None


@torch.no_grad()
def _flush_projection(pending):
    groups, seen = {}, set()
    for m, dim in pending:
        wr = m.conv_real.weight
        wi = m.conv_imag.weight if getattr(m, "conv_imag", None) is not None else None
        if id(wr) in seen:                 # a conv (or a tied weight) asked twice
            continue
        seen.add(id(wr))
        key = (tuple(wr.shape), wr.dtype, wr.device, wi is not None, dim)
        groups.setdefault(key, []).append((m, wr, wi))

    for (_shape, _dtype, _device, cplx, dim), ms in groups.items():
        W = torch.stack([wr for _, wr, _ in ms])
        if cplx:
            W = torch.complex(W, torch.stack([wi for _, _, wi in ms]))
        # the same arithmetic as `uball_project`, one axis to the right
        norm = W.abs().pow(2).sum(dim=tuple(d + 1 for d in dim), keepdim=True).sqrt()
        scales = list(torch.clamp(1 / norm, max=1).unbind(0))    # real, per conv
        # W * scale with a REAL scale is (re * scale, im * scale): multiply the
        # two halves in place rather than rebuild the complex weight and copy
        # it back. In-place on the Parameter also bumps its version, which is
        # what the planar-weight cache keys on.
        targets = [wr for _, wr, _ in ms]
        factors = scales
        if cplx:
            targets = targets + [wi for _, _, wi in ms]
            factors = scales + scales
        if _FOREACH_LIST:
            torch._foreach_mul_(targets, factors)
        else:
            for t, f in zip(targets, factors):
                t.mul_(f)


class BaseUnrolledModel(nn.Module):
    """
    Shared initialization + projection logic
    for CDL / LPDS / variants.
    """
    @torch.no_grad()
    def project_filters(self):
        for k in range(self.K):
            set_weight(self.A[k], uball_project(self.A[k].weight))
            set_weight(self.B[k], uball_project(self.B[k].weight))

    def init_filters(self, dtype=torch.cfloat):
        W = torch.randn(self.M, self.C, self.P, self.P, dtype=dtype)
        for k in range(self.K):
            set_weight(self.A[k], W)
            set_weight(self.B[k], W.conj())

    @torch.no_grad()
    def spectral_init(self):
        """Scale (A, B) so that ||B A||_2 = 1, i.e. the ISTA step size is 1.

        Both factors are divided by sqrt(|L|), so the composition B A is scaled
        by 1/|L|.  Verified by `tests/test_init.py` for real AND complex nets --
        the complex path used to be a silent no-op (see `set_weight`), leaving
        ||B A|| in the hundreds and the unrolled step size correspondingly
        wrong.
        """
        DDt = lambda x: self.B[0](self.A[0](x))
        L = power_method(
            DDt,
            torch.rand(1, self.C, 128, 128, dtype=self.A[0].weight.dtype,
                       device=self.A[0].weight.device),
            num_iter=200,
            verbose=False,
        )[0]

        scale = np.sqrt(np.abs(L))

        print(f"Power method returns L = {L}")

        for k in range(self.K):
            set_weight(self.A[k], self.A[k].weight / scale)
            set_weight(self.B[k], self.B[k].weight / scale)
