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
            torch.rand(1, self.C, 128, 128, dtype=self.A[0].weight.dtype),
            num_iter=200,
            verbose=False,
        )[0]

        scale = np.sqrt(np.abs(L))

        print(f"Power method returns L = {L}")

        for k in range(self.K):
            set_weight(self.A[k], self.A[k].weight / scale)
            set_weight(self.B[k], self.B[k].weight / scale)
