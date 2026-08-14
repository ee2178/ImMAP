import torch

from operators.base import Operator


class Mask(Operator):
    """Elementwise multiplication by `mask`, i.e. diag(mask)."""

    # Lets `CompositeOperator` recognise a SENSE encoding and fuse its Gram
    # without importing this module.  See `operators/base.py::_match_sense_gram`.
    OP_KIND = "kspace_diag"

    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.mask * x

    def adjoint(self, x):
        # diag(m)^H = diag(conj(m)).  A no-op for the usual real sampling
        # masks, but required when the "mask" is a complex image -- which is
        # exactly how the coil-map subproblem builds  E_x = M F diag(x).
        m = self.mask
        if torch.is_tensor(m) and torch.is_complex(m):
            m = torch.conj(m)
        return m * x
