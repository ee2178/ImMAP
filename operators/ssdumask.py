import torch
from operators.base import Operator
from physics.mask import gen_ssdu_mask
 
 
class SSDUMask(Operator):
    """
    Self-supervised data-undersampling mask.
 
    The mask is NOT regenerated automatically. Call `shuffle_mask(x)` once
    per training step (before the network applies this operator), so the
    same split is used for every forward/adjoint within a single
    reconstruction -- as standard SSDU requires. `forward` and `adjoint`
    just apply the current mask.
    """
 
    def __init__(self,
                 base_accel=1,
                 base_acs=20,
                 rho=(0.2, 0.2),
                 acs_lines=20,
                 device='cpu'):
        super().__init__()
        self.base_accel = base_accel
        self.base_acs = base_acs
        self.rho = rho
        self.acs_lines = acs_lines
        self.device = device          # was never stored in the original
        self.mask = None              # lazily set on first forward
 
    def forward(self, x):
        if self.mask is None:
            raise RuntimeError("SSDUMask.forward called before shuffle_mask; "
                               "no mask has been generated yet.")
        return self.mask * x
 
    def adjoint(self, y):
        if self.mask is None:
            raise RuntimeError("SSDUMask.adjoint called before shuffle_mask; "
                               "no mask has been generated yet.")
        return self.mask * y          # original referenced an undefined `x`
 
    def shuffle_mask(self, image):    # original was missing `self`
        # Call once per step. Only image.shape[-2:] (H, W) is used.
        self.mask = gen_ssdu_mask(
            image.shape[-2:],         # original used invalid image.shape[-2, -1]
            self.acs_lines,
            self.base_accel,
            self.base_acs,
            self.rho,
            device=self.device,
        )
 