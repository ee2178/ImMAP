import torch
from operators.base import Operator

class Sense(Operator):

    def __init__(self, smaps):
        self.smaps = smaps

    def forward(self, x):
        return self.smaps * x

    def adjoint(self, y):
        return torch.sum(
            torch.conj(self.smaps) * y,
            dim=1,
            keepdim=True,
        )


class SoftSense(Operator):
    def __init__(self, smaps):
        # smaps: (B, M, C, Nx, Ny)
        self.smaps = smaps

    def forward(self, x):
        # x: (B, M, 1, Nx, Ny) -> coil images (B, C, Nx, Ny)
        return torch.sum(self.smaps * x, dim=1)

    def adjoint(self, y):
        # y: (B, C, Nx, Ny) -> components (B, M, 1, Nx, Ny)
        return torch.sum(
            torch.conj(self.smaps) * y[:, None],
            dim=2,
            keepdim=True,
        )
