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
