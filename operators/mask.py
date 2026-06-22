from operators.base import Operator

class Mask(Operator):

    def __init__(self, mask):
        self.mask = mask

    def forward(self, x):
        return self.mask * x

    def adjoint(self, x):
        return self.mask * x
