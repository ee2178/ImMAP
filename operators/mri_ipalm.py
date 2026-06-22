import torch
from operators import Operator
from operators.fourier import fftc, ifftc

class MRIEncodingIPALM(Operator):
    
    # Defines a specific MRI Encoding operator for ipalmnet. We require that this operator have a grad_S method

    def __init__(self, mask):
        self.mask = mask

    def forward(self, x, smaps):
        # We now consider an operator with two variables x and smaps
        # Perform standard MRI encoding stuff
        return self.mask * fftc(smaps*x)

    def adjoint(self, x, smaps):
        return torch.sum(smaps.conj() * ifftc(self.mask*x), dim = 1, keepdim = True)

    def normal(self, x, smaps):
        # We overload the normal operator so that we can pass in x and smaps at the same time
        return torch.sum(smaps.conj() * ifftc(self.mask*fftc(smaps * x)), dim = 1, keepdim = True)

    def grad_S(self, x, y, smaps):
        # Additionally require 
        # Returns the gradient of the data consistency with respect to the sensitivity maps:
        return -x.conj()*ifftc() 
