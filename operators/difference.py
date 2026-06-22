import torch
import torch.nn as nn
import torch.nn.functional as F
from operators import Operator
from operators.padding import calc_pad_2d
from operators.fourier import fftc, ifftc

### IMPLEMENTATION OF TOTAL VARIATION DENOISING FOR SOLVING ARBITRARY LINEAR SYSTEMS

# First thing we need: the difference operator
class Diff2D(Operator):

    def __init__(self, device = 'cpu'):
        # Define the weight filters W_x, W_y
        W_x = torch.tensor([[[[1, -1], [0, 0]]]], device=device).to(torch.complex64)
        W_y = torch.tensor([[[[1, 0], [-1, 0]]]], device=device).to(torch.complex64)

        W = torch.cat([W_x, W_y], dim = 0)
        WH = torch.flip(torch.cat([W_x, W_y], dim = 1), dims = [2, 3])

        # Define our conv operator as a 1->2 channel conv layer. 
        self.D = nn.Conv2d(in_channels = 1, out_channels = 2, kernel_size = 2, bias = False)
        # Define our adjoint conv operator as a 2 -> 1 channel conv layer
        self.DH = nn.Conv2d(in_channels = 2, out_channels = 1, kernel_size = 2, bias = False)
        # Filling in weight tensors. But, we have to keep in mind that convolution in nn.Conv2d is actually spatial correlation, we need to flip our kernel!
        with torch.no_grad():
            self.D.weight.copy_(W)
            self.D.weight.requires_grad_(False)
            self.DH.weight.copy_(torch.flip(WH, dims = [2,3]))
            self.DH.weight.requires_grad_(False)

    def forward(self, x):
        '''
        Define a forward difference operator. We can implement the difference operator in several ways:
        1) Manual index shifting and subtraction (not so great in the 2D case)
        2) Circular convolution (by proper padding)
        3) Elementwise multiplication via fourier transform

        1 and 2 are essentially equivalent but we implement 2 
        '''
        # Here, we have chosen to implement our forward operator as a circular conv. 
        # We need to perform circular padding first - easy because our weight filters are 1x1x2x2
        x_pad = F.pad(x, (1,0,1,0), mode='circular')
        with torch.no_grad():
            return self.D(x_pad)

    def adjoint(self, x):
        '''
        We can similarly define the adjoint operator, which is just the sum of negative backwards difference operators in each direction.
        Otherwise, we can simply take the hermitian transpose of FW (much easier)
        '''
        # We implicitly assume here that we have consistent image dimensions as the forward operator
        # We need to perform circular padding first - adjust the padding to perform a "backwards difference"
        x_pad = F.pad(x, (0, 1, 0 ,1), mode='circular')
        with torch.no_grad():
            return self.DH(x_pad)

# However, we can very similarly define a difference operator that uses the Fourier transform interpretation of circular convolution:
class Diff2D_FFT(Operator):
    def __init__(self, device="cpu"):

        W_x = torch.tensor([[[[1, -1],
                              [0,  0]]]], device=device)

        W_y = torch.tensor([[[[1,  0],
                              [-1, 0]]]], device=device)

        self.W = torch.cat([W_x, W_y], dim=0)  # (2,1,2,2)

        self.device = device
        self.FW_cache = None

    def _get_FW(self, x):
        H, W = x.shape[-2:]

        if self.FW_cache is not None and self.FW_cache.shape[-2:] == (H, W):
            return self.FW_cache

        # pad kernel to image size
        W_pad = torch.zeros(2, 1, H, W, device=x.device)
        W_pad[:, :, :2, :2] = self.W

        # Fourier transform of operator kernel
        FW = torch.fft.fftn(W_pad, dim=(-2, -1))

        self.FW_cache = FW
        return FW

    def forward(self, x):
        X = torch.fft.fftn(x, dim=(-2, -1))
        FW = self._get_FW(x)

        Y = FW * X
        # To make this batch compatible, we need to have a 5D tensor, with the input being B x 1 x C x H x W, output takes 1 -> 2
        y = torch.fft.ifftn(Y, dim=(-2, -1)).real
        return y

    def adjoint(self, p):
        P = torch.fft.fftn(p, dim=(-2, -1))
        FW = self._get_FW(p)

        X = torch.sum(torch.conj(FW) * P, dim=-4, keepdim=True)

        x = torch.fft.ifftn(X, dim=(-2, -1))
        return x

    # Since we have the interpretation that the normal operator D^H D is actually very easy to compute, we can overwrite the usual normal operator to avoid unnecessary fffts. 

    def normal(self, x):
        X = torch.fft.fftn(x, dim = (-2, -1))
        FW = self._get_FW(x)
        return torch.fft.ifftn(torch.sum(FW.abs()**2*X, dim = 0, keepdim=True),dim = (-2, -1))
