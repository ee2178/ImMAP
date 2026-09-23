import torch
from operators import Operator
from operators.fourier import fftc, ifftc

class HighPassFilter(Operator):
    def __init__(self, sigma):
        self.sigma = sigma
        self.window_cache = None

    def get_window(self, x):
        """
        Build (or retrieve from cache) a high-pass window matching x's spatial dims.
        Returns a real-valued tensor of shape (1, 1, H, W) broadcastable over B and C.

        THE GRID MUST BE FFTSHIFTED. `forward` multiplies this window into `fftc(x)`, and
        `operators.fourier.fftc` is CENTERED (fftshift o fftn o ifftshift), so DC sits at index
        n // 2. `torch.fft.fftfreq` returns the UNSHIFTED order, with DC at index 0 -- so
        without the `fftshift` below the `1 - gaussian` notch lands at the array corner and the
        filter does the exact opposite of its name: measured at 32x32, sigma=0.2, a constant
        image came through at 0.998 of its amplitude and a Nyquist checkerboard at 0.000.
        tests/test_hpf.py pins the direction.
        """
        H, W = x.shape[-2], x.shape[-1]

        cached = self.window_cache
        if (cached is not None and cached.shape[-2:] == (H, W)
                and cached.device == x.device):
            return cached

        # Centered frequency grids in [-0.5, 0.5), DC in the MIDDLE to match fftc's output
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=x.device)).view(H, 1)  # (H, 1)
        fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=x.device)).view(1, W)  # (1, W)

        # Gaussian low-pass window centered at DC
        gaussian = torch.exp(-(fx**2 + fy**2) / (2 * self.sigma**2))  # (H, W)

        # High-pass = 1 - low-pass
        window = (1.0 - gaussian).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        self.window_cache = window
        return window

    def forward(self, x):
        """High-pass filter: attenuate low frequencies via 1 - Gaussian in Fourier domain."""
        X = fftc(x)                        # (B, C, H, W) complex
        window = self.get_window(x)        # (1, 1, H, W) real
        Y = X * window                     # elementwise multiply; window broadcasts
        return ifftc(Y)

    def adjoint(self, p):
        """
        Adjoint of the high-pass filter.
        Since the window is real-valued and symmetric, the filter is self-adjoint,
        so adjoint(p) == forward(p).
        """
        return self.forward(p)
    
