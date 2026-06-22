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
        """
        H, W = x.shape[-2], x.shape[-1]

        # Return cached window if spatial dims match
        if self.window_cache is not None and self.window_cache.shape[-2:] == (H, W):
            return self.window_cache

        # Build centered frequency grids in [-0.5, 0.5)
        fy = torch.fft.fftfreq(H, device=x.device).view(H, 1)  # (H, 1)
        fx = torch.fft.fftfreq(W, device=x.device).view(1, W)  # (1, W)

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
    
