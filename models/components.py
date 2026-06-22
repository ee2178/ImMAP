import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Nonlinearities
# ============================================================

def ST(x, t):
    """
    Soft-thresholding operator:
        ST(x, t) = sgn(x) * ReLU(|x| - t)
    """
    return x.sgn() * F.relu(x.abs() - t)


def CLIP(z, t):
    """
    Complementary clipping operator:
        CLIP(z, t) = z - ST(z, t)
    """
    return z - ST(z, t)

# ============================================================
# Complex-valued convolution block
# ============================================================
class _GaussConvNd(nn.Module):
    """Real/complex conv via Gauss's 3-multiply trick.

    Learnable params live in conv_real.weight / conv_imag.weight.
    Subclasses set up self.conv_real / self.conv_imag and define _op().
    """
    def __init__(self, complex=True):
        super().__init__()
        self.complex = complex

    # unified single-tensor view (derived; see weight setter note below)
    @property
    def weight(self):
        if self.complex:
            return torch.complex(self.conv_real.weight.data,
                                 self.conv_imag.weight.data)
        return self.conv_real.weight.data

    @weight.setter
    def weight(self, W):
        if self.complex:
            Wc = W if W.is_complex() else torch.complex(W, torch.zeros_like(W))
            self.conv_real.weight.data.copy_(Wc.real)
            self.conv_imag.weight.data.copy_(Wc.imag)
        else:
            self.conv_real.weight.data.copy_(torch.real(W))

    def _op(self, x, weight, bias=None):
        raise NotImplementedError

    def forward(self, x):
        if not self.complex:
            return self._op(x, self.conv_real.weight, self.conv_real.bias)

        wr, br = self.conv_real.weight, self.conv_real.bias
        wi, bi = self.conv_imag.weight, self.conv_imag.bias

        # complex weights, real input -> 2 ops (Gauss buys nothing here)
        if not x.is_complex():
            return torch.complex(self._op(x, wr, br), self._op(x, wi, bi))

        # complex weights, complex input -> Gauss 3-multiply trick
        x_r, x_i = x.real, x.imag
        t1 = self._op(x_r, wr, br)
        t2 = self._op(x_i, wi, bi)
        t3 = self._op(x_r + x_i, wr + wi, None if br is None else br + bi)
        return torch.complex(t1 - t2, t3 - t1 - t2)

class Conv2d(_GaussConvNd):
    def __init__(self, C, M, P, stride=1, bias=False, complex=True):
        super().__init__(complex=complex)
        self.padding = (P - 1) // 2
        self.stride = stride
        self.conv_real = nn.Conv2d(C, M, P, stride=stride,
                                   padding=self.padding, bias=bias)
        self.conv_imag = nn.Conv2d(C, M, P, stride=stride,
                                   padding=self.padding, bias=bias) if complex else None

    def _op(self, x, weight, bias=None):
        return F.conv2d(x, weight, bias=bias,
                        stride=self.stride, padding=self.padding)


class ConvTranspose2d(_GaussConvNd):
    def __init__(self, M, C, P, stride=1, bias=False, complex=True):
        super().__init__(complex=complex)
        self.padding = (P - 1) // 2
        self.output_padding = 1
        self.stride = stride
        self.conv_real = nn.ConvTranspose2d(M, C, P, stride=stride,
            padding=self.padding, output_padding=self.output_padding, bias=bias)
        self.conv_imag = nn.ConvTranspose2d(M, C, P, stride=stride,
            padding=self.padding, output_padding=self.output_padding, bias=bias) if complex else None

    def _op(self, x, weight, bias=None):
        return F.conv_transpose2d(x, weight, bias=bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding)



class ComplexConvTranspose2d(nn.Module):
    """
    Complex transpose convolution implemented via real/imag decomposition.

    Forward:
        (W_r + iW_i) * (x_r + ix_i)
    """

    def __init__(self, M, C, P, stride=1, bias=False):
        super().__init__()

        self.padding = (P - 1) // 2
        self.output_padding = 1

        self.conv_real = nn.ConvTranspose2d(
            M, C, P,
            stride=stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=bias
        )

        self.conv_imag = nn.ConvTranspose2d(
            M, C, P,
            stride=stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=bias
        )

    def forward(self, x):
        x_r, x_i = x.real, x.imag

        real = self.conv_real(x_r) - self.conv_imag(x_i)
        imag = self.conv_real(x_i) + self.conv_imag(x_r)

        return torch.complex(real, imag)

class ComplexConvTranspose2dGauss(nn.Module):
    """
    Complex transpose convolution using Gauss multiplication trick.

    Preserves:
        self.conv_real.weight
        self.conv_imag.weight

    so external code depending on those modules still works.
    """

    def __init__(self, M, C, P, stride=1, bias=False):
        super().__init__()

        self.padding = (P - 1) // 2
        self.output_padding = 1
        self.stride = stride

        # Keep these as actual modules for compatibility
        self.conv_real = nn.ConvTranspose2d(
            M,
            C,
            P,
            stride=stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=bias,
        )

        self.conv_imag = nn.ConvTranspose2d(
            M,
            C,
            P,
            stride=stride,
            padding=self.padding,
            output_padding=self.output_padding,
            bias=bias,
        )

    def _conv_transpose(self, x, weight, bias=None):
        return F.conv_transpose2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

    def forward(self, x):

        x_r = x.real
        x_i = x.imag

        wr = self.conv_real.weight
        wi = self.conv_imag.weight

        br = self.conv_real.bias
        bi = self.conv_imag.bias

        # 1
        t1 = self._conv_transpose(x_r, wr, br)

        # 2
        t2 = self._conv_transpose(x_i, wi, bi)

        # 3
        # Bias handling:
        # (br + bi) is required because:
        # (Wr + Wi)(xr + xi)
        t3 = self._conv_transpose(
            x_r + x_i,
            wr + wi,
            None if br is None else br + bi,
        )

        real = t1 - t2
        imag = t3 - t1 - t2

        return torch.complex(real, imag)

# =============================================================================
# Real-weight pixel-wise transform on (possibly) complex input
# =============================================================================
class RealPixelConvComplex(nn.Module):
    """1x1 convolution (a pixel-wise linear map) with REAL weights, applied to a
    possibly-complex input. A real matrix W acting on a complex feature map z is
    fully realizable: Wz = W·Re(z) + i W·Im(z). Storing W as a real Parameter
    (rather than a complex weight whose imaginary part would train freely) keeps
    the transform real for all of training, matches the paper's
    Wθ,φ,α ∈ R^{Mh×M}, and halves the parameters / multiplies of the transform.
 
    `.weight` is forwarded to the inner conv so init / inspection code that reads
    `self.Wtheta.weight` keeps working unchanged."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)   # real float32
 
    @property
    def weight(self):
        return self.conv.weight
 
    def forward(self, x):
        if torch.is_complex(x):
            return torch.complex(self.conv(x.real), self.conv(x.imag))
        return self.conv(x)


# ============================================================
# Learnable scalar function (used in LPDS schedules)
# ============================================================

class LearnablePolynomial(nn.Module):
    """
    Polynomial function:
        f(x) = sum_k a_k x^k

    Used for:
    - eta(sigma)
    - beta(sigma)
    """

    def __init__(self, coeffs):
        super().__init__()

        self.order = coeffs.numel() - 1
        self.coeffs = nn.Parameter(coeffs.clone().detach())

    def forward(self, x):
        x = x.reshape(-1)
        basis = torch.vander(x, N=self.order + 1, increasing=True)
        return basis @ self.coeffs
