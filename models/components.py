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
COMPLEX_MODES = ("gauss", "planar")


def set_complex_mode(mode=None):
    """Set `_GaussConvNd.COMPLEX_MODE` for the process; None leaves it alone.

    Read from a config's `training.complex_conv` by train.py and the eval
    scripts. Changes no parameter -- both modes use conv_real / conv_imag --
    so a checkpoint trained under one evaluates under the other (they agree
    to ~4e-07 in fp32; see tests/test_planar_conv.py).
    """
    if mode is None:
        return _GaussConvNd.COMPLEX_MODE
    if mode not in COMPLEX_MODES:
        raise ValueError(f"complex_conv must be one of {COMPLEX_MODES}, got {mode!r}")
    _GaussConvNd.COMPLEX_MODE = mode
    return mode


def to_planar(x):
    """A complex `(B, C, H, W)` as real `(B, 2C, H, W)` = `[re; im]`.

    One pass.  `x.real` / `x.imag` are stride-2 views of the interleaved
    storage, so this is also the contiguous-ification `F.conv2d` would
    otherwise do twice.  A real input is returned unchanged.
    """
    if not torch.is_complex(x):
        return x
    return torch.cat((x.real, x.imag), dim=1)


def to_complex(xp):
    """Inverse of `to_planar` -- re-interleaves `[re; im]` into one complex tensor."""
    if torch.is_complex(xp):
        return xp
    n = xp.shape[1] // 2
    return torch.complex(xp[:, :n], xp[:, n:])


class _GaussConvNd(nn.Module):
    """Real/complex conv, either via Gauss's 3-multiply trick or as one real conv.

    Learnable params live in conv_real.weight / conv_imag.weight.
    Subclasses set up self.conv_real / self.conv_imag and define _op().

    Two formulations of the same map, selected by `COMPLEX_MODE`:

    "gauss"    three real convs on the de-interleaved halves, combined by
               subtraction (`t1 - t2`, `t3 - t1 - t2`).  Fewest FLOPs.
    "planar"   ONE real conv on the halves stacked along the channel axis,
               against the block weight that *is* the complex multiply:

                   [re_out]   [ wr  -wi ] [re_in]
                   [im_out] = [ wi   wr ] [im_in]

               i.e. `conv2d(cat([x.real, x.imag], 1), W)` with W of shape
               (2M, 2C, P, P).  Four convs' worth of FLOPs instead of three.

    Why "planar" is faster despite the extra FLOPs
    ----------------------------------------------
    At these shapes nothing here is compute-bound.  An L40S retires ~400 FLOPs
    per element in the time it takes to READ that element, and a P=7 conv does
    49 MACs, so even the convolutions are limited by memory traffic -- one
    elementwise pass over an M=169 latent costs about what the whole analysis
    conv costs.  "gauss" wraps its three convs in five such passes: `x.real`
    and `x.imag` are stride-2 views of an interleaved complex tensor, so each
    is copied contiguous before cuDNN sees it; then `xr + xi`; then the two
    combining subtractions; then `torch.complex` re-interleaves the result.
    "planar" pays one `cat` and one `torch.complex` and nothing else.

    On the brain grid (640x320, M=169, s=2) that is ~534 -> ~198 MB moved for
    an analysis conv and ~403 -> ~200 MB for a synthesis conv.  The remaining
    conversions are the complex<->planar round trip at the module boundary;
    they vanish only if the latent is CARRIED planar between layers, which is a
    change to the dtype contract of `LPDSLayer`'s state, not to this class.

    Numerics: "planar" is the better-conditioned of the two -- `t3 - t1 - t2`
    subtracts quantities of similar size, which "planar" never forms.  The two
    agree to ~4e-07 relative in fp32.  The default is "gauss" so that runs in
    flight keep their bit pattern; flip it once the A/B has been measured
    (`scripts/profile_mg.py --ab-planar`).

    Not every case is expressible: `groups > 1` would need the block weight to
    mix channels across group boundaries, and a bias is handled INCONSISTENTLY
    by "gauss" (it applies `br - bi` to the real part and drops `bi` from the
    imaginary part -- see `_planar_bias`).  Both fall back to "gauss" so that
    nothing silently changes answer; every conv in the LPDS / multigrid family
    is `groups=1, bias=False` and takes the planar path.
    """
    # Combine the Gauss trick's three convolutions in place when no autograd
    # graph is being built. Class-level so it can be flipped globally -- a kill
    # switch if it ever misbehaves, and what `scripts/profile_mg.py` toggles to
    # measure what it is worth.
    INPLACE_COMBINE = True

    # "gauss" (default, bit-compatible with every existing run's forward) or
    # "planar". Class-level for the same reason as above.
    COMPLEX_MODE = "gauss"

    # Set by the subclass: conv_transpose2d's weight is (in, out, ...) rather
    # than (out, in, ...), which transposes the block pattern.
    _PLANAR_TRANSPOSED = False

    # Keep the planar block weight between calls when no autograd graph is
    # being built (inference, validation). See `_planar_weight`. Class-level
    # like the switches above: a kill switch, and what an A/B run toggles.
    PLANAR_WEIGHT_CACHE = True

    def __init__(self, complex=True):
        super().__init__()
        self.complex = complex
        self._planar_cache = None

    def train(self, mode=True):
        # Entering or leaving training: drop the cached block weight, so one
        # never outlives the phase it was built in.
        self._planar_cache = None
        return super().train(mode)

    # unified single-tensor view (derived; see weight setter note below)
    @property
    def weight(self):
        if self.complex:
            return torch.complex(self.conv_real.weight.data,
                                 self.conv_imag.weight.data)
        return self.conv_real.weight.data

    @weight.setter
    def weight(self, W):
        # `.data.copy_` does not bump the parameters' `_version`, which is what
        # the planar-weight cache keys on -- so invalidate it here by hand.
        self._planar_cache = None
        if self.complex:
            Wc = W if W.is_complex() else torch.complex(W, torch.zeros_like(W))
            self.conv_real.weight.data.copy_(Wc.real)
            self.conv_imag.weight.data.copy_(Wc.imag)
        else:
            self.conv_real.weight.data.copy_(torch.real(W))

    def _op(self, x, weight, bias=None):
        raise NotImplementedError

    # -- planar formulation ---------------------------------------------------
    def _planar_weight(self):
        """The (2M, 2C, P, P) real block weight, cached when that is safe.

        Building it is three `cat`s and a negation: microseconds of GPU work,
        but four kernel LAUNCHES, and a V-cycle net calls 215 different convs per
        forward -- 860 launches, about a tenth of the forward on a node where
        the host is the bottleneck (A100 profile, 2026-10-07).

        UNDER AUTOGRAD IT IS ALWAYS REBUILT: the `cat`s are what carry
        gradients back to conv_real / conv_imag, and every conv is its own
        module, used once per forward, so there is nothing to reuse within a
        training step anyway. The cache only serves no-grad calls (inference,
        validation, timing), where the same weights are read forward after
        forward.

        Staleness. The key is the two parameters' `_version` (bumped by an
        optimizer step and by `load_state_dict`), their storage address and
        their dtype (which change under `.to()` / `.double()`). The one writer
        those cannot see is `conv_real.weight.data.copy_(...)`, and for these
        classes that is the `weight` setter above -- which `set_weight`,
        `project_` and `init_filters` all go through -- so the setter drops the
        cache itself. `train()` / `eval()` drop it too. A raw in-place write to
        `conv_real.weight.data` from anywhere else WOULD be missed: go through
        the setter (tests/test_planar_conv.py pins all of this).
        """
        if torch.is_grad_enabled() or not self.PLANAR_WEIGHT_CACHE:
            return self._build_planar_weight()
        wr, wi = self.conv_real.weight, self.conv_imag.weight
        key = (wr._version, wi._version, wr.data_ptr(), wi.data_ptr(), wr.dtype)
        hit = getattr(self, "_planar_cache", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        W = self._build_planar_weight()
        self._planar_cache = (key, W)
        return W

    def _build_planar_weight(self):
        wr, wi = self.conv_real.weight, self.conv_imag.weight
        if self._PLANAR_TRANSPOSED:
            # weight is (in, out, ...), so the block pattern transposes
            top = torch.cat((wr, wi), dim=1)
            bot = torch.cat((-wi, wr), dim=1)
        else:
            top = torch.cat((wr, -wi), dim=1)
            bot = torch.cat((wi, wr), dim=1)
        return torch.cat((top, bot), dim=0)

    def _planar_bias(self):
        """`cat([br, bi])` -- the bias a complex `br + i bi` actually implies.

        This is NOT what the "gauss" branch computes.  There, with
        `t1 = wr xr + br`, `t2 = wi xi + bi`, `t3 = (wr+wi)(xr+xi) + br + bi`:

            real = t1 - t2         ->  bias  br - bi
            imag = t3 - t1 - t2    ->  bias  0

        so "gauss" offsets the real part by `-bi` and drops the imaginary bias
        entirely.  Every conv in this repo is built `bias=False`, so the two
        have never disagreed; `_planar_ok` nonetheless routes a biased conv to
        "gauss" so that enabling "planar" cannot change an answer.  Kept for
        when that inconsistency is fixed on the gauss side too.
        """
        br, bi = self.conv_real.bias, self.conv_imag.bias
        if br is None and bi is None:
            return None
        n = self.conv_real.weight.shape[1 if self._PLANAR_TRANSPOSED else 0]
        z = torch.zeros(n, device=self.conv_real.weight.device,
                        dtype=self.conv_real.weight.dtype)
        return torch.cat((br if br is not None else z,
                          bi if bi is not None else z))

    def _planar_ok(self):
        """Can this conv take the planar path?  See the class docstring."""
        return (self.COMPLEX_MODE == "planar" and self.groups == 1
                and self.conv_real.bias is None and self.conv_imag.bias is None)

    def _forward_planar(self, x):
        out = self._op(to_planar(x), self._planar_weight(), None)
        return to_complex(out)

    def forward(self, x):
        if not self.complex:
            return self._op(x, self.conv_real.weight, self.conv_real.bias)

        if x.is_complex() and self._planar_ok():
            return self._forward_planar(x)

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
        if torch.is_grad_enabled() or not self.INPLACE_COMBINE:
            return torch.complex(t1 - t2, t3 - t1 - t2)
        # Inference: t1/t2/t3 are fresh conv outputs, unaliased and dead after
        # this, so the two combining subtractions can land in place. At M=169
        # on a 160x160 latent each avoided temporary is ~17 MB of traffic, and
        # an unrolled V-cycle runs this ~100 times. Order matters -- the
        # imaginary part still needs an unmodified t1.
        imag = t3.sub_(t1).sub_(t2)
        real = t1.sub_(t2)
        return torch.complex(real, imag)

class Conv2d(_GaussConvNd):
    def __init__(self, C, M, P, stride=1, bias=False, complex=True, groups=1):
        super().__init__(complex=complex)
        self.padding = (P - 1) // 2
        self.stride = stride
        self.groups = int(groups)
        self.conv_real = nn.Conv2d(C, M, P, stride=stride, padding=self.padding,
                                   bias=bias, groups=self.groups)
        self.conv_imag = nn.Conv2d(C, M, P, stride=stride, padding=self.padding,
                                   bias=bias, groups=self.groups) if complex else None

    def _op(self, x, weight, bias=None):
        return F.conv2d(x, weight, bias=bias, stride=self.stride,
                        padding=self.padding, groups=self.groups)


class ConvTranspose2d(_GaussConvNd):
    _PLANAR_TRANSPOSED = True

    def __init__(self, M, C, P, stride=1, bias=False, complex=True, groups=1):
        super().__init__(complex=complex)
        self.padding = (P - 1) // 2
        # torch requires output_padding < stride; stride - 1 is the value that
        # makes this the exact transpose of Conv2d(C, M, P, stride=stride) for
        # any stride (it was hard-coded to 1, which crashed for stride=1 and is
        # unchanged for the common stride=2 case).
        self.output_padding = max(stride - 1, 0)
        self.stride = stride
        self.groups = int(groups)
        self.conv_real = nn.ConvTranspose2d(M, C, P, stride=stride,
            padding=self.padding, output_padding=self.output_padding, bias=bias,
            groups=self.groups)
        self.conv_imag = nn.ConvTranspose2d(M, C, P, stride=stride,
            padding=self.padding, output_padding=self.output_padding, bias=bias,
            groups=self.groups) if complex else None

    def _op(self, x, weight, bias=None):
        return F.conv_transpose2d(x, weight, bias=bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            groups=self.groups)



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

    # Same unified single-tensor view as _GaussConvNd, so the shared
    # init / projection helpers in models/base.py can write these filters.
    # Without it `self.B[k].weight = W` just parks a plain attribute on the
    # module and the filters are never initialised.
    @property
    def weight(self):
        return torch.complex(self.conv_real.weight.data,
                             self.conv_imag.weight.data)

    @weight.setter
    def weight(self, W):
        Wc = W if W.is_complex() else torch.complex(W, torch.zeros_like(W))
        self.conv_real.weight.data.copy_(Wc.real)
        self.conv_imag.weight.data.copy_(Wc.imag)

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
