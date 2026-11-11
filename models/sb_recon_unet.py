"""
SBReconUnet -- the I2SB paper's ADM UNet as an x0-regressor for MRI RECONSTRUCTION.

The bridge runs in the COMPLEX coil-combined image domain:

    x0 = the fully-sampled SENSE image (the target)       x1 = E^H y, the zero-filled adjoint

and every regressor of training/i2sb_recon.py is called as

    x0_hat, extra = net(x_t, sigma=std_fwd, x1=x1, y=y, E=E, sigma_n=sigma_n)

(the contract sb/i2sb_recon.py::predict_x0_recon builds). This one is the pure-learning baseline:
it never touches the measurement. `y`, `E` and `sigma_n` are accepted and ignored, exactly as
I2SB's own restoration nets see only the degraded image -- data consistency enters only through
x1, which IS a function of y. That is the point of the baseline: the unrolled regressors that do
use (y, E) are measured against it.

COMPLEX AS PLANAR. SBUnet is real-valued, so a complex image enters as two channels [Re, Im] and
the two output channels are recombined into one complex image. The phase is therefore learned,
not discarded: the bridge state mixes x0 and x1 linearly, and a magnitude-only regressor would
drop the phase that the posterior update keeps blending back into x_t.

`cond_x1` (default True) concatenates x1 to the input as well -- I2SB's `cond_x1` option, and the
same thing the synthesis UNets do with their prior contrast. The net then sees C = 4 real channels
(x_t and x1, each planar), otherwise 2. Without it the regressor has to read the zero-filled image
out of x_t, which near t = 0 is almost entirely x0.

`guided` (default False) appends the GUIDE image -- the side scan coil-combined under the
operator's maps, what training/recon.py::_guide hands an image-domain net -- as two more planar
channels: the guided UNet baseline, side information as plain input channels and nothing else.
Unguided, the net declares `guided = False`, so the loops pass no guide even on paired data.

`predict` (default "x0") is what the UNet's output IS. "x0": the complex x0_hat directly. "eps":
the bridge noise, as I2SB-Inversion's (and NVlabs I2SB's) nets predict it -- runner.py
compute_pred_x0: x0_hat = x_t - std_fwd * eps_hat, so the net still RETURNS x0_hat and every
loop and sampler is unchanged; train with i2sb `loss_target="eps"` for their eps-matching loss.

Every other parameter goes to SBUnet unchanged, schedule copy included: kind / tau / n_points /
beta_max MUST match cfg["i2sb"] (assert_schedule_matches is checked at startup).
"""

import torch
import torch.nn as nn

from models.sb_unet import SBUnet


def to_planar(x):
    """(B, 1, H, W) complex -> (B, 2, H, W) real [Re, Im]."""
    return torch.cat([x.real, x.imag], dim=1)


def from_planar(x):
    """(B, 2, H, W) real [Re, Im] -> (B, 1, H, W) complex."""
    return torch.complex(x[:, :1], x[:, 1:2])


class SBReconUnet(nn.Module):
    """ADM UNet on [Re x_t, Im x_t (, Re x1, Im x1)] -> complex x0_hat."""

    # The bridge lives on the measured grid and SBUnet pads internally, so the image-domain
    # embedding (operators/truncate.py) must stay an identity for this net.
    pad_stride = 1

    guide_domain = "image"         # the loops hand a guided net the coil-combined side image

    def __init__(self, cond_x1=True, guided=False, predict="x0", **unet_kws):
        super().__init__()
        self.cond_x1 = bool(cond_x1)
        self.guided = bool(guided)
        if predict not in ("x0", "eps"):
            raise ValueError(f"predict must be 'x0' or 'eps', got {predict!r}")
        self.predict = predict
        if "C" in unet_kws or "out_channels" in unet_kws:
            raise ValueError("SBReconUnet sets the UNet's C / out_channels itself (planar "
                             "complex in and out); set cond_x1 / guided instead")
        C = 2 + 2 * self.cond_x1 + 2 * self.guided
        self.unet = SBUnet(C=C, out_channels=2, **unet_kws)

    def assert_schedule_matches(self, sched, atol=1e-6):
        self.unet.assert_schedule_matches(sched, atol=atol)

    def forward(self, x_t, sigma=None, x1=None, y=None, E=None, sigma_n=None, step=None,
                guide=None):
        if not torch.is_complex(x_t):
            raise TypeError(f"SBReconUnet expects a complex bridge state, got {x_t.dtype}")
        if x_t.shape[1] != 1:
            raise ValueError(f"expected a single-channel image (B, 1, H, W), got {tuple(x_t.shape)}")
        net_in = to_planar(x_t)
        if self.cond_x1:
            if x1 is None:
                raise ValueError("SBReconUnet(cond_x1=True) needs x1, the zero-filled image")
            net_in = torch.cat([net_in, to_planar(x1)], dim=1)
        if self.guided:
            if guide is None:
                raise ValueError("SBReconUnet(guided=True) needs the guide image")
            if guide.dim() == 5:                   # (B, G, 1, H, W): one guide slice only
                if guide.shape[1] != 1:
                    raise ValueError(f"SBReconUnet takes one guide slice, got {guide.shape[1]}")
                guide = guide[:, 0]
            if guide.shape != x_t.shape or not torch.is_complex(guide):
                raise ValueError(f"guide {tuple(guide.shape)} {guide.dtype} must be a complex "
                                 f"image shaped like x_t {tuple(x_t.shape)}")
            net_in = torch.cat([net_in, to_planar(guide)], dim=1)
        elif guide is not None:
            raise ValueError("SBReconUnet(guided=False) is the unguided baseline but was handed "
                             "a guide; set guided=True or drop the guide")
        out, _ = self.unet(net_in, sigma=sigma, step=step)
        out = from_planar(out)
        if self.predict == "eps":
            if sigma is None:
                raise ValueError("SBReconUnet(predict='eps') needs sigma (std_fwd) to turn the "
                                 "predicted noise into x0_hat")
            s = torch.as_tensor(sigma, device=out.device, dtype=out.real.dtype)
            s = s.reshape(s.shape[0], *([1] * (out.dim() - 1))) if s.dim() else s
            out = x_t - s * out                    # I2SB runner.py: compute_pred_x0
        return out, None
