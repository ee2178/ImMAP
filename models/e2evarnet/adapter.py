"""
E2E-VarNet as an ImMAP model.

The network itself is `_fastmri/varnet.py`, vendored unmodified apart from its
import block (see that file's header). Nothing here changes the algorithm; this
is purely the calling-convention adapter, and every difference between the two
codebases is handled in one place so the baseline stays honest.

WHAT HAS TO BE TRANSLATED
-------------------------
                    ImMAP                        fastMRI VarNet
  measurement       y (B, C, H, W) complex       (B, C, H, W, 2) real-view
  operator          E = Mask @ FFT2D @ Sense     mask alone, (B, 1, 1, W, 1) bool
  coil maps         supplied by the dataset      ESTIMATED by its own SensitivityModel
  noise level       sigma, per-sample            not an input
  output            complex image (B, 1, H, W)   RSS MAGNITUDE (B, H, W), real

The FFT conventions already agree -- both are
`fftshift(fftn(ifftshift(x)), norm="ortho")` -- so k-space passes across with a
dtype change and no re-centring. That is worth stating because it is the one
assumption that would silently corrupt every number if it were wrong.

THREE THINGS THIS BASELINE DOES DIFFERENTLY, ON PURPOSE
-------------------------------------------------------
1. It ESTIMATES the coil maps. `Sense(smaps)` in `E` is ignored: E2E-VarNet's
   defining feature is that the sensitivity model is trained jointly with the
   cascades. Feeding it the dataset's maps would be a different, easier method
   and not the published baseline. The unrolled nets get the maps; VarNet earns
   them. That asymmetry favours the unrolled nets and should be said out loud
   when the numbers are reported.

2. It returns MAGNITUDE. `fastmri.rss(complex_abs(...))` is real and
   non-negative, so there is no phase to compare. The grid's
   `magnitude-nl1-nl2` loss and PSNR/SSIM/NRMSE all take `.abs()` first, so the
   comparison is valid -- but anything phase-sensitive downstream is not, and
   `returns_magnitude = True` is set so a caller can check rather than assume.

3. It ignores `sigma`. There is nowhere to put it: the architecture has no
   noise-adaptive parameter. On a sigma-varying grid that is a real handicap
   and part of what the comparison measures.

GRID SIZE
---------
`pad_stride = 1`: VarNet works at the measured size and its `NormUnet` pads
internally to a multiple of 16, so the image-domain embedding
(`operators/truncate.py`) is an identity here and `training/recon.py::_embed`
leaves it alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.e2evarnet._fastmri import _shim as fastmri
from models.e2evarnet._fastmri.varnet import VarNet
from operators.accessors import get_mask, get_sensitivity_map


class E2EVarNet(nn.Module):
    """fastMRI's `VarNet` behind ImMAP's `(y, E, sigma) -> (x_hat, extra)`."""

    # Read by `training/recon.py::_embed`; 1 makes the embedding an identity.
    pad_stride = 1
    # Read by anything that would otherwise assume a complex reconstruction.
    returns_magnitude = True

    def __init__(self, num_cascades=12, sens_chans=8, sens_pools=4,
                 chans=18, pools=4, mask_center=True, acs_lines=None,
                 use_smaps=False, output="rss"):
        super().__init__()
        self.net = VarNet(num_cascades=num_cascades, sens_chans=sens_chans,
                          sens_pools=sens_pools, chans=chans, pools=pools,
                          mask_center=mask_center)

        # -- KNOWN-MAPS MODE ------------------------------------------------
        # `use_smaps=True` takes the coil maps from `E` instead of estimating
        # them, which is NOT the published baseline (point 1 of the header) but
        # IS the comparison that removes the map asymmetry: every arm then gets
        # the same ESPIRiT maps and only the prior differs. Sljiva carries the
        # same switch (`use_smaps` in `src/networks/e2e_varnet.jl`), so this is
        # a supported variant of the method rather than an invention here.
        #
        # The sensitivity model is DELETED, not just skipped: left in place its
        # parameters would sit in the optimizer receiving no gradient, which
        # DDP reports as unused parameters and which quietly inflates the
        # parameter count this baseline is judged on.
        if output not in ("rss", "sense"):
            raise ValueError(f"output must be 'rss' or 'sense', got {output!r}")
        if output == "sense" and not use_smaps:
            raise ValueError(
                "output='sense' needs use_smaps=True: the coil combination is "
                "only defined against maps this model was given, and the "
                "estimated ones are not exported.")
        self.use_smaps = bool(use_smaps)
        self.output = output
        if self.use_smaps:
            self.net.sens_net = None
        # `sense` returns a COMPLEX image, the same quantity the unrolled nets
        # produce and the same one `image` holds, so nothing downstream has to
        # treat this baseline specially. Instance attribute, shadowing the
        # class default.
        self.returns_magnitude = (output == "rss")
        # Handed to SensitivityModel as `num_low_frequencies`: the band it
        # calibrates from. Three settings:
        #   int     a fixed count. Right only if it IS the mask's ACS -- a
        #           pinned 20 against Sljiva's 13-line centre_frac mask put 7
        #           unsampled columns in the band, and at R=4 one accelerated
        #           line, whose isolated sample aliases the calibration image.
        #   "mask"  count the contiguous centre run of THIS mask, per forward.
        #           The same lines the unrolled nets' online ESPIRiT calibrates
        #           from, so both sides estimate maps from identical data.
        #   None    fastMRI's own inference: 2 * min(left, right), i.e. forced
        #           symmetric -- 12 of a 13-line odd ACS.
        if acs_lines in (None, "mask"):
            self.acs_lines = acs_lines
        else:
            self.acs_lines = int(acs_lines)

    # -- conversions --------------------------------------------------------
    @staticmethod
    def _mask_to_varnet(mask, batch, device):
        """ImMAP mask -> the `(B, 1, 1, W, 1)` bool VarNet indexes.

        `SensitivityModel.get_pad_and_num_low_freqs` reads `mask[:, 0, 0, :, 0]`
        literally, so the phase-encode axis has to sit at index 3 and the tensor
        has to be 5-D. ImMAP masks are vertical lines -- constant down H -- so
        one row carries the whole pattern; that is CHECKED rather than assumed,
        because silently taking row 0 of a 2-D mask would undersample wrongly
        and still train.
        """
        m = mask
        while m.dim() < 4:
            m = m.unsqueeze(0)
        m = m[..., :, :]                                   # (B?, 1, H, W)
        col = m[..., :1, :]                                # first row
        if not torch.equal(m.to(torch.bool), col.to(torch.bool).expand_as(m)):
            raise ValueError(
                "E2EVarNet expects a mask that is constant along the readout "
                "axis (vertical phase-encode lines), which is what "
                "physics/mask.py::make_acc_mask produces. This one varies down "
                "H, so a 1-D VarNet mask cannot represent it.")
        return col.reshape(-1, 1, 1, col.shape[-1], 1)[:1].expand(
            batch, 1, 1, col.shape[-1], 1).to(device=device, dtype=torch.bool)

    # -- forward ------------------------------------------------------------
    def forward(self, y, E=None, sigma=None, **_):
        if E is None:
            raise ValueError(
                "E2EVarNet needs the encoding operator to recover the sampling "
                "mask; it reconstructs from k-space, not from an adjoint.")
        if not torch.is_complex(y):
            raise ValueError(f"expected complex k-space, got {y.dtype}.")

        mask = self._mask_to_varnet(get_mask(E), y.shape[0], y.device)
        ks = torch.view_as_real(y.contiguous())            # (B, C, H, W, 2)

        if not self.use_smaps:
            nlf = self.acs_lines
            if nlf == "mask":
                from physics.online_smaps import acs_line_count
                nlf = acs_line_count(get_mask(E))
            out = self.net(ks, mask, num_low_frequencies=nlf)
            # (B, H, W) real RSS -> (B, 1, H, W), the shape the training loop
            # and the metrics expect. Complex is NOT reconstructed; see header.
            return out.unsqueeze(1), {}

        # Known maps: `VarNet.forward` is reproduced here MINUS its first line
        # (`sens_maps = self.sens_net(...)`). The cascades already take the
        # maps as an argument, so nothing inside the method is re-implemented.
        sm = get_sensitivity_map(E)
        if sm.shape[:2] != y.shape[:2] or sm.shape[-2:] != y.shape[-2:]:
            raise ValueError(
                f"coil maps {tuple(sm.shape)} do not match the k-space "
                f"{tuple(y.shape)}; E must carry the maps for THIS batch.")
        sens = torch.view_as_real(sm.contiguous())         # (B, C, H, W, 2)

        kspace_pred = ks.clone()
        for cascade in self.net.cascades:
            kspace_pred = cascade(kspace_pred, ks, mask, sens)

        if self.output == "rss":
            out = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)),
                              dim=1)
            return out.unsqueeze(1), {}

        # SENSE combination with the given maps: sum_c conj(s_c) x_c -- exactly
        # how `image` was built, so this output is comparable to the ground
        # truth in phase as well as magnitude.
        img = fastmri.complex_mul(fastmri.ifft2c(kspace_pred),
                                  fastmri.complex_conj(sens)).sum(dim=1)
        return torch.view_as_complex(img.contiguous()).unsqueeze(1), {}

    # -- ImMAP hooks --------------------------------------------------------
    def project(self):
        """No constraint set. Present so the training loop's hasattr holds."""
        return self

    def extra_repr(self):
        return (f"cascades={len(self.net.cascades)}, "
                f"acs_lines={self.acs_lines}, "
                f"estimates_own_smaps={not self.use_smaps}, "
                f"output={self.output}")
