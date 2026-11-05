"""
sb/ -- Schrodinger-bridge algorithms for ImMAP.

    base          the schedule (a single std_fwd tensor) + shared helpers
                  (forward_sample, predict_x0, reverse_sample, ...)
    i2sb          image-domain I2SB sampling
    i2sb_recon    I2SB for MRI reconstruction: complex bridge from E^H y to the image
    latent_i2sb   latent-domain I2SB sampling (designs A and B) + encode/decode/regress helpers
    immap_sb      ImMAP-SB: I2SB + a learned data-consistency prox solved by CG
    immap_sb_ascent  ImMAP-SB as ImMAP's annealed ascent on the bridge (debiased coordinates)
    unsb          Unpaired Neural SB: the I2SB sampler, trained on-policy with GAN + transport + PatchNCE
"""

from .base import (
    BridgeSchedule, brownian, from_betas, i2sb_betas, build_schedule,
    n_steps, space_indices, forward_std, bridge_coeffs,
    forward_sample, predict_x0, reverse_sample,
)
from .i2sb import i2sb_sample
from .i2sb_recon import bridge_start, predict_x0_recon, i2sb_recon_sample
from .latent_i2sb import (
    encode, decode, latent_regress, latent_i2sb_sample, latent_i2sb_sample_imgdomain,
)
from .immap_sb import ImMAPProx, immap_sb
from .immap_sb_ascent import immap_sb_ascent, bridge_ascent_step
from .unsb import (
    UNSBState, unsb_steps, sample_stage, bridge_step, bridge_step_coeffs, make_g_fn,
    unsb_rollout, unsb_forward, unsb_sample, tau_sb, remaining_fraction, set_requires_grad,
    d_loss, e_loss, g_loss, make_nce_fn,
)
