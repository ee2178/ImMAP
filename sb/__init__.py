"""
sb/ -- Schrodinger-bridge algorithms for ImMAP.

    base          the schedule (a single std_fwd tensor) + shared helpers
                  (forward_sample, predict_x0, reverse_sample, ...)
    i2sb          image-domain I2SB sampling
    latent_i2sb   latent-domain I2SB sampling (designs A and B) + encode/decode/regress helpers
    immap_sb      ImMAP-SB self-paced annealed ascent (WORK IN PROGRESS)
"""

from .base import (
    BridgeSchedule, brownian, from_betas, i2sb_betas, build_schedule,
    n_steps, space_indices, forward_std, bridge_coeffs,
    forward_sample, predict_x0, reverse_sample,
)
from .i2sb import i2sb_sample
from .latent_i2sb import (
    encode, decode, latent_regress, latent_i2sb_sample, latent_i2sb_sample_imgdomain,
)
from .immap_sb import immap_sb
