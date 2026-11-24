# knee R=8, guided reconstruction: the end-to-end nets of guided_knee_R8.py next to I2SB-Recon
# with the small bridge UNet, without and with the guide as extra input channels.
#
# The I2SB columns are the FULL reverse-bridge sampler (evaluation/tasks.py::i2sb_recon) at each
# run's own settings -- 20 NFE, CG data consistency (7 iters, lam 7e-5) on every x0 prediction --
# on the same volumes, slices and measurement as the end-to-end dumps. Their bridge starts at the
# guide (x1_source "guide"); "+ guide" also feeds that guide to the net at every step (cond_x1).
# "<run>:recon_nodc" is the plain sampler (no CG) on the same bridge noise.
#
# Dump the I2SB runs (on bigpurple; the end-to-end ones as for guided_knee_R8.py):
#   ONLY="sbunet_small_R8_g1_tau0.19 sbunet_small_x1_R8_g1_tau0.19" \
#       sbatch bigpurple/guided_recon_dump.sbatch
#   python -m figures.webviewer figures/configs/guided_knee_R8_i2sb.py -n 3

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "guided_recon", "knee", "s4_noise-vlow")

COLUMNS = [
    ("Zero-filled",        "@zero_filled"),
    ("Guide (PD)",         "@guide"),
    ("LPDSNet",            "lpdsnet_R8"),
    ("TGVN",               "tgvn_c6_R8_g1"),
    ("LGGS",               "lggs_R8_g1"),
    ("I2SB",               "sbunet_small_R8_g1_tau0.19"),
    ("I2SB + guide",       "sbunet_small_x1_R8_g1_tau0.19"),
    # ("I2SB, no CG",          "sbunet_small_R8_g1_tau0.19:recon_nodc"),
    # ("I2SB + guide, no CG",  "sbunet_small_x1_R8_g1_tau0.19:recon_nodc"),
    ("Ground truth",       None),
]

# The viewer writes rows/guided_knee_R8_i2sb.json; left empty so the picks start empty.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
BRIGHTNESS = 1.3
# The stored knee slices are upside down (tibia on top); show them the standard way round.
FLIP = "ud"
