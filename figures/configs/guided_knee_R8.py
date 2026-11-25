# knee R=8, guided reconstruction (PDFS target, PD guide fully sampled): the end-to-end nets
# trained on torch, ported to bigpurple (scripts/port_runs.py), against each other.
#
# LPDSNet takes no guide (the unguided baseline on the same paired data); TGVN (6 cascades) and
# the three LGGS variants take the PD scan. The "Guide" column is the PD scan itself, fully
# sampled and coil-combined with the operator's maps -- the same image for every net.
#
# Dump first (on bigpurple), then copy trained_nets/guided_recon/knee/s4_noise-vlow/*/eval_dump:
#   sbatch bigpurple/guided_recon_dump.sbatch
#   python figures/viewer.py figures/configs/guided_knee_R8.py -n 3

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "guided_recon", "knee", "s4_noise-vlow")

COLUMNS = [
    ("Zero-filled",      "@zero_filled"),
    ("Guide (PD)",       "@guide"),
    # ("E2E-VarNet",       "e2evarnet_R8"),   # unguided, own maps: uncomment once dumped
    ("LPDSNet",          "lpdsnet_R8"),
    ("TGVN",             "tgvn_c6_R8_g1"),
    ("LGGS",             "lggs_R8_g1"),
    ("LGGS guide-W",     "lggs_guide_w_R8_g1"),
    # ("LGGS guide-conv",  "lggs_guide_conv_R8_g1"),   # not copied to bigpurple: no dump
    ("Ground truth",     None),
]

# The viewer writes rows/guided_knee_R8.json; left empty so the picks start empty.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
BRIGHTNESS = 1.3
# The stored knee slices are upside down (tibia on top); show them the standard way round.
FLIP = "ud"
