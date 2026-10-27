# brain R=8: the slices where MGLPDS beats BOTH other nets by the most.
#
#   Zero-filled | E2E-VarNet | LPDSNet | MGLPDS | Ground truth
#
# Rows are written by scripts/pick_mg_wins.py (rows/mgwins_brain_R8.json): the top
# slices of scripts/rank_slices.py --target mg81v6 --mode wins over the DUMPS, one
# slice per volume. Kept apart from main_brain_R8 so its hand-picked rows are untouched.
# Figure: python figures/make_figure.py figures/configs/mgwins_brain_R8.py

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "brain")

COLUMNS = [("Zero-filled", "@zero_filled"),
           ("E2E-VarNet", "varnetmaps_R8"),
           ("LPDSNet", "lpdsnet_R8"),
           ("MGLPDS", "mg81v6_R8"),
           ("Ground truth", None)]

ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48

