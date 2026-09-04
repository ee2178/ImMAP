# Multigrid recon, fastMRI brain (T2), R=8: the three nets the paper compares,
# with the zero-filled adjoint and the ground truth as bookends.
#
# Any name in figures.common.CONFIGURABLE may be set here; anything left out
# keeps the default. An unrecognised name is an error, not a silent no-op.
#
# Dump these first:
#   python scripts/dump_eval.py --runs trained_nets/mg_recon/brain \
#       --only lpdsnet_R8 mglpds_R8 mggrouplpds_R8 --n-volumes 8 --slices 0:16
# Then:
#   python figures/viewer.py figures/configs/mg_brain_R8.py -n 3

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "brain")

# (label, dump spec). `None` is the ground truth and `@<dataset>` a stack every
# dump carries; both are read from the first named column, since neither is any
# one method's output. Column order here is the figure's column order.
COLUMNS = [
    ("Zero-filled",   "@zero_filled"),
    ("LPDSNet",       "lpdsnet_R8"),
    ("MG-LPDS",       "mglpds_R8"),
    ("MG-GroupLPDS",  "mggrouplpds_R8"),
    ("Ground truth",  None),
]

# Left empty on purpose: the viewer writes rows/mg_brain_R8.json, and seeding
# ROWS here would leave "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
RESID_GAIN = 5.0
