# knee R=8: the flat LPDS stack against the multigrid schedule.
#
# Both cells are MGLPDSNet and differ in ONE parameter -- K = 30 versus
# K = [6, [4, 4, 6]] -- so anything visible here is the multigrid schedule and
# nothing else. They are compute-matched, not parameter-matched.
#
# Dump first (on the cluster), then rsync trained_nets/mg_recon/knee/*/eval_dump:
#   python scripts/dump_eval.py --runs trained_nets/mg_recon/knee \
#       --only lpdsnet_R8 mglpds_R8 --n-volumes 4 --slices 12:25
#   python figures/viewer.py figures/configs/mg_knee_R8.py -n 3

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "knee")

COLUMNS = [
    ("Zero-filled",        "@zero_filled"),
    ("LPDS (K=30)",        "lpdsnet_R8"),
    ("MG-LPDS (K=6+444)",  "mglpds_R8"),
    ("Ground truth",       None),
]

# The viewer writes rows/mg_knee_R8.json; seeding ROWS here would leave
# "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
BRIGHTNESS = 1.3
