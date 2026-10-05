# brain R=8: E2E-VarNet, flat LPDS and the multigrid V-cycle, side by side.
#
# All three were trained under the measured protocol (measured k-space + noise,
# Julia mask, operator maps estimated online by ESPIRiT), and the VarNet column
# is `varnetmaps` -- the VarNet cascades on the SAME online maps the unrolled
# nets get -- so the columns differ in the network and nothing else.
#
# The MG-LPDS column is the full-length `mglpds` cell (K=[6,[4,4,6]], M=169,
# Galerkin coarse Gram). To show an architecture-sweep cell instead, point it
# at that run directory, e.g. "mg121v6_R8" (exp9) or "mg169v6_R8_500ep"
# (exp8) -- the name is the directory under trained_nets/mg_recon/brain.
#
# Dump first (on the cluster), then rsync trained_nets/mg_recon/brain/*/eval_dump:
#   python scripts/dump_eval.py --runs trained_nets/mg_recon/brain #       --only varnetmaps_R8 lpdsnet_R8 mglpds_R8 --n-volumes 6 --slices 0:8
#   python figures/viewer.py figures/configs/mg3_brain_R8.py -n 3
#
# `--only` is a substring match, so `lpdsnet_R8` also picks up a tagged probe
# such as lpdsnet_R8_500ep if one exists; that is harmless (it is dumped and
# simply not shown here).

import os

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "brain")

COLUMNS = [
    ("Zero-filled",        "@zero_filled"),
    ("E2E-VarNet",         "varnetmaps_R8"),
    ("LPDS (K=30)",        "lpdsnet_R8"),
    ("MG-LPDS (K=6+446)",  "mglpds_R8"),
    ("Ground truth",       None),
]

# The viewer writes rows/mg3_brain_R8.json; seeding ROWS here would leave
# "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
