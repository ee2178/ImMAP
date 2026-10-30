# knee R=8: the main comparison.
#
#   Zero-filled | E2E-VarNet | LPDSNet | MGLPDS-Small | MGLPDS-Big | Ground truth
#
# The four nets of the paper's table, all trained under the measured protocol
# (measured k-space + noise, Julia mask, operator maps estimated online by
# ESPIRiT, organ mask on):
#
#   E2E-VarNet     `varnetmaps`  the VarNet cascades on the SAME online maps
#                                the unrolled nets get
#   LPDSNet        `lpdsnet`     K=30, M=169
#   MGLPDS-Small   `mg81v6`      K=[6,[4,4,6]], M=81
#   MGLPDS-Big     `mg169v6`     K=[6,[4,4,6]], M=169
#
# A column whose run has not been dumped is LEFT OUT (and named on stderr)
# rather than stopping the viewer, so this config works while some of the runs
# are still training.
#
# Dump on the cluster (knee is array task 1):
#   EXACT=1 ONLY="varnetmaps_R8 lpdsnet_R8 mg81v6_R8 mg169v6_R8" \
#       sbatch --array=1 torch/mg_recon_dump.sbatch
# Bring the dumps down (from the repo root on the cluster, then on the laptop):
#   tar czf main_knee_R8.tgz trained_nets/mg_recon/knee/{varnetmaps,lpdsnet,mg81v6,mg169v6}_R8/eval_dump
#   scp <cluster>:<ImMAP>/main_knee_R8.tgz . && tar xzf main_knee_R8.tgz
# View:
#   python figures/viewer.py figures/configs/main_knee_R8.py -n 3

import os
import sys

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "knee")

_CANDIDATES = [
    ("E2E-VarNet",    "varnetmaps_R8"),
    ("LPDSNet",       "lpdsnet_R8"),
    ("MGLPDS-Small",  "mg81v6_R8"),
    ("MGLPDS-Big",    "mg169v6_R8"),
    ("RAM",           "ram_R8"),     # exp13; left out until dumped
]


def _dumped(run):
    for d in (os.path.join(DUMP_ROOT, run, "eval_dump"), os.path.join(DUMP_ROOT, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return True
    return False


_have = [(label, run) for label, run in _CANDIDATES if _dumped(run)]
_missing = [run for _, run in _CANDIDATES if not _dumped(run)]
if _missing:
    print(f"[main_knee_R8] no dump for: {' '.join(_missing)} -- column(s) left out",
          file=sys.stderr)

COLUMNS = [("Zero-filled", "@zero_filled")] + _have + [("Ground truth", None)]

# The viewer writes rows/main_knee_R8.json; seeding ROWS here would leave
# "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
BRIGHTNESS = 1.3
# The stored knee slices are upside down (tibia on top); show them the standard
# way round. Display only -- dumps and metrics are untouched.
FLIP = "ud"
