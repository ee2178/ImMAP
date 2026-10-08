# brain R=12: the baselines and every trained multigrid width, side by side.
#
#   Zero-filled | E2E-VarNet | LPDS | MG-LPDS at M = 169, 121, 100, 81 | Ground truth
#
# All trained under the measured protocol (measured k-space + noise, Julia mask,
# operator maps estimated online by ESPIRiT); the VarNet column is `varnetmaps`,
# the VarNet cascades on the SAME online maps the unrolled nets get.
#
# The width cells (mg<M>v6: K=[6,[4,4,6]], rediscretized coarse Gram, planar
# convs) are exp9's full-length runs. `mglpds_R12` is the same M=169 V-cycle
# with the Galerkin coarse Gram -- shown as well, so the figure has an M=169
# column even if `mg169v6_R12` was never trained at full length.
#
# A column whose run has not been dumped is LEFT OUT (and named on stderr)
# rather than stopping the viewer, so this one config works whichever of the
# runs exist.
#
# Dump on the cluster (brain is array task 0), copy the dumps down, view:
#   EXACT=1 ONLY="varnetmaps_R12 lpdsnet_R12 mglpds_R12 mg169v6_R12 mg121v6_R12 mg100v6_R12 mg81v6_R12" \
#       sbatch --array=0 torch/mg_recon_dump.sbatch
#   rsync -avm --include='*/' --include='*_R12/eval_dump/*.h5' --exclude='*' \
#       <cluster>:<ImMAP>/trained_nets/mg_recon/brain/ trained_nets/mg_recon/brain/
#   python figures/viewer.py figures/configs/mgw_brain_R12.py -n 3

import os
import sys

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "brain")

_CANDIDATES = [
    ("E2E-VarNet",              "varnetmaps_R12"),
    ("LPDS (K=30)",             "lpdsnet_R12"),
    ("MG M=169 (Galerkin)",     "mglpds_R12"),
    ("MG M=169",                "mg169v6_R12"),
    ("MG M=121",                "mg121v6_R12"),
    ("MG M=100",                "mg100v6_R12"),
    ("MG M=81",                 "mg81v6_R12"),
]


def _dumped(run):
    for d in (os.path.join(DUMP_ROOT, run, "eval_dump"), os.path.join(DUMP_ROOT, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return True
    return False


_have = [(label, run) for label, run in _CANDIDATES if _dumped(run)]
_missing = [run for _, run in _CANDIDATES if not _dumped(run)]
if _missing:
    print(f"[mgw_brain_R12] no dump for: {' '.join(_missing)} -- column(s) left out",
          file=sys.stderr)

COLUMNS = [("Zero-filled", "@zero_filled")] + _have + [("Ground truth", None)]

# The viewer writes rows/mgw_brain_R12.json; seeding ROWS here would leave
# "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
