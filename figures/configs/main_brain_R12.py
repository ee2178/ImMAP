# brain R=12: the main comparison.
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
#   MGLPDS-Big     `mg169v6`     K=[6,[4,4,6]], M=169 (the `mglpds` stopgap
#                                was retired 2026-10-21)
#
# A column whose run has not been dumped is LEFT OUT (and named on stderr)
# rather than stopping the viewer, so this config works while some of the runs
# are still training.
#
# Dump on the cluster (brain is array task 0):
#   EXACT=1 ONLY="varnetmaps_R12 lpdsnet_R12 mg81v6_R12 mg169v6_R12" \
#       sbatch --array=0 torch/mg_recon_dump.sbatch
# A multigrid run LAUNCHED BEFORE 2026-10-07 used the Galerkin coarse operator
# on slices whose width is not a multiple of 8 (brain 768x396); prefix the dump
# with IMMAP_REDISCRETIZE_EMBEDDED=0 to evaluate it with the operator it trained
# with. Leave it off for runs launched after that date.
# Bring the dumps down (from the repo root on the cluster, then on the laptop):
#   tar czf main_brain_R12.tgz trained_nets/mg_recon/brain/{varnetmaps,lpdsnet,mg81v6,mg169v6}_R12/eval_dump
#   scp <cluster>:<ImMAP>/main_brain_R12.tgz . && tar xzf main_brain_R12.tgz
# View:
#   python figures/viewer.py figures/configs/main_brain_R12.py -n 3

import os
import sys

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "mg_recon", "brain")

_CANDIDATES = [
    ("E2E-VarNet",    "varnetmaps_R12"),
    ("LPDSNet",       "lpdsnet_R12"),
    ("MGLPDS-Small",  "mg81v6_R12"),
    ("MGLPDS-Big",    "mg169v6_R12"),
    ("RAM",           "ram_R12"),     # exp13; left out until dumped
]


def _dumped(run):
    for d in (os.path.join(DUMP_ROOT, run, "eval_dump"), os.path.join(DUMP_ROOT, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return True
    return False


def _pick(runs):
    """The first run of `runs` (one name or a preference list) with a dump."""
    for run in ((runs,) if isinstance(runs, str) else runs):
        if _dumped(run):
            return run
    return None


_have = [(label, _pick(runs)) for label, runs in _CANDIDATES if _pick(runs)]
_missing = [runs if isinstance(runs, str) else "/".join(runs)
            for _, runs in _CANDIDATES if not _pick(runs)]
if _missing:
    print(f"[main_brain_R12] no dump for: {' '.join(_missing)} -- column(s) left out",
          file=sys.stderr)

COLUMNS = [("Zero-filled", "@zero_filled")] + _have + [("Ground truth", None)]

# The viewer writes rows/main_brain_R12.json; seeding ROWS here would leave
# "pick 3" already full before you had looked at anything.
ROWS = []

STATUS_METRIC = "psnr"
ZOOM_W = ZOOM_H = 48
