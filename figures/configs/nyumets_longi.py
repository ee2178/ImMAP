# NYUMets, CT1 synthesis from T1: the UNet bridge with and without the prior study.
#
#   T1 (input) | Prior CT1 | UNet | UNet + prior w2 | CT1 (truth)
#
#   UNet             I2SB_Unet_NYUMets_CT1_from_all_bs64
#                    the bridge UNet, FLAIR/T1/T2 conditioning, no prior study
#   UNet + prior w2  I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64
#                    the same + the prior study's T1/CT1 at z-2..z+2 as input channels
#
# Every image is on the per-volume scale (each volume over its in-brain median), masked to the
# loader's T1 & CT1 mask, and turned into the radiological view by the dump. "Prior CT1" is the
# prior study's CT1 at the matched slice -- the study the w2 net was shown (attr prior_volume).
# Only multi-study patients are dumped, so both nets are compared on the same population.
#
# Dump on BigPurple (default: 60 val volumes, original slices 80-99; any range works -- the viewer
# scrubs whatever slices a volume's dump holds and keys picks by ORIGINAL slice number):
#   sbatch bigpurple/nyumets_dump.sbatch
#   N_VOLUMES=0 SLICES=70:110 BATCH=20 sbatch --time=08:00:00 bigpurple/nyumets_dump.sbatch
# Bring the dumps down (from the repo root on the cluster, then on the laptop):
#   tar czf nyumets_longi.tgz trained_nets/nyumets/I2SB_Unet_NYUMets_CT1_from_all{,_plus_otherT1CT1_w2}_bs64/eval_dump
#   scp <cluster>:<ImMAP>/nyumets_longi.tgz . && tar xzf nyumets_longi.tgz
# View:
#   python -m figures.webviewer            (and pick nyumets_longi)
#   python figures/viewer.py figures/configs/nyumets_longi.py -n 3
#
# Any other NYUMets run dumped with scripts/dump_eval_nyumets.py shows up in the web viewer's
# model menu (everything under trained_nets/nyumets with an eval_dump/).

import os
import sys

DUMP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "trained_nets", "nyumets")

_CANDIDATES = [
    ("UNet",            "I2SB_Unet_NYUMets_CT1_from_all_bs64"),
    ("UNet + prior w2", "I2SB_Unet_NYUMets_CT1_from_all_plus_otherT1CT1_w2_bs64"),
]


def _dumped(run):
    for d in (os.path.join(DUMP_ROOT, run, "eval_dump"), os.path.join(DUMP_ROOT, run)):
        if os.path.isdir(d) and any(f.endswith(".h5") for f in os.listdir(d)):
            return True
    return False


_have = [(label, run) for label, run in _CANDIDATES if _dumped(run)]
_missing = [run for _, run in _CANDIDATES if not _dumped(run)]
if _missing:
    print(f"[nyumets_longi] no dump for: {' '.join(_missing)} -- column(s) left out",
          file=sys.stderr)

COLUMNS = ([("T1 (input)", "@zero_filled"), ("Prior CT1", "@prior_ct1")] + _have
           + [("CT1 (truth)", None)])

ROWS = []

STATUS_METRIC = "psnr"
FIG_METRICS = ("psnr", "ssim")
FIG_INPUT_METRICS = False       # T1 and prior CT1 are inputs of another contrast: no PSNR on them
FIG_TITLE = r"NYUMets, T1 $\rightarrow$ CT1"
ZOOM_W = ZOOM_H = 40
CROP_PAD = 4
