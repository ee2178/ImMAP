#!/bin/bash
#
# One-time setup of the paired fastMRI knee data on bigpurple (run on a login node, from the repo
# root /gpfs/scratch/ee2178/ImMAP):
#
#     bash bigpurple/setup_knee_data.sh
#
# 1. Symlinks ../datasets/fastmri_multiknee -> /gpfs/scratch/rl5285/fastmrimutiknee, the paired
#    PD / PDFS copy with train/ val/ test/ subdirectories. scripts/make_guided_recon_configs.py
#    uses ../datasets/fastmri_multiknee/{train,val,test} as its k-space roots when
#    IMMAP_CLUSTER=bigpurple (the bigpurple launchers set it); every pair of the split CSVs is
#    looked up in all three, so it does not matter which subdirectory holds which file.
# 2. Checks that every PD and PDFS file of config/knee/guided/splits/*.csv is there, and that the
#    per-volume scales (config/knee/guided/volume_scales.json, computed on torch and committed --
#    the same raw files, keyed by file name) cover them.
#
# The offline target maps are NOT shared with torch (~1 GB per volume): write them here with
#     sbatch bigpurple/espirit_knee_pairs.sbatch

SRC="${SRC:-/gpfs/scratch/rl5285/fastmrimutiknee}"
LINK="${LINK:-../datasets/fastmri_multiknee}"

[ -f scripts/make_guided_recon_configs.py ] || { echo "run from the ImMAP repo root" >&2; exit 1; }
for s in train val test; do
    [ -d "${SRC}/${s}" ] || { echo "missing ${SRC}/${s}" >&2; exit 1; }
done
mkdir -p "$(dirname "${LINK}")"
if [ -L "${LINK}" ]; then
    echo "exists: ${LINK} -> $(readlink "${LINK}")"
elif [ -e "${LINK}" ]; then
    echo "${LINK} exists and is not a symlink; leaving it alone" >&2
    exit 1
else
    ln -s "${SRC}" "${LINK}"
    echo "linked ${LINK} -> ${SRC}"
fi

python - "${LINK}" <<'PY'
import csv, json, os, sys
root = sys.argv[1]
dirs = [os.path.join(root, s) for s in ("train", "val", "test")]
have = {f: d for d in dirs for f in os.listdir(d) if f.endswith(".h5")}
need, missing = set(), []
for split in ("train", "val", "test"):
    with open(f"config/knee/guided/splits/{split}.csv", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("pd", "pdfs"):
                n = os.path.basename(r[k])
                need.add(n)
                if n not in have:
                    missing.append((split, n))
print(f"{len(have)} h5 files under {root}; the split CSVs name {len(need)}; missing {len(missing)}")
for s, n in missing[:10]:
    print(f"  missing ({s}): {n}")
vs = "config/knee/guided/volume_scales.json"
if os.path.exists(vs):
    vols = json.load(open(vs))["volumes"]
    gap = sorted(need - set(vols))
    print(f"{vs}: covers {len(need) - len(gap)}/{len(need)}" + (f", e.g. missing {gap[:3]}" if gap else ""))
else:
    print(f"{vs} not found: commit it from torch (torch/volume_scales_knee.sbatch writes it)")
sys.exit(1 if missing else 0)
PY
