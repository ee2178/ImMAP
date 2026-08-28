#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --job-name=resume
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#
# Resume ONE run. Task-agnostic: it does not care whether the run is recon,
# denoising, i2sb or synthesis -- it hands the config back to train.py, which
# already knows how to continue from `paths.ckpt`.
#
#     sbatch slurm/resume.sbatch trained_nets/mg_recon/knee/mglpds_R8
#     sbatch slurm/resume.sbatch trained_nets/mg_recon/knee/mglpds_R8/config.json
#     sbatch slurm/resume.sbatch 3xk91abz            # <- a wandb run id
#     RUN=trained_nets/task/brain/mglpds_R8_warm sbatch slurm/resume.sbatch
#
# A directory is resolved to `<dir>/config.json`. Anything that is not an
# existing path is taken as a WANDB RUN ID and looked up by scanning
# `trained_nets/` for the config that recorded it -- `train.py` writes the
# assigned id into `config.json` after `wandb.init`, so the mapping is already
# on disk. No network and no wandb credentials are involved, which matters on a
# compute node. Override the search root with RUNS_ROOT=...
#
# Look before you leap:
#
#     bash slurm/resume.sbatch --check trained_nets/mg_recon/knee/mglpds_R8
#     bash slurm/resume.sbatch --check 3xk91abz
#
# which prints the run's progress and exits without submitting anything.
#
# ONE RUN PER SUBMIT, DELIBERATELY
# --------------------------------
# The previous resume script enumerated every unfinished run and indexed an
# array into that list. The list SHRINKS as runs finish, so a task that started
# after another had completed saw a shorter list and its index pointed at a
# different run -- silently resuming some twice and skipping others, and only
# when the cluster was busy enough for tasks to queue behind each other.
#
# There is no index here to shift. For several runs, submit several jobs:
#
#     for d in trained_nets/mg_recon/knee/*/; do
#         bash slurm/resume.sbatch --check "$d" && sbatch slurm/resume.sbatch "$d"
#     done
#
# `--check` exits non-zero on a finished or unresumable run, so that loop skips
# them without submitting.
#
# WHICH CONFIG, AND WHY IT MATTERS
# --------------------------------
# A run directory ends up with two, and they are not interchangeable:
#
#   config.gen.json  written by the launcher from the generator.
#                    `paths.ckpt: null`, `wandb.id: null` -- a FRESH run.
#   config.json      written by train.py after wandb.init. Carries the assigned
#                    `wandb.id` and `paths.ckpt = <save_dir>/net.ckpt`, and has
#                    `model.params.init` forced false so the spectral init does
#                    not overwrite the loaded weights.
#
# Only `config.json` resumes. This script never regenerates anything:
# re-running a launcher with REGENERATE=1 would rewrite config.gen.json and
# start over under a new wandb id, which is the failure this exists to avoid.
#
# It also means a resumed run keeps the schedule it was LAUNCHED with. Picking
# up a newer schedule is a different operation -- delete config.json and use the
# original launcher -- and it discards the existing progress.
#
# SUBMIT FROM THE REPO ROOT. `#SBATCH --output=logs/...` resolves against the
# SUBMIT directory, not this file's location and not the `cd` below. From
# slurm/ the job dies before the script runs, so there is no log to say why.
# `#SBATCH` directives cannot expand variables, so this cannot be fixed here.

set -o pipefail

# --- arguments -------------------------------------------------------------
CHECK_ONLY=0
TARGET="${RUN:-}"
for _arg in "$@"; do
    case "${_arg}" in
        --check) CHECK_ONLY=1 ;;
        -*)      echo "[resume] unknown flag '${_arg}'" >&2; exit 2 ;;
        *)       TARGET="${_arg}" ;;
    esac
done

if [ -z "${TARGET}" ]; then
    echo "[resume] give a run directory, a config path, or a wandb run id:" >&2
    echo "         sbatch $0 trained_nets/mg_recon/knee/mglpds_R8" >&2
    echo "         sbatch $0 3xk91abz" >&2
    echo "         bash   $0 --check trained_nets/mg_recon/knee/mglpds_R8" >&2
    exit 2
fi
RUNS_ROOT="${RUNS_ROOT:-trained_nets}"

# --- locate the repo -------------------------------------------------------
# sbatch runs a COPY of this file from the node's spool dir, so $BASH_SOURCE
# points there rather than at the checkout.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[resume] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, \$PWD, ~/scratch/ee2178/ImMAP)." >&2
    echo "[resume] run from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0 ..." >&2
    exit 1
fi

if [ "${CHECK_ONLY}" = "0" ]; then
    source ~/.bashrc
    conda activate gcdl
fi
cd "${IMMAP_ROOT}"
mkdir -p logs

# --- resolve + report ------------------------------------------------------
# Reading the checkpoint needs torch, so --check wants the env active too; it
# degrades to a path-only report rather than lying about progress.
CONFIG="$(python - "${TARGET}" "${RUNS_ROOT}" <<'PY'
import glob, json, os, sys

target = sys.argv[1].rstrip("/\\")
runs_root = sys.argv[2] if len(sys.argv) > 2 else "trained_nets"


def by_wandb_id(wid, root):
    """Every run whose config.json recorded this wandb id.

    Scanned off disk rather than queried from the wandb API: `train.py` writes
    the assigned id into config.json right after `wandb.init`, so the mapping
    is already local -- no network, no credentials, and it works on a compute
    node with no outbound access.
    """
    hits, scanned = [], 0
    for path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        scanned += 1
        try:
            with open(path) as f:
                c = json.load(f)
        except Exception:                                         # noqa: BLE001
            continue
        if str((c.get("wandb") or {}).get("id")) == wid:
            hits.append((path, c))
    return hits, scanned


if not os.path.exists(target):
    # not a path -> treat it as a wandb run id
    hits, scanned = by_wandb_id(target, runs_root)
    if not hits:
        known = []
        for path in sorted(glob.glob(os.path.join(runs_root, "**", "config.json"),
                                     recursive=True)):
            try:
                with open(path) as f:
                    c = json.load(f)
            except Exception:                                     # noqa: BLE001
                continue
            wid = (c.get("wandb") or {}).get("id")
            if wid:
                known.append(f"    {wid}  {os.path.dirname(path)}")
        msg = [f"[resume] {target!r} is not a path, and no config under "
               f"{runs_root}/ records it as a wandb id ({scanned} scanned)."]
        if known:
            msg.append(f"[resume] ids that ARE on disk ({len(known)}):")
            msg.extend(known[:20])
            if len(known) > 20:
                msg.append(f"    ... and {len(known) - 20} more")
        else:
            msg.append(f"[resume] no config under {runs_root}/ has a wandb id "
                       f"at all -- has anything finished wandb.init yet?")
        sys.exit("\n".join(msg))
    if len(hits) > 1:
        # A copied run directory is the usual cause. Refuse rather than pick.
        lines = [f"[resume] wandb id {target!r} matches {len(hits)} runs:"]
        lines += [f"    {os.path.dirname(p)}" for p, _ in hits]
        lines.append("[resume] pass the directory you mean instead.")
        sys.exit("\n".join(lines))
    cfg_path = hits[0][0]
    print(f"[resume] wandb id {target} -> {os.path.dirname(cfg_path)}",
          file=sys.stderr)

elif os.path.isdir(target):
    cfg_path = os.path.join(target, "config.json")
    gen_path = os.path.join(target, "config.gen.json")
    if not os.path.exists(cfg_path):
        if os.path.exists(gen_path):
            sys.exit(
                f"[resume] {target} has no config.json, only config.gen.json.\n"
                f"         That run never reached train.py's save_config, so "
                f"there is nothing to resume FROM -- it would start over.\n"
                f"         If that is what you want, submit it through its "
                f"launcher instead.")
        sys.exit(f"[resume] {target} holds neither config.json nor config.gen.json.")
else:
    cfg_path = target
    if not os.path.exists(cfg_path):
        sys.exit(f"[resume] {cfg_path} does not exist.")

with open(cfg_path) as f:
    cfg = json.load(f)

name = cfg.get("experiment", {}).get("name", "?")
save_dir = cfg.get("paths", {}).get("save_dir", "?")
ckpt = cfg.get("paths", {}).get("ckpt")
wid = cfg.get("wandb", {}).get("id")
t = cfg.get("training", {})
spe, nep = t.get("steps_per_epoch"), t.get("num_epochs")

print(f"[resume] config   : {cfg_path}", file=sys.stderr)
print(f"[resume] run      : {name}", file=sys.stderr)
print(f"[resume] save_dir : {save_dir}", file=sys.stderr)
print(f"[resume] wandb.id : {wid}", file=sys.stderr)

if not ckpt:
    sys.exit("[resume] paths.ckpt is null -- this config starts a FRESH run, it "
             "does not resume. Use the launcher that produced it.")
if not os.path.exists(ckpt):
    sys.exit(f"[resume] paths.ckpt points at {ckpt}, which does not exist.")

# The checkpoint is written only on an IMPROVING epoch, so `step` can lag the
# true progress. That biases toward "not done", i.e. toward resuming -- the
# safe direction, since a resumed complete run exits immediately.
try:
    import torch
except ImportError:
    print("[resume] torch not importable; cannot read progress. Activate the "
          "env (conda activate gcdl) for a progress report.", file=sys.stderr)
    print(cfg_path)
    raise SystemExit(0)

try:
    # `weights_only=False` is required from torch 2.6 (the default flipped) and
    # does not exist before 2.0. Match training/common.py::load_ckpt, then fall
    # back so this check runs under any torch.
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt, map_location="cpu")
    step = blob.get("step") or 0
except Exception as e:                                            # noqa: BLE001
    sys.exit(f"[resume] cannot read {ckpt}: {type(e).__name__}: {e}")

ep = int(step) // int(spe) if spe else 0
print(f"[resume] progress : epoch {ep} / {nep}   (step {step})", file=sys.stderr)
if nep is not None and ep >= nep:
    sys.exit(f"[resume] already complete ({ep} >= {nep} epochs). Nothing to do.")

print(cfg_path)
PY
)"
STATUS=$?

if [ "${STATUS}" -ne 0 ] || [ -z "${CONFIG}" ]; then
    exit "${STATUS}"
fi

if [ "${CHECK_ONLY}" = "1" ]; then
    echo "[resume] resumable. Submit with:  sbatch $0 ${TARGET}"
    exit 0
fi

echo "[resume] resuming ${CONFIG}"
python train.py "${CONFIG}"
