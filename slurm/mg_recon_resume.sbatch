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
#SBATCH --job-name=MGRecon-resume
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-13          # MUST equal the resumable-run count; checked at runtime
#
# Resume interrupted mg_recon runs.
#
# ANATOMY picks which tree to walk -- knee, brain, or all. It defaults to knee
# because that is what has been run so far. Set it below, or per-invocation:
#
#     ANATOMY=brain sbatch --array=0-N slurm/mg_recon_resume.sbatch
#     bash slurm/mg_recon_resume.sbatch --list brain
#
# `--list` always reports runs excluded BY THE ANATOMY FILTER, so switching to
# the other anatomy later cannot silently leave work behind.
#
# SUBMIT FROM THE REPO ROOT, or make sure `logs/` exists relative to wherever
# you do run sbatch. `#SBATCH --output=logs/...` is resolved against the SUBMIT
# directory -- not this file's location, and not the `cd "${IMMAP_ROOT}"` below.
# Submitting from slurm/ makes slurmd try to open slurm/logs/..., fail, and kill
# the job BEFORE the script runs, so there is no log to explain what happened.
# `#SBATCH` directives cannot expand variables, so this cannot be fixed here.
#
# LISTING IS MANDATORY, NOT ADVISORY. `--list` writes a MANIFEST, and the array
# reads that instead of re-enumerating:
#
#     conda activate gcdl
#     bash slurm/mg_recon_resume.sbatch --list
#     sbatch --array=0-<N-1> slurm/mg_recon_resume.sbatch
#
# The manifest exists because the run list SHRINKS as runs finish. When each
# array task enumerated for itself, a task starting after another had completed
# a run saw a shorter list, and its index pointed at a different run than
# `--list` had shown -- silently resuming some runs twice and skipping others.
# That is only visible when tasks queue behind each other, i.e. exactly when the
# cluster is busy. Now the index is frozen at list time.
#
# A manifest going stale is safe in the one direction that matters: a run that
# finished after listing is still in the file, gets resumed, and train.py exits
# immediately because start_epoch >= num_epochs.
#
# knee and brain are separate arrays, as with the two original launchers -- an
# index means something different in each, so do not mix them in one submission.
# They also get separate manifests (logs/resume_<anatomy>.manifest).
#
# WHICH CONFIG THIS USES, AND WHY IT MATTERS
# ------------------------------------------
# Each run directory ends up with TWO configs, and they are not interchangeable:
#
#   config.gen.json  written by slurm/_mg_recon_body.sh from the generator.
#                    `paths.ckpt: null`, `wandb.id: null` -- a FRESH run.
#   config.json      written by train.py after wandb.init. Carries the assigned
#                    `wandb.id` and `paths.ckpt = <save_dir>/net.ckpt`, and has
#                    `model.params.init` forced false so the spectral init does
#                    not overwrite the loaded weights.
#
# Only `config.json` resumes. Re-running the original launcher would regenerate
# `config.gen.json` (REGENERATE=1) and start over under a new wandb run id --
# which is exactly the failure this script exists to avoid. Nothing here
# regenerates anything.
#
# It also means a resumed run keeps the schedule it was LAUNCHED with. The
# recent generator change (6000 x 50 steps, val@100, T_max 300000) applies to
# runs started after it, not to these. Restarting a run to pick up the new
# schedule is a different operation -- delete its config.json and use the
# original launcher -- and it throws away the existing progress.
#
# OVERRIDES (env vars)
#   ANATOMY=knee|brain|all            which tree to walk (default: knee)
#   RESUME_ONLY="mglpds mgaltsplit"   only these run-dir names (substring match)
#   RESUME_SKIP="lpdsnet"             exclude these (substring match)
#   FORCE_ALL=1                       ignore the done check, resume everything
#   INCLUDE_UNSTARTED=1               also (re)start runs that never wrote a
#                                     checkpoint, from their config.gen.json
#   IMMAP_ROOT=/path/to/ImMAP         override the repo location

# --- locate the repo -------------------------------------------------------
# sbatch runs a COPY of this file from the node's spool dir, so $BASH_SOURCE
# points there rather than at the checkout.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/train.py" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/train.py" ]; then
    echo "[resume] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[resume] run from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

# --- which anatomy, and list-vs-run ----------------------------------------
# EDIT THIS, or pass it as an env var / an argument.
ANATOMY="${ANATOMY:-knee}"          # knee | brain | all

LIST_ONLY=0
for _arg in "$@"; do
    case "${_arg}" in
        --list)            LIST_ONLY=1 ;;
        knee|brain|all)    ANATOMY="${_arg}" ;;
        *) echo "[resume] unknown argument '${_arg}'; expected --list and/or" \
                "knee|brain|all" >&2
           exit 1 ;;
    esac
done
case "${ANATOMY}" in
    knee|brain|all) ;;
    *) echo "[resume] ANATOMY must be knee, brain or all; got '${ANATOMY}'" >&2
       exit 1 ;;
esac
export ANATOMY

if [ "${LIST_ONLY}" = "0" ]; then
    source ~/.bashrc
    conda activate gcdl
fi
cd "${IMMAP_ROOT}"
mkdir -p logs

MANIFEST="logs/resume_${ANATOMY}.manifest"

# --- enumerate the runs that need resuming ---------------------------------
# ONLY --list enumerates. It writes MANIFEST; the array path below reads that
# file and never re-derives the list, so an index cannot shift under a task.
if [ "${LIST_ONLY}" = "1" ]; then
    # Drop any previous manifest FIRST. If this enumeration dies (no conda env,
    # unreadable checkpoint, ...) the array must find nothing rather than an
    # older, longer list whose indices no longer mean what --list just printed.
    rm -f "${MANIFEST}"
    # The heredoc body has to start at column 0; the `then`/`fi` around it do not.
    if ! python - "${MANIFEST}" <<'PY'
import datetime, glob, json, os, sys

manifest = sys.argv[1]
anatomy = os.environ.get("ANATOMY", "knee")
only = [t for t in os.environ.get("RESUME_ONLY", "").split() if t]
skip = [t for t in os.environ.get("RESUME_SKIP", "").split() if t]
force = os.environ.get("FORCE_ALL") == "1"
include_unstarted = os.environ.get("INCLUDE_UNSTARTED") == "1"

# Reading a checkpoint needs torch. Without it every run would come back at
# "epoch 0" and the listing would look plausible while being uniformly wrong, so
# fail loudly instead -- `--list` deliberately does not activate the env.
try:
    import torch
except ImportError:
    sys.exit("[resume] torch is not importable -- `conda activate gcdl` first. "
             "Without it every run would be reported at epoch 0.")


def progress(run_dir):
    """(done, start_epoch, num_epochs) for a run, from its ckpt + config.

    The checkpoint is written only on an IMPROVING epoch, so `step` can lag the
    true progress. That biases the answer toward "not done", i.e. toward
    resuming -- the safe direction; a resumed complete run exits immediately.
    """
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    t = cfg["training"]
    spe, nep = t["steps_per_epoch"], t["num_epochs"]
    ckpt = os.path.join(run_dir, "net.ckpt")
    if not os.path.exists(ckpt):
        return None, 0, nep
    try:
        step = torch.load(ckpt, map_location="cpu", weights_only=False).get("step") or 0
    except Exception as e:                                        # noqa: BLE001
        print(f"# unreadable ckpt in {run_dir}: {e}", file=sys.stderr)
        return False, 0, nep
    ep = int(step) // int(spe)
    return ep >= nep, ep, nep

eligible, done, unstarted, broken, other_anatomy = [], [], [], [], []
for run in sorted(glob.glob("trained_nets/mg_recon/*/*")):
    if not os.path.isdir(run):
        continue
    run = run.replace(os.sep, "/")      # keep the printed path shell-usable
    # Runs dropped by the ANATOMY filter are COUNTED, not silently ignored.
    # It is the one filter that is a standing default rather than a per-call
    # choice, so it is the one that could quietly leave work behind.
    if anatomy != "all" and run.split("/")[2] != anatomy:
        if os.path.exists(os.path.join(run, "config.json")):
            other_anatomy.append(run)
        continue
    if only and not any(t in run for t in only):
        continue
    if skip and any(t in run for t in skip):
        continue
    if not os.path.exists(os.path.join(run, "config.json")):
        # never reached train.py's save_config -> nothing to resume from
        if os.path.exists(os.path.join(run, "config.gen.json")):
            unstarted.append((run, "no config.json"))
        continue
    try:
        is_done, ep, nep = progress(run)
    except Exception as e:                                        # noqa: BLE001
        broken.append((run, str(e)))
        continue
    if is_done is None:
        unstarted.append((run, "no net.ckpt"))
    elif is_done and not force:
        done.append((run, ep, nep))
    else:
        eligible.append((run, ep, nep))

if include_unstarted:
    for run, _ in unstarted:
        eligible.append((run, 0, 0))
    unstarted = []

w = max([len(r) for r, *_ in eligible + done] + [30])
print(f"\nANATOMY={anatomy}")
print("RESUMABLE")
print(f"{'idx':<6}{'run':<{w+2}}{'epoch':>8} / total")
for i, (run, ep, nep) in enumerate(eligible):
    print(f"{i:<6}{run:<{w+2}}{ep:>8} / {nep}")
if done:
    print("\nCOMPLETE (skipped; FORCE_ALL=1 to include):")
    for run, ep, nep in done:
        print(f"      {run:<{w+2}}{ep:>8} / {nep}")
if unstarted:
    print("\nNO CHECKPOINT (skipped; INCLUDE_UNSTARTED=1 restarts them fresh):")
    for run, why in unstarted:
        print(f"      {run:<{w+2}}  {why}")
if broken:
    print("\nUNREADABLE:")
    for run, why in broken:
        print(f"      {run:<{w+2}}  {why}")
if other_anatomy:
    others = sorted({r.split("/")[2] for r in other_anatomy})
    print(f"\nEXCLUDED BY ANATOMY={anatomy}: {len(other_anatomy)} run(s) under "
          f"{', '.join(others)}")
    print(f"      see them with: ANATOMY={others[0]} (or all)")

# The manifest FREEZES the index the array will resolve. Written even when
# empty, so a later submit against a stale file cannot silently reuse an older,
# longer list.
os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
with open(manifest, "w") as f:
    f.write(f"# mg_recon resume manifest -- anatomy={anatomy}\n")
    f.write(f"# written {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    f.write(f"# {len(eligible)} run(s); array index = line order below\n")
    for run, ep, nep in eligible:
        f.write(f"{run}\n")

n = len(eligible)
print(f"\n{n} run(s) to resume.  manifest: {manifest}")
if n:
    env = "" if anatomy == "knee" else f"ANATOMY={anatomy} "
    print(f"submit with:  {env}sbatch --array=0-{n - 1} "
          f"slurm/mg_recon_resume.sbatch")
else:
    print("nothing to resume.")
PY
    then
        echo "[resume] enumeration FAILED -- the manifest was NOT written, so a" >&2
        echo "         submission cannot silently pick up a stale list." >&2
        echo "         (the error is above; usually a missing conda env)" >&2
        exit 1
    fi
    exit 0
fi

# --- array mode: read the frozen manifest, never re-enumerate --------------
set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

if [ ! -f "${MANIFEST}" ]; then
    echo "[resume] no ${MANIFEST}." >&2
    echo "         The list step writes it, and the array indexes it. Run:" >&2
    echo "           bash $0 --list ${ANATOMY}" >&2
    exit 1
fi

RUNS="$(grep -v '^[[:space:]]*\(#.*\)\?$' "${MANIFEST}" || true)"
if [ -z "${RUNS}" ]; then
    echo "[resume] ${MANIFEST} lists no runs -- nothing to resume." >&2
    echo "         Re-check with: bash $0 --list ${ANATOMY}" >&2
    exit 1
fi

N_TOTAL="$(echo "${RUNS}" | wc -l)"
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[resume] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} runs in ${MANIFEST}" >&2
    echo "         -- fix --array (should be 0-$(( N_TOTAL - 1 )))" >&2
    exit 1
fi

echo "[resume] manifest ${MANIFEST}:"
sed -n '1,3p' "${MANIFEST}" | sed 's/^/         /'

RUN_DIR="$(echo "${RUNS}" | sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p")"
CONFIG="${RUN_DIR}/config.json"

# INCLUDE_UNSTARTED runs have no config.json; fall back to the generated one.
if [ ! -f "${CONFIG}" ]; then
    CONFIG="${RUN_DIR}/config.gen.json"
    echo "[resume] no config.json in ${RUN_DIR}; starting fresh from config.gen.json"
fi
if [ ! -f "${CONFIG}" ]; then
    echo "[resume] ${RUN_DIR} has neither config.json nor config.gen.json" >&2
    exit 1
fi

echo "[resume] task ${SLURM_ARRAY_TASK_ID}/$(( N_TOTAL - 1 )): ${RUN_DIR}"
python - "${CONFIG}" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
print(f"[resume] {cfg['experiment']['name']}  ckpt={cfg['paths'].get('ckpt')}  "
      f"wandb.id={cfg.get('wandb', {}).get('id')}  "
      f"epochs={cfg['training']['num_epochs']} x {cfg['training']['steps_per_epoch']}")
PY

python train.py "${CONFIG}"
