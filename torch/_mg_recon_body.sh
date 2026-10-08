# Shared body for torch/mg_recon_{knee,brain}.sbatch and the torch/exp*.sbatch
# launchers -- NOT submittable alone.
#
# The caller sets ANATOMY (knee|brain) and sources this. Everything else is
# identical between the launchers, so it lives here rather than being copied: two
# launchers that drift apart is exactly how a grid ends up half-comparable.
#
# The caller must also set IMMAP_ROOT (the repo checkout) -- it cannot be derived
# here, because sbatch runs a COPY of the launcher from the node's spool dir and
# $BASH_SOURCE points there rather than at the repo.
#
# The caller may also override, before sourcing:
#   CONFIG_ROOT     default "config"
#   SWEEP_EPOCHS    "" = the config's num_epochs; e.g. 20 to probe first. A run
#                   shortened this way is TAGGED (see RUN_TAG), so it never
#                   lands in -- or later blocks -- the full-length cell's run dir.
#   RUN_TAG         suffix for the run dir and the wandb name:
#                   trained_nets/mg_recon/<anatomy>/<model>_R<r>_<tag>.
#                   Unset = "<SWEEP_EPOCHS>ep" when SWEEP_EPOCHS is set, nothing
#                   otherwise. Set it to "" explicitly to put a shortened run in
#                   the canonical dir anyway.
#   REGENERATE      1 = regenerate configs from the generator before running
#   ONLINE_SMAPS_KWS  JSON replacing mri.online_smaps_kws for this launch only,
#                   e.g. '{"thresh_eig": 0.0, "maxit": 10}'. A PROBE knob: it
#                   REQUIRES an explicit non-empty RUN_TAG, so a probe with other
#                   maps can never land in (or block) the cell's own run dir.
#   VAL_EVERY       overrides training.val_every_epochs (e.g. 5 for a short
#                   probe that should still validate and log panels).
#   ONLY            "" = every model tag; e.g. "mglpds mggrouplpds" for a subset
#   ACCELS          "" = every acceleration; e.g. "8" for R=8 only, "8 12" for
#                   two (space-separated; commas are accepted too)
#   ATTN            "" = the generator default (flex); triton|gather to override.
#   FORCE_RESTART   1 = start the cell from scratch even though its run dir
#                   already holds a launch, overwriting it. See LAUNCH-ONCE.
#   ORGAN_MASK      "" = off; 1 restricts the loss, the metrics AND the logged
#                   panel to the coil-sensitivity support. For knee, whose air
#                   background is most of the slice. Runs with it on are NOT
#                   comparable to runs with it off -- the metrics measure a
#                   different region.
#
#                   Set it per ANATOMY, not per experiment. REGENERATE rewrites
#                   every config for the anatomy (ONLY/ACCELS narrow which cell
#                   RUNS, not which configs are WRITTEN), so two launchers that
#                   touch one anatomy with different ORGAN_MASK values will
#                   fight: whichever regenerates last decides, and a resume can
#                   pick up the other one's answer. Currently BOTH anatomies
#                   are 1, set in mg_recon_{knee,brain}.sbatch and every
#                   torch/exp*.sbatch launcher. Change them together.
#                   Set it in the launcher rather than relying on the default,
#                   so the run directory records which REGION it was scored
#                   over. (This said "backend" -- copy-paste from ATTN.)
#   PLANAR_STATE    "" = off; 1 writes training.planar_state=true into every
#                   cell that supports it (lpdsnet, mglpds, mg<M>v<V>): iterates
#                   carried as real [re; im] tensors, thresholding as one fused
#                   kernel forward and backward. A SPEED setting -- same
#                   parameters, same map to fp roundoff -- so unlike every other
#                   key it is NOT part of the launch-once comparison: a cell
#                   launched without it is resumed, not refused, and the resume
#                   runs planar (IMMAP_PLANAR_STATE=1; train.py records it in the
#                   run's config.json, so later resumes keep it). A run that
#                   already has it keeps it whatever this is set to; to switch
#                   one back, resume it with IMMAP_PLANAR_STATE=0 exported.
#
# ONLY / ACCELS narrow AND RENUMBER the cell list, so an experiment that runs a
# subset gets its own dense 0..N-1 array range. The bound is checked at runtime
# against the filtered list, so a stale --array fails loudly rather than
# silently training the wrong cell.
#
# LAUNCH-ONCE, WITH AUTO-RESUME
# -----------------------------
# One cell is often listed by several launchers (brain R8 lpdsnet, mglpds and
# varnetmaps are in both exp1 and exp4; the full grids list everything), and a
# launcher is often resubmitted after a walltime kill. So when the run dir
# already holds a config.gen.json:
#
#   * identical to what would be written now -> the cell was launched before,
#     and what happens depends on where that launch got to:
#       - its job is still RUNNING or QUEUED   -> SKIP. The launching job's id
#         is kept in <run dir>/.launch_job and checked with squeue, so two jobs
#         never train one run dir at once.
#       - its checkpoint has reached num_epochs -> SKIP (complete).
#       - it has a checkpoint short of that     -> RESUME, from the run's own
#         config.json: same wandb run, optimizer, scheduler, step and
#         backtracking state -- what torch/mg_recon_resume.sbatch does, without
#         the listing step. The checkpoint is written only on an IMPROVING
#         epoch, so a resume restarts from the last improvement, not the last
#         epoch.
#       - it never wrote a checkpoint (killed or crashed before its first
#         improving epoch) -> started FRESH from config.gen.json; nothing is
#         lost.
#   * different (a noise, mask or smap_root change, ...) -> REFUSE, naming the
#     keys that differ. Never silently kept, never silently overwritten.
#
# So resubmitting the same launcher is how an unfinished experiment continues.
# FORCE_RESTART=1 overrides all of it and retrains from scratch, overwriting
# the run. A repeated SWEEP_EPOCHS probe is the same tagged cell, so it resumes
# or skips like any other -- force it, or delete the probe's run dir.
#
# WHAT THIS CANNOT SEE: data regenerated IN PLACE. Sensitivity maps (or any
# other input) rewritten under the same path leave the config identical, so a
# run trained on the old data is SKIPPED as if it were current. When data
# changes under a path, move the affected run dirs aside before submitting,
# e.g.  mv trained_nets/mg_recon/brain trained_nets/mg_recon/brain_old

: "${ANATOMY:?the calling sbatch must set ANATOMY=knee|brain}"
: "${IMMAP_ROOT:?the calling sbatch must set IMMAP_ROOT=/path/to/ImMAP}"
CONFIG_ROOT="${CONFIG_ROOT:-config}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"
# `-`, not `:-`: an explicitly EMPTY RUN_TAG is a request, not unset.
RUN_TAG="${RUN_TAG-${SWEEP_EPOCHS:+${SWEEP_EPOCHS}ep}}"
FORCE_RESTART="${FORCE_RESTART:-}"
REGENERATE="${REGENERATE:-1}"
ONLY="${ONLY:-}"
ACCELS="${ACCELS:-}"
# Lists are space-separated; accept commas too ("8,12"), which otherwise reach
# the generator as one unparsable argument.
ONLY="${ONLY//,/ }"
ACCELS="${ACCELS//,/ }"
ATTN="${ATTN:-}"
ORGAN_MASK="${ORGAN_MASK:-}"
# PHASE_CORRECT  "" = off; 1 phase-corrects the online coil maps
#                (--online-phase-correct). Like ORGAN_MASK, regeneration writes
#                every config of the anatomy, so set it the same in every
#                launcher of that anatomy.
PHASE_CORRECT="${PHASE_CORRECT:-}"
PLANAR_STATE="${PLANAR_STATE:-}"

source ~/.bashrc
conda activate gcdl
cd "${IMMAP_ROOT}"

set -eo pipefail
: "${SLURM_ARRAY_TASK_ID:?must be run as an array job (sbatch sets SLURM_ARRAY_TASK_ID)}"

mkdir -p logs

# ---- resolve this task's cell from the generator ------------------------------------------
# --anatomy so each launcher indexes ONLY its own cells: the array bound is
# per-anatomy, and knee task 3 and brain task 3 are different runs.
CELLS="$(python scripts/make_mg_recon_configs.py --list-cells \
    --anatomy "${ANATOMY}" ${ONLY:+--only ${ONLY}} ${ACCELS:+--accels ${ACCELS}})"
N_TOTAL="$(echo "${CELLS}" | wc -l)"

# An array that is too SHORT is the quiet failure: every task succeeds and the
# grid is simply missing cells, with nothing in any log to say so. Warn from the
# task that would have been last. Not fatal -- `sbatch --array=0,6` to rerun two
# cells is a legitimate thing to do -- but it must be visible.
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] \
        && [ "${SLURM_ARRAY_TASK_MAX}" -lt "$((N_TOTAL - 1))" ] \
        && [ "${SLURM_ARRAY_TASK_ID}" -eq "${SLURM_ARRAY_TASK_MAX}" ]; then
    echo "[grid] NOTE: --array tops out at ${SLURM_ARRAY_TASK_MAX} but there are" \
         "${N_TOTAL} ${ANATOMY} cells${ONLY:+ (ONLY=\"${ONLY}\")}." \
         "Cells $((SLURM_ARRAY_TASK_MAX + 1))..$((N_TOTAL - 1)) are NOT being run." >&2
fi

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_TOTAL}" ]; then
    echo "[grid] task ${SLURM_ARRAY_TASK_ID} >= ${N_TOTAL} ${ANATOMY} cells" \
         "${ONLY:+(ONLY=\"${ONLY}\")}${ACCELS:+ (ACCELS=\"${ACCELS}\")} -- fix --array"
    echo "       (should be 0-$(( N_TOTAL - 1 )))"
    exit 1
fi

CELL="$(echo "${CELLS}" | awk -v i="${SLURM_ARRAY_TASK_ID}" '$1 == i')"
RTAG="$(echo "${CELL}" | cut -f3)"          # e.g. R8
MODEL="$(echo "${CELL}" | cut -f4)"

BASE_CONFIG="${CONFIG_ROOT}/${ANATOMY}/mg/${MODEL}_${RTAG}.json"

echo "[grid] ${ANATOMY} task ${SLURM_ARRAY_TASK_ID}/$(( N_TOTAL - 1 )): ${RTAG} ${MODEL}"
echo "       config: ${BASE_CONFIG}"

# ---- (re)generate the configs -------------------------------------------------------------
# Cheap, and it keeps a stale hand-edited config from silently deciding a run.
if [ "${REGENERATE}" = "1" ]; then
    # PROTOCOL=legacy reproduces the old synthetic grid; unset = the
    # generator's default (measured). See the generator's module docstring.
    python scripts/make_mg_recon_configs.py --out "${CONFIG_ROOT}" \
        --anatomy "${ANATOMY}" ${ATTN:+--attn "${ATTN}"} \
        ${ORGAN_MASK:+--organ-mask} ${PROTOCOL:+--protocol "${PROTOCOL}"}         ${PHASE_CORRECT:+--online-phase-correct} ${PLANAR_STATE:+--planar-state} >/dev/null
fi

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[grid] ${BASE_CONFIG} does not exist. Generate it with:"
    echo "       python scripts/make_mg_recon_configs.py --out ${CONFIG_ROOT}"
    exit 1
fi

RUN_DIR="trained_nets/mg_recon/${ANATOMY}/${MODEL}_${RTAG}${RUN_TAG:+_${RUN_TAG}}"
GEN_CONFIG="${RUN_DIR}/config.gen.json"
LAUNCH_LOCK="${RUN_DIR}/.launch_job"     # SLURM job id of the last launch
mkdir -p "${RUN_DIR}"

# Is SLURM job $1 still queued or running (and not this job)?
_job_alive() {
    local id="$1"
    [ -n "${id}" ] || return 1
    [ "${id}" = "${SLURM_JOB_ID:-}" ] && return 1
    if ! command -v squeue >/dev/null 2>&1; then
        echo "[grid] WARNING: squeue not found; cannot tell whether job ${id}" \
             "still owns ${RUN_DIR}. Assuming it does not." >&2
        return 1
    fi
    [ -n "$(squeue -h -j "${id}" 2>/dev/null)" ]
}

# ---- per-cell bookkeeping (epoch override, run tag, fresh wandb run, launch-once) ---------
# Exit status 3 means "already launched with this exact config -- nothing to do";
# any other non-zero status is a real refusal and fails the task.
set +e
FORCE_RESTART="${FORCE_RESTART}" RUN_TAG="${RUN_TAG}" \
python - "${BASE_CONFIG}" "${GEN_CONFIG}" "${RUN_DIR}" "${SWEEP_EPOCHS}" <<'PY'
import json, os, sys

# write_config, NOT json.dump: train.py reads this file back with yaml.safe_load,
# where json's `1e-06` for eta_min comes back as the STRING '1e-06' and
# CosineAnnealingLR dies doing arithmetic on it. See training/common.py.
from training.common import write_config

base, out, save_dir = sys.argv[1:4]
epochs = sys.argv[4] if len(sys.argv) > 4 else ""
tag = os.environ.get("RUN_TAG", "")
force = os.environ.get("FORCE_RESTART") == "1"

with open(base) as f:
    cfg = json.load(f)

if cfg.get("task") != "recon":
    raise SystemExit(f"[grid] {base} has task={cfg.get('task')!r}, expected 'recon'.")

mri = cfg["mri"]
# The two kspace_types the generator writes: "simulated" (--protocol legacy)
# and "measurement_awgn" (--protocol measured, measured k-space + known AWGN).
# Plain "measurement" is still refused -- it adds no noise and pins sigma, so a
# noise-adaptive net would be told a level that is not in its data.
if mri.get("kspace_type") not in ("simulated", "measurement_awgn"):
    raise SystemExit(
        f"[grid] {base} has kspace_type={mri.get('kspace_type')!r}. This grid "
        f"trains on 'simulated' (legacy) or 'measurement_awgn' (measured) k-space; "
        f"regenerate with scripts/make_mg_recon_configs.py.")

# The expected range is READ FROM THE GENERATOR, not written here as well. It
# was duplicated as a literal (0.0, 0.01) and that made changing the noise level
# a two-file edit whose second half fails at submit time, on every cell at once.
# The generator's module level is stdlib-only -- write_config is imported lazily
# inside main() -- so this import is safe on a node without torch.
# Same relative path the REGENERATE step above invokes; the body runs from the
# repo root.
import importlib.util
try:
    _spec = importlib.util.spec_from_file_location(
        "_mkcfg", "scripts/make_mg_recon_configs.py")
    _gen = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gen)
    # NOISE_STD is keyed BY ANATOMY (brain and knee train at different sigmas),
    # so the range to compare against comes from the config's own anatomy. The
    # isinstance check keeps this working if it is ever flattened back.
    _ns = _gen.NOISE_STD
    if isinstance(_ns, dict):
        _ns = _ns[cfg["data"]["train"]["anatomy"]]
    _expected = tuple(float(v) for v in _ns)
except Exception as _e:                 # noqa: BLE001
    # Do not block the run on a check that cannot be performed -- but say so,
    # because a silently skipped guard is worse than none.
    print(f"[grid] WARNING: could not read NOISE_STD from the generator "
          f"({_e}); the noise-range check is SKIPPED.", file=sys.stderr)
    _expected = None

lo, hi = cfg["training"]["noise_std"]
if _expected is not None and (float(lo), float(hi)) != _expected:
    raise SystemExit(
        f"[grid] {base} has noise_std={[lo, hi]}, but the generator's "
        f"NOISE_STD is {list(_expected)}. The config is stale -- regenerate it "
        f"(REGENERATE=1) rather than editing it by hand.")
if hi <= lo:
    raise SystemExit(f"[grid] {base} has noise_std={[lo, hi]}: hi must exceed lo.")

# preproc='image' pads y~ but leaves E's mask/maps behind, and subtracts a plain
# mean where reconstruction needs the E^H E DC correction. The models raise on
# the first of those, but catching it here names the config, not a tensor shape.
#
# Only models that HAVE the knob are subject to it. A baseline reconstructing
# straight from k-space -- E2EVarNet -- has no preprocessing stage at all, so
# `preproc` is absent from its params and there is nothing to check. Keyed on
# the key's presence rather than an exemption list, so the next baseline that
# lacks it does not have to be added here.
#
# AltSplitCDLNet stays exempt for a different reason: with smap_update its
# unroll re-forms E^H y per layer.
params = cfg["model"]["params"]
if ("preproc" in params and cfg["model"]["type"] != "AltSplitCDLNet"
        and params["preproc"] not in ("kspace", "identity")):
    raise SystemExit(
        f"[grid] {base}: model.params.preproc={params['preproc']!r}. "
        f"Reconstruction needs 'kspace' (pads the operator, removes DC through "
        f"E^H E) or 'identity' (no preprocessing at all).")

paths = cfg.setdefault("paths", {})
paths["save_dir"] = save_dir
paths["ckpt"] = None                    # fresh run; train.py rewrites this to net.ckpt
cfg.setdefault("wandb", {})["id"] = None
if epochs:
    cfg["training"]["num_epochs"] = int(epochs)
    cfg["scheduler"]["params"]["T_max"] = int(epochs) * cfg["training"]["steps_per_epoch"]
# Probe overrides. ONLINE_SMAPS_KWS changes the OPERATOR (the maps every step
# sees), so it is refused without an explicit tag: under the default tag a
# 0.95 and a 0.0 probe of the same length would share one run dir.
_kws = os.environ.get("ONLINE_SMAPS_KWS", "").strip()
if _kws:
    if not os.environ.get("RUN_TAG"):
        raise SystemExit("[grid] ONLINE_SMAPS_KWS needs an explicit RUN_TAG "
                         "(e.g. RUN_TAG=te0), so the probe gets its own run dir.")
    if cfg["mri"].get("online_smaps") != "espirit":
        raise SystemExit(f"[grid] ONLINE_SMAPS_KWS set, but {base} has "
                         f"online_smaps={cfg['mri'].get('online_smaps')!r}; the "
                         f"kws are ESPIRiT's.")
    cfg["mri"]["online_smaps_kws"] = json.loads(_kws)
_ve = os.environ.get("VAL_EVERY", "").strip()
if _ve:
    cfg["training"]["val_every_epochs"] = int(_ve)
# The tag goes in the wandb name as well as the run dir: a shortened run must
# not share a name with the full-length cell it samples.
if tag:
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{tag}"

# ---- launch-once: never restart a launched cell silently, never keep a stale one ----------
if os.path.exists(out) and not force:
    with open(out) as f:
        old = json.load(f)

    def _flat(d, p=""):
        if not isinstance(d, dict):
            return {p: d}
        acc = {}
        for k, v in d.items():
            if k == "_comment":                       # documentation, not setup
                continue
            if p or k not in ("paths", "wandb"):      # bookkeeping, not setup
                acc.update(_flat(v, f"{p}.{k}" if p else k))
        return acc

    # Execution-only keys: they change how fast the net runs, not what it
    # computes (same parameters, same map to fp roundoff, checkpoints
    # interchangeable), so a difference in one is not a different experiment.
    _EXEC_KEYS = ("training.planar_state",)
    a, b = _flat(old), _flat(cfg)
    diff = sorted(k for k in set(a) | set(b)
                  if a.get(k) != b.get(k) and k not in _EXEC_KEYS)
    if not diff:
        print(f"[grid] {save_dir}: already launched with this exact config "
              f"(possibly by another launcher) -- checking whether to skip or "
              f"resume it. FORCE_RESTART=1 starts it over.")
        sys.exit(3)
    raise SystemExit(
        f"[grid] {save_dir} was already launched, under a config that no longer "
        f"matches the generator's -- differing in "
        f"{diff[:8]}{' ...' if len(diff) > 8 else ''}. Not keeping it silently "
        f"and not overwriting it silently. Resubmit with FORCE_RESTART=1 to "
        f"retrain it from scratch under the current config (this overwrites the "
        f"run), or resume it with torch/mg_recon_resume.sbatch if the old setup "
        f"is what you want.")

write_config(cfg, out)

_acs = (f"cf{mri['center_frac']}" if mri.get("center_frac") is not None
        else mri.get("acs_lines"))
_r = f"{mri['R']}{'(eff)' if mri.get('adjust_accel') else '(nominal)'}"
print(f"[grid] online_smaps={mri.get('online_smaps')} kws={mri.get('online_smaps_kws')}")
print(f"[grid] {cfg['model']['type']} R={_r} acs={_acs} "
      f"sigma~U{cfg['training']['noise_std']} epochs={cfg['training']['num_epochs']} -> {out}")
PY
_rc=$?
set -e
TRAIN_CONFIG="${GEN_CONFIG}"
if [ "${_rc}" -eq 3 ]; then
    # Launched before under this exact config: running, complete, resumable,
    # or never checkpointed. See LAUNCH-ONCE at the top.
    _owner="$(cat "${LAUNCH_LOCK}" 2>/dev/null || true)"
    if _job_alive "${_owner}"; then
        echo "[grid] SKIP ${RUN_DIR}: still queued/running as job ${_owner}."
        exit 0
    fi
    _state="$(python - "${RUN_DIR}" <<'PY'
import json, os, sys
import torch

run = sys.argv[1]
cfg_path, ckpt = os.path.join(run, "config.json"), os.path.join(run, "net.ckpt")
if not os.path.exists(cfg_path):
    print("fresh no config.json (train.py never got past wandb.init)"); sys.exit()
if not os.path.exists(ckpt):
    print("fresh no net.ckpt (stopped before its first improving epoch)"); sys.exit()
t = json.load(open(cfg_path))["training"]
def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:            # torch < 1.13 has no weights_only
        return torch.load(path, map_location="cpu")
try:
    step = _load(ckpt).get("step") or 0
except Exception as e:                                            # noqa: BLE001
    # Never restart over an unreadable checkpoint: that would discard the run.
    sys.exit(f"[grid] cannot read {ckpt} ({e}); not resuming, not restarting. "
             f"Inspect it, or FORCE_RESTART=1 to retrain from scratch.")
ep, nep = int(step) // int(t["steps_per_epoch"]), int(t["num_epochs"])
print(f"{'done' if ep >= nep else 'resume'} epoch {ep}/{nep} (step {step})")
PY
)"
    case "${_state%% *}" in
        done)
            echo "[grid] SKIP ${RUN_DIR}: complete -- ${_state#* }."
            exit 0 ;;
        resume)
            echo "[grid] RESUME ${RUN_DIR} at ${_state#* } from its config.json"
            TRAIN_CONFIG="${RUN_DIR}/config.json" ;;
        fresh)
            echo "[grid] RESTART ${RUN_DIR} from scratch: launched before, but" \
                 "${_state#* }." ;;
        *)
            echo "[grid] could not determine the state of ${RUN_DIR}" >&2
            exit 1 ;;
    esac
    # A cell launched before the planar state existed (or without it) carries
    # no such key in the config it continues from. If the generator now writes
    # it for this cell, switch it on through the environment: train.py lets
    # that override the config and records it in config.json.
    if [ "$(python -c "import json, sys; print(int(bool((json.load(open(sys.argv[1])).get('training') or {}).get('planar_state'))))" "${BASE_CONFIG}")" = "1" ] \
            && [ -z "${IMMAP_PLANAR_STATE:-}" ]; then
        export IMMAP_PLANAR_STATE=1
        echo "[grid] planar state ON for this run (IMMAP_PLANAR_STATE=1)."
    fi
elif [ "${_rc}" -ne 0 ]; then
    exit "${_rc}"
fi

echo "${SLURM_JOB_ID:-}" > "${LAUNCH_LOCK}"
python train.py "${TRAIN_CONFIG}"
