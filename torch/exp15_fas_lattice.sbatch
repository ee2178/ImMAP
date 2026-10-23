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
#SBATCH --job-name=Exp15-FASLat
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.out
#SBATCH --error=/home/ee2178/scratch/ee2178/ImMAP/torch/logs/slurm_%A_%a.err
#SBATCH --array=0-4
#
# EXPERIMENT 15 -- V-CYCLE CORRECTNESS FIXES vs mg81v6, KNEE R=8, FULL LENGTH.
#
#   0   knee  R8   mg81v6vtx    lattice fix,  legacy dual FAS, untied FAS copies
#   1   knee  R8   mg81v6vtxA   lattice fix + FAS option A   (fas_dual=fixed_point, fas_copies=tied)
#   2   knee  R8   mg81v6vtxB   lattice fix + FAS option B   (fas_dual=gradient,    fas_copies=tied)
#   3   knee  R8   mg81v6vtxAnest   = vtxA + fas_parent_pi=true   (option A, tied, parent fix)
#   4   knee  R8   mg81v6vtxBnest   = vtxB + fas_parent_pi=true   (option B, tied, parent fix)
#
# OPTION B IS THE DEFAULT (2026-10-23) for new multigrid cells: A trained to
# better numbers, but B is the more accurate scheme, so B + tied copies + the
# parent-pi fix is the standard FAS (make_mg_recon_configs.py::MG_FAS). Cell 3
# keeps A with the same fixes as the comparison.
#
# PER-LEVEL SUPERVISION is NOT in this array (2026-10-23). The cell is defined
# (mg81v6vtxBnestsup = vtxBnest + training.level_loss_weight 0.1) and can be
# launched on its own when wanted:
#   ONLY="mg81v6vtxBnestsup" sbatch --array=0 torch/exp15_fas_lattice.sbatch
#
# NESTED FAS (cells 3, 4). Level 2's correction must restrict level 1's
# residual NET OF level 1's own pi: pi^2 = r_2(R u_1) - R(r_1(u_1) - pi^1).
# Cells 0-2 restrict r_1 alone (as mg81v6 and mg_lpds.jl always have), so a
# SOLVED level 1 still moves level 2 by R pi^1 -- and pi^1 grows as the coarse
# filters train away from the fine ones. Cells 1, 2 are kept exactly as
# launched so their run dirs resume (models/mg_lpds.py::_check_fas_parent_pi).
#
# TO ADD 3, 4 WHILE 0-2 ARE RUNNING, submit only the new indices (a cell already
# running or finished under its name is resumed or skipped by the launch guard):
#   sbatch --array=3-4 torch/exp15_fas_lattice.sbatch
#
# The FAS cells TIE the downsampler's operators to the first fine / coarse
# layers' (fas_copies=tied): with its own trained copies the FAS correction
# carries their drift into the coarse level and no dual placement stays
# coherent after training (models/mg_lpds.py::FAS_COPIES). Tied, mg81v6 loses
# the copies' 22% of its parameters (1.73M -> 1.35M).
#
# The lattice fix (transfer_filter=sinc_vertex) is ON in every cell: it is a
# correctness fix, the standard for every multigrid cell from now on
# (make_mg_recon_configs.py::MG_TRANSFER_FILTER), not an ablation arm. So
#     mg81v6 -> mg81v6vtx            what the lattice fix is worth
#     mg81v6vtx -> vtxA / vtxB        what a coherent FAS (tied copies + option
#                                     A or B) is worth on top of it
#     vtxA -> vtxAnest, vtxB -> vtxBnest   what the parent-pi fix is worth
#     vtxAnest vs vtxBnest            A vs B with every fix in
#
# (Index order is the generator's MODELS order, not the order of ONLY.)
#
# BASELINE: exp11's knee/mg81v6_R8 (MGLPDS-Small), NOT retrained here. Every
# cell is mg81v6 with only `model.params.transfer_filter`, `.fas_dual`,
# `.fas_copies` and `.fas_parent_pi` changed (checked config-for-config against
# the regenerated mg81v6_R8 on 2026-10-22 for cells 0-2: those keys, the run dir
# and the wandb name differ, nothing else).
#
# WHAT THE TWO FIXES ARE (scripts/make_mg_recon_configs.py, EXP15):
#   LATTICE  the stride-2 P=7 conv puts latent sample i ON image pixel 2i, the
#            cell-centred sinc puts coarse pixel j at 2j + 1/2: the coarse image
#            and latent grids are offset by 0.5 / 1.5 / 3.5 fine px at levels
#            1 / 2 / 3. `sinc_vertex` (7 taps, same |H| to within 0.02) is the
#            vertex-centred sinc: offset 0 everywhere.
#   FAS      the dual correction pi_z is subtracted inside the prox ("legacy"),
#            so a coarse level moves even when the fine level is solved. Option
#            A ("fixed_point") adds it after the prox: coherent always. Option B
#            ("gradient") forms it from the dual gradient, inside the prox:
#            coherent while R z stays in the coarse dual set.
#   tests/test_fas_lattice.py checks all of this.
#
# NOTE on the wandb val curves: mg81v6_R8 was launched before 2026-10-20 and
# validates on measured k-space; configs generated now do too (the generator no
# longer writes val_kspace_type), so the curves are comparable. Compare the
# final nets with main_eval anyway -- the SAME measurement for every run:
#     RUNS="knee/mg81v6_R8 knee/mg81v6vtx_R8 knee/mg81v6vtxA_R8 knee/mg81v6vtxB_R8 knee/mg81v6vtxAnest_R8 knee/mg81v6vtxBnest_R8" \
#     OUT=results/exp15_knee_R8 VAL_KSPACE=measurement_awgn sbatch --array=0-5 torch/main_eval.sbatch
#
# A quick look before they finish: notebooks/vcycle_levels_real.ipynb on any of
# these run dirs (the level figures read transfer_filter / fas_dual from the
# run's config like every other key).
#
# EXISTING RUN DIRS. LAUNCH-ONCE (torch/_mg_recon_body.sh) resumes an
# unfinished trained_nets/mg_recon/knee/<cell>_R8 and skips a finished one.
# These names are new, so nothing should be there:
#     ls -d trained_nets/mg_recon/knee/mg81v6{vtx,vtxA,vtxB,vtxAnest,vtxBnest}_R8
#
# FULL LENGTH: 300k steps, as exp11 / exp14; a cell may need a second 48 h
# slot. When a slot ends, RESUBMIT THIS FILE: unfinished cells resume from
# their checkpoint, finished or still-queued ones are skipped.
#
# SUBMIT FROM THE REPO ROOT.
#   sbatch torch/exp15_fas_lattice.sbatch
#   ONLY="mg81v6vtxA" sbatch --array=0 torch/exp15_fas_lattice.sbatch     # one cell
#   SWEEP_EPOCHS=20 ONLY="mg81v6vtxB" sbatch --array=0 torch/exp15_fas_lattice.sbatch   # smoke probe
#
# Check the index map any time:
#   python scripts/make_mg_recon_configs.py --list-cells --anatomy knee \
#       --accels 8 --only mg81v6vtx mg81v6vtxA mg81v6vtxB mg81v6vtxAnest mg81v6vtxBnest

ANATOMY=knee
ONLY="${ONLY:-mg81v6vtx mg81v6vtxA mg81v6vtxB mg81v6vtxAnest mg81v6vtxBnest}"
ACCELS="${ACCELS:-8}"
PLANAR_STATE="${PLANAR_STATE-1}"     # as exp11; `-`, so an explicitly EMPTY value turns it off
ATTN="flex"                          # matches the other knee launchers --
ORGAN_MASK=1                         # regeneration rewrites every knee config,
                                     # so a mismatch silently changes the anatomy
CONFIG_ROOT="config"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-}"     # empty = the config's num_epochs (full length)
REGENERATE=1                         # 1 = regenerate configs before running

# sbatch runs a COPY of this file from the node's spool dir, so a
# $BASH_SOURCE-relative path looks for the body next to that copy -- which was
# never shipped. Resolve the checkout explicitly.
for _c in "${IMMAP_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$HOME/scratch/ee2178/ImMAP"; do
    if [ -n "${_c}" ] && [ -f "${_c}/torch/_mg_recon_body.sh" ]; then
        IMMAP_ROOT="${_c}"
        break
    fi
done
if [ ! -f "${IMMAP_ROOT:-}/torch/_mg_recon_body.sh" ]; then
    echo "[exp15] cannot locate the ImMAP checkout (tried IMMAP_ROOT," \
         "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}, ~/scratch/ee2178/ImMAP)." >&2
    echo "[exp15] submit from the repo root, or: IMMAP_ROOT=/path/to/ImMAP sbatch $0" >&2
    exit 1
fi

source "${IMMAP_ROOT}/torch/_mg_recon_body.sh"
