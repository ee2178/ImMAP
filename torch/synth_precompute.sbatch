#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8           # GPU inference + h5 IO; 8 is plenty
#SBATCH --mem=32GB                  # one subject volume in memory at a time -> light
#SBATCH --time=02:00:00             # inference over train+val subjects; lower for faster scheduling
#SBATCH --job-name=SynthPrecompute-BraTS
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ee2178@nyu.edu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# --- environment ---
source ~/.bashrc                    # ensure conda is available
conda activate env                 # replace with your actual env name

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- run ---
cd ~/scratch/ee2178/ImMAP           # repo root (where the script + config live)

echo "Job $SLURM_JOB_ID on $(hostname)  |  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Precompute the e2e synthesis output yhat = net([flair,t1,t2]) and write it into each *_img.h5
# (dataset key 'yhat'), for use as the I2SB bridge prior x1 (yhat -> T1ce). Point SYNTH_CFG at the
# TRAINED synth run's SAVED config.json (it carries paths.ckpt). Idempotent: existing 'yhat' is
# skipped unless you add --overwrite.
SYNTH_CFG=trained_nets/brats/Synth_T1ce_Pretrain_VGG_CosLR/config.json   # edit to your synth run
DATA_DIR=/home/ee2178/scratch/ee2178/datasets/BraTS

for split in train val; do
    echo "=== precompute yhat: $split ==="
    python3 datasets/BraTS/synth_precompute.py "$SYNTH_CFG" "$DATA_DIR/BraTS2021_DataSet_$split"
done

echo "Done  |  $(date)"
