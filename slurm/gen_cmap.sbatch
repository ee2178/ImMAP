#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_89_tandon_advanced
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8           # mostly GPU + nifti/h5 IO; 8 is plenty
#SBATCH --mem=32GB                  # one volume at a time -> light; 32GB is generous
#SBATCH --time=01:00:00             # ~1251 subjects; lower it for faster scheduling
#SBATCH --job-name=CMapGen-BraTS
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

python3 preprocessing/cmap.py --config config/BraTS/cmap.yaml

echo "Done  |  $(date)"
