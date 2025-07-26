#!/bin/bash
#SBATCH --job-name=neutral_mlp
#SBATCH --output=/projects/mccleary_group/habjan.e/PID_code/slurm_logs/neutral_mlp.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate PIDenv

python3 /projects/mccleary_group/habjan.e/PID_code/pid_mlp/mlp_pid/neutral_mlp.py