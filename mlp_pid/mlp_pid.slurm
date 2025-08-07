#!/bin/bash
#SBATCH --job-name=mlp_pid
#SBATCH --output=/projects/mccleary_group/habjan.e/PID_code/slurm_logs/mlp_pid.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate PIDenv

python3 /projects/mccleary_group/habjan.e/PID_code/pid_mlp/mlp_pid/mlp_pid.py