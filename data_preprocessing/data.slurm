#!/bin/bash
#SBATCH --job-name=pid_data
#SBATCH --output=/projects/mccleary_group/habjan.e/PID_code/slurm_logs/pid_data.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --partition=short

source activate PIDenv

python3 /projects/mccleary_group/habjan.e/PID_code/pid_mlp/data_preprocessing/data_preprocessing.py