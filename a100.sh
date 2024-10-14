#!/bin/bash

## Job name is unspecified, do it manually in the command line with `-J job_name`
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gpus=a100-40:1
#SBATCH --constraint=xgph
#SBATCH --time=00:01:00
#SBATCH --output=log/%x-%j.slurmlog
#SBATCH --error=log/%x-%j.slurmlog

job_name=$SLURM_JOB_NAME

echo "Running on $(hostname)"
echo "Job started: $(date)"

echo "Job name: $job_name"
echo "Args: $@"

nvidia-smi 
NVCC=/usr/local/cuda/bin/nvcc

[[ -f $NVCC ]] || { echo "ERROR: NVCC Compiler not found at $NVCC, exiting..."; exit 1; }
echo "NVCC info: $($NVCC --version)"

echo -e "\n====> Compiling... (REMEMBER TO CHECK MAKE OUTPUT)\n"

# make

echo -e "\n====> Running...\n"

srun --cpu-bind core $@

echo -e "\n====> Finished running.\n"

echo "Job ended: $(date)"
