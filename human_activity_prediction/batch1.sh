#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=team06
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.2
export WANDB_API_KEY=94a2bd29b83d7c325f6b06587474e39f855f79f8
# Run your python code
python3 main.py
