#!/bin/bash
#SBATCH --job-name=torch
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=522:00:00
#SBATCH --output slurm.%J.out # %J is job id, given from slurm
#SBATCH --error slurm.%J.err
srun bash child.sh
