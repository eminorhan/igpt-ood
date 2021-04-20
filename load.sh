#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=igpt
#SBATCH --output=igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/load.py

echo "Done"
