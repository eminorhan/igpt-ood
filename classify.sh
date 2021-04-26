#!/bin/bash

#SBATCH --account=cds
##SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=62GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=classify
#SBATCH --output=classify_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/classify.py

echo "Done"
