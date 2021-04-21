#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_igpt
#SBATCH --output=train_igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/train.py '/scratch/eo41/minGPT/saycam_labeled'

echo "Done"
