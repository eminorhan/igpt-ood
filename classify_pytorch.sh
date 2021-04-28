#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=classify_pt
#SBATCH --output=classify_pt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/classify_pytorch.py --model_size 's' --prly 15 --batch_size 256 --print_freq 1000 --epochs 250

echo "Done"
