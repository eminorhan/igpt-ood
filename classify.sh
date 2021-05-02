#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=classify
#SBATCH --output=classify_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/classify.py --model_size 's' --prly 12 --batch_size 128 --print_freq 10000 --epochs 300

echo "Done"
