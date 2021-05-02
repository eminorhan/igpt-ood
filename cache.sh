#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=cache_igpt
#SBATCH --output=cache_igpt_%A_%a.out

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/cache.py --print_freq 1000 --batch_size 17 --model_size 'l' --prly 25 --partition 0 --fragment 'val' --model_path '/scratch/eo41/image-gpt/models/l/model.ckpt-1000000.index' --cluster_path '/scratch/eo41/image-gpt/models/l/kmeans_centers.npy'

echo "Done"
