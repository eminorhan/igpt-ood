#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=cache_igpt
#SBATCH --output=cache_igpt_%A_%a.out
#SBATCH -a 0-5

PARRAY=(10 11 12 13 14 15)    

p=${PARRAY[`expr $SLURM_ARRAY_TASK_ID % ${#PARRAY[@]}`]}
v=`expr $SLURM_ARRAY_TASK_ID / ${#PARRAY[@]}`

module purge
module load cuda/11.1.74

python -u /scratch/eo41/image-gpt/cache.py --print_freq 10 --batch_size 64 --model_size 's' --prly $p --partition $v --fragment 'val' --model_path '/scratch/eo41/image-gpt/models/s/model.ckpt-1000000.index' --cluster_path '/scratch/eo41/image-gpt/models/s/kmeans_centers.npy' --val_path '/scratch/eo41/imagenet_a/'

echo "Done"
