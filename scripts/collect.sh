#!/bin/bash

#SBATCH --account=cds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=collect
#SBATCH --output=collect_%A_%a.out

module purge

python -u /scratch/eo41/image-gpt/collect.py

echo "Done"
