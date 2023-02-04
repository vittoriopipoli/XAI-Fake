#!/bin/bash
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --job-name="xai_prod"
#SBATCH --array=1-2

python main.py --config configs/resnet18_binary_classification.yaml --verbose