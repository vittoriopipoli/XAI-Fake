#!/bin/bash
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --job-name="xai_dev"
#SBATCH --array=1-1

source venv/bin/activate

python main.py --config configs/resnet18_binary_classification.yaml --verbose

