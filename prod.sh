#!/bin/bash
#SBATCH --partition=prod
#SBATCH --gres=gpu:1
#SBATCH --job-name="xai_prod"
#SBATCH --array=1-1

source venv/bin/activate
python main.py --config configs/gen-training-unet-trans_train.yaml --verbose