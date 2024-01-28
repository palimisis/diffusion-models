#!/bin/bash 
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -e
# Activate Anaconda work environment 
# source /home/$USER/minicoda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate pt


python3 /home/it21902/diffusion-models/train.py $@
# python3 /home/it21902/diffusion-models/distributed_example.py $@
