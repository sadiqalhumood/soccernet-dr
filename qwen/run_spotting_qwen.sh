#!/bin/bash
#SBATCH --job-name=spotting_qwen
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/spotting_qwen_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/spotting_qwen_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Zero-shot Qwen action spotting baseline.
# Needs all 1100 _qwen_features.npz files to exist first.
# Run after extract_qwen_features.sh has fully completed.

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/qwen

python baseline_qwen_emb_spotting.py --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos --window_s 10 --index_stride_s 2 --device cuda
