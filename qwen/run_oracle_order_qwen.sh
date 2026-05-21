#!/bin/bash
#SBATCH --job-name=qwen_oracle_order
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_oracle_order_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_oracle_order_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Oracle(Order) for Qwen: given the correct game, rank windows within it
# using Qwen3-VL-Embedding text encoder vs pre-extracted visual features.
#
# Diagnostic: if R@1 > 0%, text encoding is fine and cross-video 0% is
# just noise from global ranking. If R@1 == 0%, text encoding is broken.

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/qwen

python baseline_oracle_order_qwen.py \
    --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
    --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --window_s 10 \
    --delta_t 5 \
    --top_k 1 5 10 \
    --device cuda
