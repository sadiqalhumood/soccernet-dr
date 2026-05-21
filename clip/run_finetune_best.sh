#!/bin/bash
#SBATCH --job-name=clip_ft_best
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_ft_best_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_ft_best_%j.err

# Best fine-tuning run — 1 epoch on A100, all issues addressed:
#   - InfoNCE loss (batch=32, 31 negatives per positive)
#   - Visual encoder frozen (prevent catastrophic forgetting)
#   - Window=±5s (matches eval)
#   - LR warmup + cosine decay
#   - Gradient clipping
#   - Both anonymized + description captions (~42k samples vs 21k before)
#
# Estimated time: ~20 min on A100

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

python finetune_clip_best.py \
    --epochs 1 \
    --batch_size 32 \
    --window_s 5.0 \
    --lr 1e-5 \
    --temperature 0.07 \
    --warmup_frac 0.1 \
    --max_norm 1.0 \
    --output_dir /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_best
