#!/bin/bash
#SBATCH --job-name=qwen_emb_baseline
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_emb_baseline_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_emb_baseline_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Requires A100 (80GB recommended):
#   - Qwen3-VL-Embedding-8B model: ~16GB
#   - Global index float16 (4096-dim, ~1.07M windows): ~8.5GB
#   - Total: ~25GB — fits comfortably

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/qwen

python baseline_qwen_emb.py \
    --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
    --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --window_s 10 \
    --index_stride_s 2 \
    --delta_t 5 \
    --top_k 1 5 10 \
    --device cuda
