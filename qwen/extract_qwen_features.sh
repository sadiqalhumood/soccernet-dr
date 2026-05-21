#!/bin/bash
#SBATCH --job-name=qwen_extract
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_extract_%A_%a.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_extract_%A_%a.err
#SBATCH --array=0-1099%16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:35:00

# Qwen3-VL-Embedding-8B requires A100 (40 or 80GB).
# Do NOT use V100 (16GB) — the 8B model will OOM.
# 1100 tasks (one per video half), max 16 running at once (%16).
# Estimated time: ~10-15 min per half → 30 min limit is safe.
# Advantage over large batches: a failed task loses 1 video, not 69.

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/qwen

python extract_qwen_features.py \
    --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --fps 2.0 \
    --batch_size 16 \
    --model_name Qwen/Qwen3-VL-Embedding-8B \
    --device cuda \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --num_jobs "$SLURM_ARRAY_TASK_COUNT"
