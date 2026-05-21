#!/bin/bash
#SBATCH --job-name=clip_spotting
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_spotting_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_spotting_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Runs all three action spotting baselines sequentially.
# Evaluation: global/within-game AP at delta_t = 1, 2, 5s (17-class mAP).
#
#   Oracle(Time)  — random within-game AP  (~5 min,  no GPU needed)
#   Oracle(Order) — CLIP  within-game AP  (~30 min, GPU)
#   CLIP cross-video — global mAP         (~60 min, GPU)

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

FEATURE_ROOT=/ibex/scratch/alhumosm/SoccerNet/soccernet_videos

echo "=== Oracle(Time) ==="
python baseline_oracle_time_spotting.py \
    --feature_root $FEATURE_ROOT

echo "=== Oracle(Order) ==="
python baseline_oracle_order_spotting.py \
    --feature_root $FEATURE_ROOT \
    --window_s 10 \
    --device cuda

echo "=== CLIP cross-video ==="
python baseline_clip_spotting.py \
    --feature_root $FEATURE_ROOT \
    --window_s 10 \
    --index_stride_s 2 \
    --device cuda
