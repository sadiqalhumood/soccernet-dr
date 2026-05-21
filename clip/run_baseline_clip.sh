#!/bin/bash
#SBATCH --job-name=clip_baseline
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_baseline_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_baseline_%j.err

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

python baseline_clip.py \
    --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
    --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --window_s 10 \
    --index_stride_s 2 \
    --delta_t 5 \
    --top_k 1 5 10 \
    --device cuda
