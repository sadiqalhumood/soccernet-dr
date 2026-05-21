#!/bin/bash
#SBATCH --job-name=clip_ft_simple
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_ft_simple_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_ft_simple_%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# Runs 1 epoch of simple CLIP fine-tuning (1 positive + 1 negative per caption).
# To train epoch 2: change --epochs to 2 and re-submit. Script auto-resumes.
# To train epoch 3: change --epochs to 3 and re-submit. And so on.

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs
mkdir -p /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_simple

source ~/.bashrc
conda activate soccernet

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/alhumosm/soccernet/clip

python finetune_clip_simple.py \
    --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
    --video_root   /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --output_dir   /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_simple \
    --epochs 1 \
    --batch_size 4 \
    --n_frames 4 \
    --window_s 2.5 \
    --lr 1e-6 \
    --num_workers 4 \
    --device cuda
