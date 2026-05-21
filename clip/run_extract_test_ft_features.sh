#!/bin/bash
#SBATCH --job-name=clip_extract_test_ft
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_test_ft_%A_%a.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_test_ft_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Extracts CLIP features for TEST split videos only, using the fine-tuned checkpoint.
# Output: 1_clip_ft_features.npz and 2_clip_ft_features.npz alongside each test video.
# Run this after fine-tuning is complete.
#
# Then compare on test queries:
#   Zero-shot:   python baseline_clip.py --eval_split test
#   Fine-tuned:  python baseline_clip.py --eval_split test --feature_suffix _ft

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

python extract_clip_features.py \
    --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --split test \
    --checkpoint /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_simple/clip_simple_best.pt \
    --output_suffix _ft \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --num_jobs 8 \
    --device cuda
