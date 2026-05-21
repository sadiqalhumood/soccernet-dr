#!/bin/bash
#SBATCH --job-name=clip_extract_best
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_best_%A_%a.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_best_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Extract test-split features using the clip_ft_best checkpoint.
# Output: 1_clip_best_features.npz and 2_clip_best_features.npz alongside each test video.
# Submit after clip_ft_best training finishes.

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

python extract_clip_features.py \
    --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
    --split test \
    --checkpoint /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_best/best.pt \
    --output_suffix _best \
    --job_id "$SLURM_ARRAY_TASK_ID" \
    --num_jobs 8 \
    --device cuda
