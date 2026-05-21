#!/bin/bash
#SBATCH --job-name=clip_extract_all_ft
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_all_ft_%A_%a.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/clip_extract_all_ft_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Re-extracts CLIP features for ALL games using the fine-tuned checkpoint.
# Outputs: {half}_clip_ft_features.npz alongside every .mkv in soccernet_videos.
# Run after finetune_clip_simple.py finishes — checkpoint at clip_simple_best.pt.
#
# Then run:
#   Moment retrieval (test split): python baseline_clip.py --eval_split test --feature_suffix _ft
#   Action spotting (all games):   python baseline_clip_spotting.py --feature_suffix _ft

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

python extract_clip_features.py --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos --checkpoint /ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_simple/clip_simple_best.pt --output_suffix _ft --job_id "$SLURM_ARRAY_TASK_ID" --num_jobs 8 --device cuda
