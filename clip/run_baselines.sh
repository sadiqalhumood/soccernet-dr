#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/baselines_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/baselines_%j.err

mkdir -p /ibex/scratch/alhumosm/SoccerNet/logs
mkdir -p /home/alhumosm/soccernet/clip/results

source ~/.bashrc
conda activate soccernet

cd /home/alhumosm/soccernet/clip

CAPTION_ROOT=/ibex/scratch/alhumosm/SoccerNet/caption-2024
FEATURE_ROOT=/ibex/scratch/alhumosm/SoccerNet/soccernet_videos

echo "=============================="
echo "1/3: Oracle (Time + Order)"
echo "=============================="
python baseline_oracle_time_order.py \
    --caption_root "$CAPTION_ROOT" \
    --delta_t 5 \
    --top_k 1 5 10

echo "=============================="
echo "2/3: Oracle (Time)"
echo "=============================="
python baseline_oracle_time.py \
    --caption_root "$CAPTION_ROOT" \
    --delta_t 5 \
    --top_k 1 5 10 \
    --seed 42

echo "=============================="
echo "3/3: Random"
echo "=============================="
python baseline_random.py \
    --caption_root "$CAPTION_ROOT" \
    --feature_root "$FEATURE_ROOT" \
    --delta_t 5 \
    --top_k 1 5 10 \
    --seed 42

echo "All baselines done."
