#!/bin/bash
#SBATCH --job-name=qwen_eval_test
#SBATCH --output=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_eval_test_%j.out
#SBATCH --error=/ibex/scratch/alhumosm/SoccerNet/logs/qwen_eval_test_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

source ~/.bashrc
conda activate soccernet
cd /home/alhumosm/soccernet/qwen

CAPTION_ROOT=/ibex/scratch/alhumosm/SoccerNet/caption-2024
FEATURE_ROOT=/ibex/scratch/alhumosm/SoccerNet/soccernet_videos
LOGS=/ibex/scratch/alhumosm/SoccerNet/logs

echo "=== [1/3] Oracle(Order) Qwen — moment retrieval test set ==="
python baseline_oracle_order_qwen.py \
    --caption_root $CAPTION_ROOT \
    --feature_root $FEATURE_ROOT \
    --eval_split test \
    --window_s 10 \
    --delta_t 5 \
    --top_k 1 5 10 \
    --device cuda \
    --results_file $LOGS/oracle_order_qwen_test_results.json

echo "=== [2/3] Qwen within-game threshold — action spotting test set ==="
python baseline_qwen_spotting_threshold.py \
    --feature_root $FEATURE_ROOT \
    --eval_split test \
    --window_s 10 \
    --device cuda \
    --results_file $LOGS/spotting_qwen_threshold_test_results.json

echo "=== [3/3] Qwen cross-video — action spotting test set ==="
python baseline_qwen_emb_spotting.py \
    --feature_root $FEATURE_ROOT \
    --eval_split test \
    --window_s 10 \
    --index_stride_s 2 \
    --device cuda \
    --results_file $LOGS/spotting_qwen_test_results.json

echo "=== All Qwen test-set baselines done ==="
