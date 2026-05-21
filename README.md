# SoccerNet VLM Retrieval

Zero-shot and fine-tuned baselines for **soccer video moment retrieval** and **action spotting** using CLIP and Qwen3-VL-Embedding as vision-language backbones.

Both tasks share a single sliding-window inference pipeline: extract per-frame embeddings offline, mean-pool a ±5s window around each frame, then rank by cosine similarity to a text query at inference time.

**PI:** Bernard Ghanem (KAUST)
**Supervisor:** Silvio Giancola (SoccerNet team lead, KAUST)
**Student:** Sadiq Al-Humood (KAUST)

---

## Tasks

| Task | Dataset | Query | Ground Truth | Metric |
|---|---|---|---|---|
| Moment Retrieval | SoccerNet Dense Captioning | Free-form caption | Single timestamp | Recall@K at δt ∈ {1,2,5}s |
| Action Spotting | SoccerNet-v2 | Action class label | All occurrences | mAP at δt ∈ {1,2,5}s |

---

## Results

### Moment Retrieval (δt = 5s)

†98-game test index · ‡471-game full cross-video index

| Method | Backbone | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| Random | — | 0.00% | 0.00% | 0.00% |
| Oracle(Time) | — | 1.38% | 6.76% | 13.20% |
| Zero-shot cross-video† | CLIP | 1.44% | 4.09% | 6.00% |
| Zero-shot cross-video‡ | CLIP | 0.77% | 2.23% | 3.28% |
| Zero-shot cross-video† | Qwen | 0.76% | 2.03% | 3.12% |
| Fine-tuned (InfoNCE, batch 16)† | CLIP | 1.42% | 4.09% | 5.99% |
| Fine-tuned (pairwise MSE, batch 4)‡ | CLIP | 0.41% | 1.25% | 1.75% |
| Oracle(Order) | CLIP | 4.36% | 13.40% | 19.39% |
| Oracle(Order) | Qwen | 3.05% | 9.36% | 14.84% |
| Oracle(Time+Order) | — | 100% | 100% | 100% |

### Action Spotting

| Method | Backbone | mAP@1s | mAP@2s | mAP@5s |
|---|---|---|---|---|
| Oracle(Time) — random floor | — | 0.60% | 1.11% | 2.81% |
| Oracle(Order) — within-game | CLIP | 1.48% | 2.10% | 4.64% |
| Cross-video zero-shot | CLIP | 0.72% | 1.16% | 1.83% |
| Cross-video zero-shot (100 games) | Qwen | 0.75% | 1.21% | 1.77% |

---

## Repository Structure

```
clip/
  extract_clip_features.py          # Extract CLIP ViT-L/14 frame embeddings (2fps)
  extract_features.sh               # SLURM array job for extraction (V100)
  utils.py                          # Shared utilities (game discovery, NMS, evaluation)

  baseline_clip.py                  # Zero-shot CLIP moment retrieval (cross-video)
  baseline_clip_spotting.py         # Zero-shot CLIP action spotting (cross-video)
  baseline_random.py                # Random baseline
  baseline_oracle_time.py           # Oracle(Time): correct game, random ranking
  baseline_oracle_order.py          # Oracle(Order): correct game, CLIP ranking
  baseline_oracle_time_order.py     # Oracle(Time+Order): trivial 100% sanity check
  baseline_oracle_time_spotting.py  # Spotting: random within-game AP
  baseline_oracle_order_spotting.py # Spotting: CLIP within-game AP

  finetune_clip_simple.py           # Pairwise MSE fine-tuning (1 pos + 1 neg, batch 4, V100)
  finetune_clip_best.py             # InfoNCE fine-tuning (in-batch negatives, batch 16, A100)

  run_baseline_clip.sh              # SLURM: run moment retrieval baseline
  run_baselines.sh                  # SLURM: run all moment retrieval baselines
  run_spotting_baselines.sh         # SLURM: run all 3 spotting baselines
  run_finetune_clip_simple.sh       # SLURM: pairwise MSE fine-tuning (V100, auto-resume)
  run_finetune_best.sh              # SLURM: InfoNCE fine-tuning (A100, 1 epoch)
  run_extract_test_ft_features.sh   # SLURM: re-extract test split (pairwise checkpoint)
  run_extract_all_ft_features.sh    # SLURM: re-extract all games (pairwise checkpoint)
  run_extract_best_features.sh      # SLURM: re-extract test split (InfoNCE checkpoint)
  run_eval_best_test.sh             # SLURM: evaluate InfoNCE checkpoint on test split

  analyze_results.py                # Parse results JSON, print per-event breakdown

qwen/
  extract_qwen_features.py          # Extract Qwen3-VL-Embedding-8B frame embeddings
  extract_qwen_features.sh          # SLURM array job (A100 required)
  utils.py                          # Shared utilities

  baseline_qwen_emb.py              # Zero-shot Qwen moment retrieval (cross-video)
  baseline_qwen_emb_spotting.py     # Zero-shot Qwen action spotting (cross-video)
  baseline_oracle_order_qwen.py     # Oracle(Order) diagnostic for Qwen

  run_baseline_qwen_emb.sh          # SLURM: moment retrieval baseline
  run_spotting_qwen.sh              # SLURM: action spotting baseline
  run_oracle_order_qwen.sh          # SLURM: oracle(order) diagnostic
  run_eval_test_qwen.sh             # SLURM: eval on test split

  analyze_results.py                # Parse results JSON
```

---

## Setup

```bash
conda create -n soccernet python=3.10
pip install torch torchvision openai-clip decord scipy numpy tqdm
pip install transformers>=4.57.0 accelerate qwen-vl-utils>=0.0.14
pip install SoccerNet
```

---

## IBEX Paths (KAUST cluster)

| What | Path |
|---|---|
| Videos + features | `/ibex/scratch/alhumosm/SoccerNet/soccernet_videos/` |
| Labels-v2.json | `soccernet_videos/<league>/<season>/<game>/Labels-v2.json` |
| Captions | `/ibex/scratch/alhumosm/SoccerNet/caption-2024/` |
| Checkpoints | `/ibex/scratch/alhumosm/SoccerNet/checkpoints/` |
| Logs | `/ibex/scratch/alhumosm/SoccerNet/logs/` |

---

## Pipeline Overview

```
Input: Video + Text query
  ↓
Extract per-frame embeddings at 2fps  →  [T × D] float16
  ↓
Sliding window: mean-pool frames within ±5s, L2-normalize  →  window embeddings
Encode text query once  →  text embedding
  ↓
Cosine similarity (text, window) for all T windows  →  ranked scores
  ↓
Moment Retrieval: return top-K timestamps
Action Spotting:  use scores as AP confidence values
```

### Feature dimensions

| Backbone | Dim | GPU | Time/half |
|---|---|---|---|
| CLIP ViT-L/14 | 768 | V100 | ~85s |
| Qwen3-VL-Embedding-8B | 4096 | A100 | ~10–15 min |

---

## Dataset Split

```python
from SoccerNet.utils import getListGames
train_games = getListGames("train", task="caption")  # 281 games
valid_games = getListGames("valid", task="caption")  #  92 games
test_games  = getListGames("test",  task="caption")  #  98 games
```

---

## CLIP Fine-Tuning

Two fine-tuning approaches, both training for one epoch on 21,390 anonymized captions from the 281 training games.

### Pairwise MSE (V100, batch 4)

- **Positive:** 5s clip centered on annotated moment, 4 frames, mean-pooled
- **Negative:** random frame from a different game
- **Loss:** `(1 − sim_pos)² + sim_neg²`
- AdamW, LR 1e-6, batch 4, fp16 — fits on a single V100 (16 GB)

```bash
sbatch clip/run_finetune_clip_simple.sh          # train
sbatch clip/run_extract_test_ft_features.sh      # re-extract test split
sbatch clip/run_extract_all_ft_features.sh       # re-extract all games (for spotting)
```

**Result:** R@1 = 0.41% on 471-game index (degraded from zero-shot 0.77% — random negatives insufficient)

### InfoNCE (A100, batch 16)

- **Positive:** same 5s clip construction
- **Negatives:** all other (query, clip) pairs within the batch — 15 negatives per positive
- **Loss:** symmetric cross-entropy over in-batch similarity matrix (CLIP training objective)
- AdamW, LR 1e-5, batch 16, fp16

```bash
sbatch clip/run_finetune_best.sh                 # train (A100, ~20 min)
sbatch clip/run_extract_best_features.sh         # re-extract test split
sbatch clip/run_eval_best_test.sh                # evaluate on test split
```

**Result:** R@1 = 1.42% on 98-game test index (vs zero-shot 1.44% — no meaningful gain)
