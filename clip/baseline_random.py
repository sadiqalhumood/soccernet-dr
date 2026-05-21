"""
Random baseline for cross-video moment retrieval.

For each query, picks K random windows from the global index (all games, all halves).
No model, no features used for ranking — pure chance.
Sets the absolute performance floor.

A hit requires: correct game + correct half + within delta_t of ground truth.

Usage:
    python baseline_random.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --delta_t 5 \
        --top_k 1 5 10 \
        --seed 42
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import discover_games, get_half, load_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", required=True, type=Path)
    p.add_argument("--feature_root", required=True, type=Path,
                   help="Root where *_clip_features.npz files live (used only for timestamps)")
    p.add_argument("--index_stride_s", type=float, default=2.0,
                   help="Stride used when building timestamp pool (default: 2)")
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="Hit tolerance in seconds (default: 5)")
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/home/alhumosm/soccernet/clip/results/random_baseline_results.json"))
    return p.parse_args()


def get_feature_path(caption_json: Path, caption_root: Path, feature_root: Path, half: int) -> Path:
    rel = caption_json.parent.relative_to(caption_root)
    return feature_root / rel / f"{half}_clip_features.npz"


def build_timestamp_pool(games: list[Path], caption_root: Path, feature_root: Path,
                         stride_ms: float) -> list[tuple]:
    """
    Load timestamps (not embeddings) from all npz files.
    Returns list of (caption_json, half, timestamp_ms).
    """
    pool = []
    for caption_json in tqdm(games, desc="Building timestamp pool"):
        for half in [1, 2]:
            feature_path = get_feature_path(caption_json, caption_root, feature_root, half)
            if not feature_path.exists():
                continue
            d = np.load(feature_path)
            timestamps_ms = d["timestamps_ms"]

            # Subsample at stride_ms to match CLIP index density
            selected = [0]
            for i in range(1, len(timestamps_ms)):
                if timestamps_ms[i] - timestamps_ms[selected[-1]] >= stride_ms:
                    selected.append(i)

            for i in selected:
                pool.append((caption_json, half, int(timestamps_ms[i])))

    log.info(f"Timestamp pool: {len(pool):,} windows from {len(games)} games")
    return pool


def is_hit(predictions: list[tuple], gt_json: Path, gt_half: int,
           gt_ms: int, delta_ms: float, k: int) -> bool:
    for pred_json, pred_half, pred_ms in predictions[:k]:
        if pred_json == gt_json and pred_half == gt_half and abs(pred_ms - gt_ms) <= delta_ms:
            return True
    return False


def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    feature_root = args.feature_root.expanduser()
    stride_ms = args.index_stride_s * 1000
    delta_ms = args.delta_t * 1000
    max_k = max(args.top_k)
    rng = np.random.default_rng(args.seed)

    all_games = discover_games(caption_root)
    log.info(f"Found {len(all_games)} games")

    # Pool always uses ALL games (full search space); only eval queries are filtered
    pool = build_timestamp_pool(all_games, caption_root, feature_root, stride_ms)
    pool_size = len(pool)

    eval_games = all_games
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="caption"))
        eval_games = [g for g in all_games if str(g.parent.relative_to(caption_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(eval_games)} eval games")
    if args.max_games is not None:
        eval_games = eval_games[:args.max_games]

    hits = defaultdict(list)
    skipped = 0

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        by_half = defaultdict(list)
        for ann in annotations:
            by_half[get_half(ann)].append(ann)

        for half, anns in by_half.items():
            feature_path = get_feature_path(caption_json, caption_root, feature_root, half)
            if not feature_path.exists():
                skipped += len(anns)
                continue

            for ann in anns:
                gt_ms = int(ann["position"])

                # Sample max_k distinct random windows from the entire global pool
                indices = rng.choice(pool_size, size=max_k, replace=False)
                predictions = [pool[i] for i in indices]

                for k in args.top_k:
                    hits[k].append(is_hit(predictions, caption_json, half, gt_ms, delta_ms, k))

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"RANDOM baseline — {len(all_games)} games ({pool_size:,} windows)")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Delta_t: {args.delta_t}s  |  seed: {args.seed}")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.4f}%")
    print(f"{'='*50}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "random",
        "seed": args.seed,
        "caption_root": str(args.caption_root),
        "feature_root": str(args.feature_root),
        "index_stride_s": args.index_stride_s,
        "delta_t": args.delta_t,
        "games_in_pool": len(all_games),
        "windows_in_pool": pool_size,
        "total_queries": total,
        "skipped_queries": skipped,
        "recall_at_k": recall_at_k,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
