"""
Oracle (Time) baseline for cross-video moment retrieval.

For each query, the model is given the correct game but picks a random ground-truth
timestamp from that game (all halves pooled). It knows WHERE to look but not WHEN.

This represents the performance you'd get if you could always identify the right game
but had to guess randomly among all annotated event times in that game.

A hit requires: correct game + correct half + within delta_t of ground truth.

Usage:
    python baseline_oracle_time.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
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
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="Hit tolerance in seconds (default: 5)")
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/home/alhumosm/soccernet/clip/results/oracle_time_baseline_results.json"))
    return p.parse_args()


def is_hit(predictions: list[tuple], gt_json: Path, gt_half: int,
           gt_ms: int, delta_ms: float, k: int) -> bool:
    for pred_json, pred_half, pred_ms in predictions[:k]:
        if pred_json == gt_json and pred_half == gt_half and abs(pred_ms - gt_ms) <= delta_ms:
            return True
    return False


def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    delta_ms = args.delta_t * 1000
    max_k = max(args.top_k)
    rng = np.random.default_rng(args.seed)

    all_games = discover_games(caption_root)
    log.info(f"Found {len(all_games)} games")
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="caption"))
        all_games = [g for g in all_games if str(g.parent.relative_to(caption_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(all_games)} games")

    eval_games = all_games if args.max_games is None else all_games[:args.max_games]

    hits = defaultdict(list)
    skipped = 0

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        # Collect all gt timestamps across both halves for this game
        game_pool = [(get_half(ann), int(ann["position"])) for ann in annotations]

        by_half = defaultdict(list)
        for ann in annotations:
            by_half[get_half(ann)].append(ann)

        for half, anns in by_half.items():
            for ann in anns:
                gt_ms = int(ann["position"])

                # Sample max_k timestamps from this game's ground-truth pool
                # (correct game, random event time)
                indices = rng.choice(len(game_pool), size=min(max_k, len(game_pool)), replace=False)
                predictions = [(caption_json, game_pool[i][0], game_pool[i][1]) for i in indices]

                for k in args.top_k:
                    hits[k].append(is_hit(predictions, caption_json, half, gt_ms, delta_ms, k))

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"ORACLE (Time) baseline — {len(all_games)} games")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Delta_t: {args.delta_t}s  |  seed: {args.seed}")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.4f}%")
    print(f"{'='*50}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "oracle_time",
        "seed": args.seed,
        "caption_root": str(args.caption_root),
        "eval_split": args.eval_split,
        "delta_t": args.delta_t,
        "games_evaluated": len(eval_games),
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
