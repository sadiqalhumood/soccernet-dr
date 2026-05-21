"""
Oracle (Time + Order) baseline for cross-video moment retrieval.

For each query, returns the exact ground-truth timestamp as the prediction.
This is a perfect cheat — it always knows the right game, half, and moment.

Expected result: Recall@1 = 100% at any delta_t > 0.
If this does not hit 100%, the evaluation pipeline has a bug.

Usage:
    python baseline_oracle_time_order.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
        --delta_t 5 \
        --top_k 1 5 10
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
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--results_file", type=Path,
                   default=Path("/home/alhumosm/soccernet/clip/results/oracle_time_order_baseline_results.json"))
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

    all_games = discover_games(caption_root)
    log.info(f"Found {len(all_games)} games")

    eval_games = all_games if args.max_games is None else all_games[:args.max_games]

    hits = defaultdict(list)

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        by_half = defaultdict(list)
        for ann in annotations:
            by_half[get_half(ann)].append(ann)

        for half, anns in by_half.items():
            for ann in anns:
                gt_ms = int(ann["position"])

                # Predict the exact ground-truth timestamp — perfect cheat
                predictions = [(caption_json, half, gt_ms)]

                for k in args.top_k:
                    hits[k].append(is_hit(predictions, caption_json, half, gt_ms, delta_ms, k))

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"ORACLE (Time + Order) baseline — {len(all_games)} games")
    print(f"Evaluated {total} queries")
    print(f"Delta_t: {args.delta_t}s")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.4f}%")
    print(f"{'='*50}")
    if recall_at_k[1] != 100.0:
        print(f"  WARNING: Recall@1 is not 100% — check the evaluation pipeline!")
    else:
        print(f"  Evaluation pipeline OK.")
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "oracle_time_order",
        "caption_root": str(args.caption_root),
        "delta_t": args.delta_t,
        "games_evaluated": len(eval_games),
        "total_queries": total,
        "recall_at_k": recall_at_k,
        "pipeline_ok": recall_at_k[1] == 100.0,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
