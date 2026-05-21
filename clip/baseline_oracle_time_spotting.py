"""
Oracle (Time) baseline for action spotting — random within-game AP.

Given the correct game, assigns RANDOM scores to all windows and computes
within-game Average Precision for each (game, label) pair, then averages.

Evaluation: within-game AP at δt ∈ {1, 2, 5}s
  - For each (game, label): sort windows randomly, compute AP vs. all GTs
  - Report mean AP per class, and mAP across all 17 classes

This is the floor: what AP do you get with zero signal?
Expected: AP ≈ (#GTs in game) / (#windows in game) per label.

Usage:
    python baseline_oracle_time_spotting.py \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import discover_spotting_games, load_spotting_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DELTA_T_LIST = [1.0, 2.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/spotting_oracle_time_results.json"))
    return p.parse_args()


def load_timestamps(labels_json: Path) -> list:
    """Return all (half, timestamp_ms) pairs from pre-extracted feature files."""
    pairs = []
    for half in [1, 2]:
        fp = labels_json.parent / f"{half}_clip_features.npz"
        if fp.exists():
            ts = np.load(fp)["timestamps_ms"]
            for t in ts:
                pairs.append((half, int(t)))
    return pairs


def compute_within_game_ap(all_pairs, scores, gt_list, delta_ms):
    """
    all_pairs: list of (half, time_ms) — all windows in the game
    scores   : float array [N] — higher = more confident
    gt_list  : list of (half, time_ms) — ground truths for this label in this game
    Returns AP in [0, 1].
    """
    total_gts = len(gt_list)
    if total_gts == 0 or len(all_pairs) == 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    matched = [False] * total_gts
    n_tp = n_fp = 0
    ap = prev_recall = 0.0

    for idx in order:
        half, time_ms = all_pairs[idx]
        is_tp = False
        for i, (gt_half, gt_ms) in enumerate(gt_list):
            if not matched[i] and gt_half == half and abs(time_ms - gt_ms) <= delta_ms:
                matched[i] = True
                is_tp = True
                break

        n_tp += is_tp
        n_fp += not is_tp
        precision = n_tp / (n_tp + n_fp)
        recall    = n_tp / total_gts
        if recall > prev_recall:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
        if n_tp == total_gts:
            break

    return ap


def main():
    args = parse_args()
    feature_root = args.feature_root.expanduser()
    rng = np.random.default_rng(args.seed)

    all_games = discover_spotting_games(feature_root)
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="spotting"))
        all_games = [g for g in all_games if str(g.parent.relative_to(feature_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(all_games)} games")
    eval_games = all_games if args.max_games is None else all_games[:args.max_games]
    log.info(f"Evaluating {len(eval_games)} games")

    # ap_per_label[label][delta_t] = list of per-(game,label) APs
    ap_records = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for labels_json in tqdm(eval_games, desc="Games"):
        queries = load_spotting_annotations(labels_json)
        all_pairs = load_timestamps(labels_json)
        if not all_pairs:
            skipped += sum(len(v) for v in queries.values())
            continue

        scores = rng.random(len(all_pairs)).astype(np.float32)

        for label, gt_list in queries.items():
            for dt in DELTA_T_LIST:
                ap = compute_within_game_ap(all_pairs, scores, gt_list, dt * 1000)
                ap_records[label][dt].append(ap)

    # Aggregate
    all_labels = sorted(ap_records.keys())
    map_per_dt = {}
    per_class   = {}
    for dt in DELTA_T_LIST:
        class_aps = {}
        for label in all_labels:
            vals = ap_records[label][dt]
            class_aps[label] = float(np.mean(vals)) if vals else 0.0
        map_per_dt[dt] = float(np.mean(list(class_aps.values())))
        per_class[dt]  = class_aps

    print(f"\n{'='*60}")
    print(f"Oracle (Time) — Action Spotting  |  {len(eval_games)} games")
    print(f"{'='*60}")
    print(f"{'Label':<25}  " + "  ".join(f"AP@{dt:.0f}s" for dt in DELTA_T_LIST))
    print(f"{'-'*60}")
    for label in all_labels:
        row = f"{label:<25}  "
        row += "  ".join(f"{per_class[dt][label]*100:6.2f}%" for dt in DELTA_T_LIST)
        print(row)
    print(f"{'-'*60}")
    print("mAP" + " " * 22 + "  ".join(f"{map_per_dt[dt]*100:6.2f}%" for dt in DELTA_T_LIST))
    print(f"{'='*60}\n")

    results = {
        "timestamp":       datetime.now().isoformat(),
        "baseline":        "oracle_time_spotting",
        "feature_root":    str(args.feature_root),
        "delta_t_list":    DELTA_T_LIST,
        "games_evaluated": len(eval_games),
        "skipped":         skipped,
        "mAP_per_delta_t": {str(dt): map_per_dt[dt] for dt in DELTA_T_LIST},
        "per_class_AP":    {str(dt): per_class[dt]  for dt in DELTA_T_LIST},
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {args.results_file}")


if __name__ == "__main__":
    main()
