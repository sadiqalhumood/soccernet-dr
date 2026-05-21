"""
Oracle (Order) baseline for action spotting — CLIP within-game AP.

Given the correct game (oracle), uses CLIP cosine similarity to rank all windows
within that game, then computes within-game Average Precision.

Evaluation: within-game AP at δt ∈ {1, 2, 5}s
  - For each (game, label): CLIP scores all windows, compute AP vs. all GTs
  - Report mean AP per class, and mAP across all 17 classes

This tests: "given the correct game, how well does zero-shot CLIP
discriminate action instances from background windows?"

Usage:
    python baseline_oracle_order_spotting.py \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --device cuda
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
from tqdm import tqdm

from utils import discover_spotting_games, load_spotting_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DELTA_T_LIST = [1.0, 2.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_root", required=True, type=Path)
    p.add_argument("--clip_model",   type=str, default="ViT-L/14")
    p.add_argument("--window_s",     type=float, default=10.0)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--max_games",    type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/spotting_oracle_order_results.json"))
    return p.parse_args()


def load_features(fp: Path):
    d = np.load(fp)
    return d["embeddings"].astype(np.float32), d["timestamps_ms"]


def build_window_embeddings(embeddings, timestamps_ms, window_ms):
    half_w = window_ms / 2.0
    window_embs = np.empty_like(embeddings)
    for i in range(len(timestamps_ms)):
        t = timestamps_ms[i]
        lo = np.searchsorted(timestamps_ms, t - half_w)
        hi = np.searchsorted(timestamps_ms, t + half_w, side="right")
        window_embs[i] = embeddings[lo:hi].mean(axis=0)
    norms = np.linalg.norm(window_embs, axis=-1, keepdims=True)
    window_embs /= np.clip(norms, 1e-8, None)
    return window_embs


def encode_label(model, label, device):
    tokens = clip.tokenize([label], truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]


def compute_within_game_ap(all_pairs, scores, gt_list, delta_ms):
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
    window_ms    = args.window_s * 1000

    log.info(f"Loading CLIP {args.clip_model} ...")
    model, _ = clip.load(args.clip_model, device=args.device)
    model.eval()

    label_emb_cache = {}

    all_games = discover_spotting_games(feature_root)
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="spotting"))
        all_games = [g for g in all_games if str(g.parent.relative_to(feature_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(all_games)} games")
    eval_games = all_games if args.max_games is None else all_games[:args.max_games]
    log.info(f"Evaluating {len(eval_games)} games")

    ap_records = defaultdict(lambda: defaultdict(list))
    skipped = 0

    for labels_json in tqdm(eval_games, desc="Games"):
        queries = load_spotting_annotations(labels_json)
        if not queries:
            continue

        # Load features for both halves
        all_pairs   = []   # (half, time_ms)
        all_window_embs = []
        for half in [1, 2]:
            fp = labels_json.parent / f"{half}_clip_features.npz"
            if not fp.exists():
                continue
            embs, ts = load_features(fp)
            window_embs = build_window_embeddings(embs, ts, window_ms)
            for i, t in enumerate(ts):
                all_pairs.append((half, int(t)))
                all_window_embs.append(window_embs[i])

        if not all_pairs:
            skipped += sum(len(v) for v in queries.values())
            continue

        window_tensor = torch.from_numpy(
            np.stack(all_window_embs)
        ).to(args.device)   # [N, 768]

        for label, gt_list in queries.items():
            if label not in label_emb_cache:
                label_emb_cache[label] = encode_label(model, label, args.device)
            label_tensor = torch.from_numpy(label_emb_cache[label]).to(args.device)

            scores = (window_tensor @ label_tensor).cpu().numpy()

            for dt in DELTA_T_LIST:
                ap = compute_within_game_ap(all_pairs, scores, gt_list, dt * 1000)
                ap_records[label][dt].append(ap)

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
    print(f"Oracle (Order) — Action Spotting  |  {len(eval_games)} games  |  CLIP {args.clip_model}")
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
        "baseline":        "oracle_order_spotting",
        "model":           args.clip_model,
        "feature_root":    str(args.feature_root),
        "window_s":        args.window_s,
        "delta_t_list":    DELTA_T_LIST,
        "games_evaluated": len(eval_games),
        "skipped":         skipped,
        "labels_cached":   sorted(label_emb_cache.keys()),
        "mAP_per_delta_t": {str(dt): map_per_dt[dt] for dt in DELTA_T_LIST},
        "per_class_AP":    {str(dt): per_class[dt]  for dt in DELTA_T_LIST},
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {args.results_file}")


if __name__ == "__main__":
    main()
