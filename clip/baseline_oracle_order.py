"""
Oracle (Order) baseline for cross-video moment retrieval.

Given the correct game (oracle knowledge), uses CLIP cosine similarity to rank
all windows within that game and returns the top-K.

This tests: "if we could always identify the right game, how well does CLIP
find the right moment within it?" — no random guessing, CLIP does the ranking.

Compare with:
  Oracle(Time)       — knows correct game, picks timestamps RANDOMLY
  Oracle(Order)      — knows correct game, ranks with CLIP          ← this script
  Oracle(Time+Order) — knows exact timestamp (perfect, 100%)

Expected result: ~4-5% R@1, higher than Oracle(Time)'s 1.35%.
This matches the old within-video bug numbers — that baseline was
accidentally measuring exactly this.

Usage:
    python baseline_oracle_order.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --window_s 10 \
        --delta_t 5 \
        --top_k 1 5 10
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

from utils import discover_games, get_half, load_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", required=True, type=Path,
                   help="Path to caption split, e.g. ~/SoccerNet/caption-2024")
    p.add_argument("--feature_root", required=True, type=Path,
                   help="Root where *_clip_features.npz files live")
    p.add_argument("--window_s", type=float, default=10.0,
                   help="Sliding window size in seconds (default: 10)")
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="Hit tolerance in seconds for Recall@K (default: 5)")
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--clip_model", type=str, default="ViT-L/14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/home/alhumosm/soccernet/clip/results/oracle_order_baseline_results.json"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature helpers (same as baseline_clip.py)
# ---------------------------------------------------------------------------

def get_feature_path(caption_json: Path, caption_root: Path, feature_root: Path, half: int) -> Path:
    rel = caption_json.parent.relative_to(caption_root)
    return feature_root / rel / f"{half}_clip_features.npz"


def load_features(npz_path: Path):
    d = np.load(npz_path)
    return d["embeddings"].astype(np.float32), d["timestamps_ms"]


def build_window_embeddings(embeddings: np.ndarray, timestamps_ms: np.ndarray,
                             window_ms: float) -> np.ndarray:
    """Mean-pool frames within ±window_ms/2, L2-normalize. Returns [T, 768]."""
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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def encode_text(model, text: str, device: str) -> np.ndarray:
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]   # [768]


def nms_topk(timestamps_ms: np.ndarray, scores: np.ndarray,
             delta_ms: float, k: int) -> list[int]:
    """Greedy NMS within a single game's windows. Returns up to k timestamps."""
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        t = int(timestamps_ms[idx])
        if all(abs(t - s) >= delta_ms for s in selected):
            selected.append(t)
        if len(selected) >= k:
            break
    return selected


def is_hit(predictions: list[tuple], gt_json: Path, gt_half: int,
           gt_ms: int, delta_ms: float, k: int) -> bool:
    for pred_json, pred_half, pred_ms in predictions[:k]:
        if pred_json == gt_json and pred_half == gt_half and abs(pred_ms - gt_ms) <= delta_ms:
            return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    feature_root  = args.feature_root.expanduser()
    window_ms = args.window_s * 1000
    delta_ms  = args.delta_t * 1000
    max_k     = max(args.top_k)

    log.info(f"Caption root : {caption_root}")
    log.info(f"Feature root : {feature_root}")
    log.info(f"Window       : {args.window_s}s")
    log.info(f"Delta t      : {args.delta_t}s")
    log.info(f"Top-K        : {args.top_k}")

    log.info(f"Loading CLIP model: {args.clip_model}")
    model, _ = clip.load(args.clip_model, device=args.device)
    model.eval()

    all_games = discover_games(caption_root)
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="caption"))
        all_games = [g for g in all_games if str(g.parent.relative_to(caption_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(all_games)} games")
    eval_games = all_games if args.max_games is None else all_games[:args.max_games]
    log.info(f"Evaluating {len(eval_games)} games")

    hits    = defaultdict(list)
    skipped = 0

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        # Pre-load features for both halves of this game (oracle: we know the game)
        half_features = {}
        for half in [1, 2]:
            fp = get_feature_path(caption_json, caption_root, feature_root, half)
            if fp.exists():
                embs, ts = load_features(fp)
                window_embs = build_window_embeddings(embs, ts, window_ms)
                # Move to GPU for fast scoring
                half_features[half] = (
                    torch.from_numpy(window_embs).to(args.device),  # [T, 768]
                    ts,
                )

        by_half = defaultdict(list)
        for ann in annotations:
            by_half[get_half(ann)].append(ann)

        for half, anns in by_half.items():
            if half not in half_features:
                skipped += len(anns)
                continue

            window_tensor, timestamps_ms = half_features[half]

            for ann in anns:
                gt_ms = int(ann["position"])
                text  = ann["description"]

                text_emb    = encode_text(model, text, args.device)          # [768]
                text_tensor = torch.from_numpy(text_emb).to(args.device)

                # Score all windows in the correct game's half
                scores = (window_tensor @ text_tensor).cpu().numpy()         # [T]

                # NMS within this half, return top-K timestamps
                top_ts = nms_topk(timestamps_ms, scores, delta_ms, max_k)
                predictions = [(caption_json, half, ts) for ts in top_ts]

                for k in args.top_k:
                    hits[k].append(is_hit(predictions, caption_json, half,
                                          gt_ms, delta_ms, k))

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"ORACLE (Order) baseline — {len(eval_games)} games")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Window: {args.window_s}s  |  Delta_t: {args.delta_t}s")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.4f}%")
    print(f"{'='*50}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "oracle_order",
        "model": args.clip_model,
        "caption_root": str(args.caption_root),
        "eval_split": args.eval_split,
        "feature_root": str(args.feature_root),
        "window_s": args.window_s,
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
