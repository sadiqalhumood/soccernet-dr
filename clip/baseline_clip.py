"""
Zero-shot CLIP baseline for cross-video moment retrieval.

Builds a global index from ALL games, then for each test caption searches
across every game to find the correct game and timestamp.
A hit requires both the correct game/half AND being within delta_t of ground truth.

Usage:
    python baseline_clip.py \
        --caption_root ~/SoccerNet/caption-2024 \
        --feature_root ~/SoccerNet \
        --window_s 10 \
        --index_stride_s 2 \
        --delta_t 5 \
        --top_k 1 5 10 \
        --clip_model ViT-L/14
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import clip
import numpy as np
import torch
from SoccerNet.utils import getListGames
from tqdm import tqdm

from utils import discover_games, get_half, load_annotations, smooth_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", required=True, type=Path,
                   help="Path to caption split, e.g. ~/SoccerNet/caption-2024")
    p.add_argument("--feature_root", required=True, type=Path,
                   help="Root where *_clip_features.npz files live")
    p.add_argument("--window_s", type=float, default=10.0,
                   help="Sliding window size in seconds (default: 10)")
    p.add_argument("--index_stride_s", type=float, default=2.0,
                   help="Stride between index entries in seconds (default: 2)")
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="Hit tolerance in seconds for Recall@K (default: 5)")
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10],
                   help="K values to report Recall@K (default: 1 5 10)")
    p.add_argument("--sigma_s", type=float, default=2.0,
                   help="Gaussian smoothing sigma in seconds (default: 2)")
    p.add_argument("--clip_model", type=str, default="ViT-L/14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None,
                   help="Limit evaluation to N games (default: all)")
    p.add_argument("--results_file", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/clip_baseline_results.json"),
                   help="Where to save results as JSON")
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"],
                   help="Only evaluate queries from this split (default: all games)")
    p.add_argument("--feature_suffix", type=str, default="",
                   help="Suffix for feature files, e.g. '_ft' loads 1_clip_ft_features.npz")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def get_feature_path(caption_json: Path, caption_root: Path, feature_root: Path,
                     half: int, suffix: str = "") -> Path:
    rel = caption_json.parent.relative_to(caption_root)
    return feature_root / rel / f"{half}_clip{suffix}_features.npz"


def load_features(npz_path: Path):
    d = np.load(npz_path)
    embeddings = d["embeddings"].astype(np.float32)    # [T, 768]
    timestamps_ms = d["timestamps_ms"]                 # [T]
    return embeddings, timestamps_ms


# ---------------------------------------------------------------------------
# Sliding window aggregation
# ---------------------------------------------------------------------------

def build_window_embeddings(embeddings: np.ndarray, timestamps_ms: np.ndarray,
                             window_ms: float) -> np.ndarray:
    """
    For each frame position, mean-pool all frame embeddings within ±window_ms/2.
    Returns L2-normalized window embeddings of shape [T, 768].
    """
    half_window = window_ms / 2.0
    T = len(timestamps_ms)
    window_embs = np.empty_like(embeddings)

    for i in range(T):
        t = timestamps_ms[i]
        lo = np.searchsorted(timestamps_ms, t - half_window)
        hi = np.searchsorted(timestamps_ms, t + half_window, side="right")
        window_embs[i] = embeddings[lo:hi].mean(axis=0)

    norms = np.linalg.norm(window_embs, axis=-1, keepdims=True)
    window_embs /= np.clip(norms, 1e-8, None)
    return window_embs


# ---------------------------------------------------------------------------
# Global index
# ---------------------------------------------------------------------------

def build_global_index(games: list[Path], caption_root: Path, feature_root: Path,
                        window_ms: float, stride_ms: float, suffix: str = ""):
    """
    Load features for every game/half, build window embeddings, and subsample
    at stride_ms to keep the index size manageable.

    Returns:
        index_embs : np.ndarray [N, 768] float16
        index_meta : list of (caption_json Path, half int, timestamp_ms int), length N
    """
    index_embs = []
    index_meta = []

    for caption_json in tqdm(games, desc="Building index"):
        for half in [1, 2]:
            feature_path = get_feature_path(caption_json, caption_root, feature_root, half, suffix)
            if not feature_path.exists():
                continue

            embeddings, timestamps_ms = load_features(feature_path)
            window_embs = build_window_embeddings(embeddings, timestamps_ms, window_ms)

            # Subsample at stride_ms
            selected = [0]
            for i in range(1, len(timestamps_ms)):
                if timestamps_ms[i] - timestamps_ms[selected[-1]] >= stride_ms:
                    selected.append(i)

            for i in selected:
                index_embs.append(window_embs[i])
                index_meta.append((caption_json, half, int(timestamps_ms[i])))

    index_embs = np.stack(index_embs).astype(np.float16)   # [N, 768] float16
    log.info(f"Global index: {len(index_meta):,} windows from {len(games)} games")
    return index_embs, index_meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def encode_text(model, text: str, device: str) -> np.ndarray:
    tokens = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()[0]    # [768]


def global_nms_topk(index_meta: list, scores: np.ndarray,
                     delta_ms: float, k: int) -> list[tuple]:
    """
    Greedy NMS over the global index. Two entries suppress each other only if
    they are from the same game+half AND within delta_ms of each other.
    Returns up to k (caption_json, half, timestamp_ms) tuples.
    """
    order = np.argsort(scores)[::-1]
    selected = []   # list of (caption_json, half, ts_ms)

    for idx in order:
        candidate = index_meta[idx]
        c_json, c_half, c_ts = candidate

        too_close = any(
            c_json == s_json and c_half == s_half and abs(c_ts - s_ts) < delta_ms
            for s_json, s_half, s_ts in selected
        )
        if not too_close:
            selected.append(candidate)
        if len(selected) >= k:
            break

    return selected


def is_hit(predictions: list[tuple], gt_json: Path, gt_half: int,
           gt_ms: int, delta_ms: float, k: int) -> bool:
    """
    True if any of the top-k predictions matches the correct game+half
    AND is within delta_ms of the ground truth timestamp.
    """
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
    feature_root = args.feature_root.expanduser()
    window_ms = args.window_s * 1000
    stride_ms = args.index_stride_s * 1000
    delta_ms = args.delta_t * 1000
    max_k = max(args.top_k)
    sigma_positions = args.sigma_s / args.index_stride_s   # sigma in index positions

    log.info(f"Caption root   : {caption_root}")
    log.info(f"Feature root   : {feature_root}")
    log.info(f"Window         : {args.window_s}s")
    log.info(f"Index stride   : {args.index_stride_s}s")
    log.info(f"Delta t        : {args.delta_t}s")
    log.info(f"Top-K          : {args.top_k}")

    # Discover all games (used for both index and queries)
    all_games = discover_games(caption_root)
    log.info(f"Found {len(all_games)} games total")

    # Determine which games to evaluate queries from
    if args.eval_split:
        split_game_names = set(getListGames(args.eval_split, task="caption"))
        eval_games = [g for g in all_games if str(g.parent.relative_to(caption_root)) in split_game_names]
        log.info(f"Evaluating queries from {args.eval_split} split: {len(eval_games)} games")
    else:
        eval_games = all_games

    # Load CLIP text encoder
    log.info(f"Loading CLIP model: {args.clip_model}")
    model, _ = clip.load(args.clip_model, device=args.device)
    model.eval()

    if args.max_games is not None:
        eval_games = eval_games[:args.max_games]

    # Phase 1: Build index from the same games as the queries (fair evaluation)
    index_embs, index_meta = build_global_index(
        eval_games, caption_root, feature_root, window_ms, stride_ms, args.feature_suffix
    )
    # Move to GPU as float32 for fast matmul
    index_tensor = torch.from_numpy(index_embs.astype(np.float32)).to(args.device)  # [N, 768]

    # Phase 2: Evaluate queries
    log.info(f"Evaluating queries from {len(eval_games)} games")

    hits = defaultdict(list)
    per_query = []
    skipped = 0

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        by_half = defaultdict(list)
        for ann in annotations:
            by_half[get_half(ann)].append(ann)

        for half, anns in by_half.items():
            # Skip if this game's features don't exist (can't evaluate queries from it)
            feature_path = get_feature_path(caption_json, caption_root, feature_root, half, args.feature_suffix)
            if not feature_path.exists():
                skipped += len(anns)
                continue

            for ann in anns:
                gt_ms = int(ann["position"])
                text = ann["description"]

                text_emb = encode_text(model, text, args.device)   # [768]

                # Score ALL windows in the global index
                text_tensor = torch.from_numpy(text_emb).to(args.device)
                scores = (index_tensor @ text_tensor).cpu().numpy()  # [N]

                # NOTE: Gaussian smoothing is not applied here because the index mixes
                # windows from different games — smoothing across game boundaries is meaningless.
                # Smoothing is still valid within a single game; skipped for simplicity.

                predictions = global_nms_topk(index_meta, scores, delta_ms, max_k)

                query_hits = {}
                for k in args.top_k:
                    h = is_hit(predictions, caption_json, half, gt_ms, delta_ms, k)
                    hits[k].append(h)
                    query_hits[f"hit@{k}"] = h

                per_query.append({
                    "game": str(caption_json.parent.name),
                    "half": half,
                    "gt_ms": gt_ms,
                    "description": text,
                    **query_hits,
                })

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"CROSS-VIDEO retrieval — {len(eval_games)} games in index")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Window: {args.window_s}s  |  Stride: {args.index_stride_s}s  |  delta_t: {args.delta_t}s")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.2f}%")
    print(f"{'='*50}\n")

    # Save results to JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.clip_model,
        "caption_root": str(args.caption_root),
        "feature_root": str(args.feature_root),
        "window_s": args.window_s,
        "index_stride_s": args.index_stride_s,
        "delta_t": args.delta_t,
        "games_in_index": len(eval_games),
        "total_queries": total,
        "skipped_queries": skipped,
        "recall_at_k": recall_at_k,
        "per_query": per_query,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
