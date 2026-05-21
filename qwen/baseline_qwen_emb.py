"""
Zero-shot Qwen3-VL-Embedding baseline for cross-video moment retrieval.

Mirrors clip/baseline_clip.py exactly — same sliding window, same global index,
same NMS and evaluation. The only differences are:
  - Loads *_qwen_features.npz (4096-dim) instead of *_clip_features.npz (768-dim)
  - Encodes text queries with Qwen3-VL-Embedding-8B instead of CLIP

Run after qwen/extract_qwen_features.sh has finished.

Usage:
    python baseline_qwen_emb.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --window_s 10 \
        --index_stride_s 2 \
        --delta_t 5 \
        --top_k 1 5 10

Quick test (2 games):
    python baseline_qwen_emb.py \
        --caption_root /ibex/scratch/alhumosm/SoccerNet/caption-2024 \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --max_games 2
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from SoccerNet.utils import getListGames

from utils import discover_games, get_half, load_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
TASK_INSTRUCTION = "Retrieve relevant video clips for the query: "


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", required=True, type=Path,
                   help="Path to caption split, e.g. ~/SoccerNet/caption-2024")
    p.add_argument("--feature_root", required=True, type=Path,
                   help="Root where *_qwen_features.npz files live")
    p.add_argument("--window_s", type=float, default=10.0,
                   help="Sliding window size in seconds (default: 10)")
    p.add_argument("--index_stride_s", type=float, default=2.0,
                   help="Stride between index entries in seconds (default: 2)")
    p.add_argument("--delta_t", type=float, default=5.0,
                   help="Hit tolerance in seconds for Recall@K (default: 5)")
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10],
                   help="K values to report Recall@K (default: 1 5 10)")
    p.add_argument("--qwen_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None,
                   help="Limit evaluation to N games (default: all)")
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"],
                   help="Only evaluate queries from this split (default: all games)")
    p.add_argument("--results_file", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/qwen_emb_baseline_results.json"),
                   help="Where to save results as JSON")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def get_feature_path(caption_json: Path, caption_root: Path, feature_root: Path, half: int) -> Path:
    rel = caption_json.parent.relative_to(caption_root)
    return feature_root / rel / f"{half}_qwen_features.npz"


def load_features(npz_path: Path):
    d = np.load(npz_path)
    embeddings = d["embeddings"].astype(np.float32)    # [T, 4096]
    timestamps_ms = d["timestamps_ms"]                 # [T]
    return embeddings, timestamps_ms


# ---------------------------------------------------------------------------
# Sliding window aggregation (identical to CLIP baseline)
# ---------------------------------------------------------------------------

def build_window_embeddings(embeddings: np.ndarray, timestamps_ms: np.ndarray,
                             window_ms: float) -> np.ndarray:
    """
    For each frame position, mean-pool all frame embeddings within ±window_ms/2.
    Returns L2-normalized window embeddings of shape [T, 4096].
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
# Global index (identical to CLIP baseline)
# ---------------------------------------------------------------------------

def build_global_index(games: list[Path], caption_root: Path, feature_root: Path,
                        window_ms: float, stride_ms: float):
    """
    Load Qwen features for every game/half, build window embeddings, and
    subsample at stride_ms.

    Returns:
        index_embs : np.ndarray [N, 4096] float16
        index_meta : list of (caption_json Path, half int, timestamp_ms int)
    """
    index_embs = []
    index_meta = []
    missing = 0

    for caption_json in tqdm(games, desc="Building index"):
        for half in [1, 2]:
            feature_path = get_feature_path(caption_json, caption_root, feature_root, half)
            if not feature_path.exists():
                missing += 1
                continue

            embeddings, timestamps_ms = load_features(feature_path)
            window_embs = build_window_embeddings(embeddings, timestamps_ms, window_ms)

            # Subsample at stride_ms
            selected = [0]
            for i in range(1, len(timestamps_ms)):
                if timestamps_ms[i] - timestamps_ms[selected[-1]] >= stride_ms:
                    selected.append(i)

            for i in selected:
                index_embs.append(window_embs[i].astype(np.float16))
                index_meta.append((caption_json, half, int(timestamps_ms[i])))

    if missing:
        log.warning(f"{missing} feature files missing — those games excluded from index")

    index_embs = np.stack(index_embs)   # [N, 4096] float16 (already float16)
    log.info(f"Global index: {len(index_meta):,} windows from {len(games)} games")
    return index_embs, index_meta


# ---------------------------------------------------------------------------
# Text encoding with Qwen3-VL-Embedding
# ---------------------------------------------------------------------------

def load_qwen(model_name: str, device: str):
    log.info(f"Loading {model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_name,
        padding_side="left",
        trust_remote_code=True,
    )
    return model, processor


def encode_text(model, processor, text: str, device: str) -> np.ndarray:
    """
    Encode a single text query with Qwen3-VL-Embedding.
    Prepends task instruction as recommended for the query side.
    Returns L2-normalized float32 vector of shape [4096].
    """
    query = TASK_INSTRUCTION + text
    messages = [[{"role": "user", "content": [{"type": "text", "text": query}]}]]
    formatted = processor.apply_chat_template(messages[0], tokenize=False,
                                               add_generation_prompt=False)
    inputs = processor(text=formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, -1, :]          # EOS pooling [1, 4096]
    emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.cpu().numpy()[0]                        # [4096]


# ---------------------------------------------------------------------------
# NMS and hit detection (identical to CLIP baseline)
# ---------------------------------------------------------------------------

def global_nms_topk(index_meta: list, scores: np.ndarray,
                     delta_ms: float, k: int) -> list[tuple]:
    order = np.argsort(scores)[::-1]
    selected = []

    for idx in order:
        c_json, c_half, c_ts = index_meta[idx]
        too_close = any(
            c_json == s_json and c_half == s_half and abs(c_ts - s_ts) < delta_ms
            for s_json, s_half, s_ts in selected
        )
        if not too_close:
            selected.append(index_meta[idx])
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
    window_ms  = args.window_s * 1000
    stride_ms  = args.index_stride_s * 1000
    delta_ms   = args.delta_t * 1000
    max_k      = max(args.top_k)

    log.info(f"Caption root   : {caption_root}")
    log.info(f"Feature root   : {feature_root}")
    log.info(f"Window         : {args.window_s}s")
    log.info(f"Index stride   : {args.index_stride_s}s")
    log.info(f"Delta t        : {args.delta_t}s")
    log.info(f"Top-K          : {args.top_k}")

    all_games = discover_games(caption_root)
    log.info(f"Found {len(all_games)} games")

    # Determine eval games before building index (index uses same split as queries)
    eval_games = all_games
    if args.eval_split:
        split_game_names = set(getListGames(args.eval_split, task="caption"))
        eval_games = [g for g in all_games if str(g.parent.relative_to(caption_root)) in split_game_names]
        log.info(f"Evaluating queries from {args.eval_split} split: {len(eval_games)} games")
    if args.max_games is not None:
        eval_games = eval_games[:args.max_games]

    # Load Qwen text encoder
    model, processor = load_qwen(args.qwen_model, args.device)

    # Phase 1: Build index from the same games as the queries (fair evaluation)
    index_embs, index_meta = build_global_index(
        eval_games, caption_root, feature_root, window_ms, stride_ms
    )
    # Keep as float16 on GPU — 4096-dim index would be ~17GB as float32,
    # float16 halves that to ~8.5GB, leaving room for the 8B model (~16GB)
    index_tensor = torch.from_numpy(index_embs).to(args.device)   # [N, 4096] float16

    # Phase 2: Evaluate queries
    log.info(f"Evaluating queries from {len(eval_games)} games")

    hits      = defaultdict(list)
    per_query = []
    skipped   = 0

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
                text  = ann["description"]

                text_emb    = encode_text(model, processor, text, args.device)   # [4096]
                text_tensor = torch.from_numpy(text_emb).to(args.device)

                # Cast index to float32 for matmul precision
                scores = (index_tensor.float() @ text_tensor).cpu().numpy()      # [N]

                predictions = global_nms_topk(index_meta, scores, delta_ms, max_k)

                query_hits = {}
                for k in args.top_k:
                    h = is_hit(predictions, caption_json, half, gt_ms, delta_ms, k)
                    hits[k].append(h)
                    query_hits[f"hit@{k}"] = h

                per_query.append({
                    "game":        caption_json.parent.name,
                    "half":        half,
                    "gt_ms":       gt_ms,
                    "description": text,
                    **query_hits,
                })

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*50}")
    print(f"CROSS-VIDEO retrieval — Qwen3-VL-Embedding")
    print(f"{len(eval_games)} games in index")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Window: {args.window_s}s  |  Stride: {args.index_stride_s}s  |  delta_t: {args.delta_t}s")
    print(f"{'='*50}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.2f}%")
    print(f"{'='*50}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.qwen_model,
        "caption_root": str(args.caption_root),
        "feature_root": str(args.feature_root),
        "window_s": args.window_s,
        "index_stride_s": args.index_stride_s,
        "delta_t": args.delta_t,
        "eval_split": args.eval_split or "all",
        "games_in_index": len(eval_games),
        "total_queries": total,
        "skipped_queries": skipped,
        "recall_at_k": recall_at_k,
        "per_query":   per_query,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
