"""
Oracle (Order) baseline for moment retrieval using Qwen3-VL-Embedding features.

Given the correct game (oracle knowledge), uses Qwen cosine similarity to rank
all windows within that game and returns the top-K.

Key diagnostic purpose: if this gives >0% R@1, the Qwen text encoding works and
the cross-video 0.00% is from poor global ranking noise. If this is also 0%, the
text encoding is broken.

Compare with:
  CLIP Oracle(Order)    — 4.47% R@1 at δt=5s
  Qwen Oracle(Order)    — this script (expected: similar to or higher than CLIP)
  Qwen cross-video      — 0.00% R@1 (previously observed)

Usage:
    python baseline_oracle_order_qwen.py \
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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils import discover_games, get_half, load_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
TASK_INSTRUCTION = "Retrieve relevant video clips for the query: "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", required=True, type=Path)
    p.add_argument("--feature_root", required=True, type=Path)
    p.add_argument("--window_s", type=float, default=10.0)
    p.add_argument("--delta_t", type=float, default=5.0)
    p.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--qwen_model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/qwen_oracle_order_results.json"))
    return p.parse_args()


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
    """Encode a single text query. Returns L2-normalized float32 [4096]."""
    query = TASK_INSTRUCTION + text
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
    inputs = processor(text=formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    emb = outputs.hidden_states[-1][:, -1, :]
    emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.cpu().numpy()[0]


def get_feature_path(caption_json: Path, caption_root: Path, feature_root: Path, half: int) -> Path:
    rel = caption_json.parent.relative_to(caption_root)
    return feature_root / rel / f"{half}_qwen_features.npz"


def load_features(npz_path: Path):
    d = np.load(npz_path)
    return d["embeddings"].astype(np.float32), d["timestamps_ms"]


def build_window_embeddings(embeddings: np.ndarray, timestamps_ms: np.ndarray,
                             window_ms: float) -> np.ndarray:
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


def nms_topk(timestamps_ms: np.ndarray, scores: np.ndarray,
             delta_ms: float, k: int) -> list:
    order = np.argsort(scores)[::-1]
    selected = []
    for idx in order:
        t = int(timestamps_ms[idx])
        if all(abs(t - s) >= delta_ms for s in selected):
            selected.append(t)
        if len(selected) >= k:
            break
    return selected


def is_hit(top_ts: list, gt_ms: int, delta_ms: float, k: int) -> bool:
    return any(abs(t - gt_ms) <= delta_ms for t in top_ts[:k])


def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    feature_root  = args.feature_root.expanduser()
    window_ms = args.window_s * 1000
    delta_ms  = args.delta_t * 1000
    max_k     = max(args.top_k)

    log.info(f"Caption root : {caption_root}")
    log.info(f"Feature root : {feature_root}")
    log.info(f"Window       : {args.window_s}s  |  Delta_t : {args.delta_t}s")

    model, processor = load_qwen(args.qwen_model, args.device)

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
    score_samples = []   # log a few cosine sims for diagnostics

    for caption_json in tqdm(eval_games, desc="Games"):
        annotations = load_annotations(caption_json)
        if not annotations:
            continue

        # Load features for both halves (oracle: we know the game)
        half_features = {}
        for half in [1, 2]:
            fp = get_feature_path(caption_json, caption_root, feature_root, half)
            if fp.exists():
                embs, ts = load_features(fp)
                window_embs = build_window_embeddings(embs, ts, window_ms)
                half_features[half] = (
                    torch.from_numpy(window_embs).to(args.device),
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

                text_emb    = encode_text(model, processor, text, args.device)
                text_tensor = torch.from_numpy(text_emb).to(args.device)

                scores = (window_tensor.float() @ text_tensor).cpu().numpy()

                # Log a few cosine similarity stats for diagnostics
                if len(score_samples) < 20:
                    gt_idx = np.argmin(np.abs(timestamps_ms - gt_ms))
                    score_samples.append({
                        "gt_score":   float(scores[gt_idx]),
                        "max_score":  float(scores.max()),
                        "mean_score": float(scores.mean()),
                        "rank_of_gt": int(np.sum(scores > scores[gt_idx])) + 1,
                    })

                top_ts = nms_topk(timestamps_ms, scores, delta_ms, max_k)
                for k in args.top_k:
                    hits[k].append(is_hit(top_ts, gt_ms, delta_ms, k))

    total = len(hits[args.top_k[0]])
    recall_at_k = {k: round(float(np.mean(hits[k])) * 100, 4) for k in sorted(args.top_k)}

    print(f"\n{'='*55}")
    print(f"ORACLE (Order) — Qwen3-VL-Embedding  |  {len(eval_games)} games")
    print(f"Evaluated {total} queries  |  skipped {skipped}")
    print(f"Window: {args.window_s}s  |  Delta_t: {args.delta_t}s")
    print(f"{'='*55}")
    for k, recall in recall_at_k.items():
        print(f"  Recall@{k:<3} = {recall:.4f}%")
    print(f"{'='*55}")
    print(f"\nDiagnostic — sample cosine similarities (first 20 queries):")
    for s in score_samples[:5]:
        print(f"  gt_score={s['gt_score']:.4f}  max={s['max_score']:.4f}  "
              f"mean={s['mean_score']:.4f}  rank_of_gt={s['rank_of_gt']}")
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "oracle_order_qwen",
        "model": args.qwen_model,
        "caption_root": str(args.caption_root),
        "eval_split": args.eval_split,
        "feature_root": str(args.feature_root),
        "window_s": args.window_s,
        "delta_t": args.delta_t,
        "games_evaluated": len(eval_games),
        "total_queries": total,
        "skipped_queries": skipped,
        "recall_at_k": recall_at_k,
        "score_diagnostics": score_samples,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
