"""
Zero-shot Qwen3-VL-Embedding baseline for action spotting.

Mirrors clip/baseline_clip_spotting.py exactly — same global index, same AP
computation. The only differences are:
  - Loads *_qwen_features.npz (4096-dim) instead of *_clip_features.npz (768-dim)
  - Encodes the 17 action class labels with Qwen3-VL-Embedding-8B

Run after qwen/extract_qwen_features.sh has finished (all 1100 halves done).

Usage:
    python baseline_qwen_emb_spotting.py \
        --feature_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \
        --window_s 10 --index_stride_s 2 --device cuda
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

from utils import discover_spotting_games, load_spotting_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
TASK_INSTRUCTION = "Retrieve relevant video clips for the query: "
DELTA_T_LIST = [1.0, 2.0, 5.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_root",   required=True, type=Path)
    p.add_argument("--window_s",       type=float, default=10.0)
    p.add_argument("--index_stride_s", type=float, default=2.0)
    p.add_argument("--qwen_model",     type=str, default=DEFAULT_MODEL)
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--max_games",      type=int, default=None)
    p.add_argument("--eval_split", type=str, default=None, choices=["train", "valid", "test"])
    p.add_argument("--results_file",   type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/logs/spotting_qwen_results.json"))
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


def build_global_index(all_games, window_ms, stride_ms):
    """
    Returns:
        index_embs : float16 [N, 4096]
        index_meta : list of (labels_json, half, timestamp_ms)
    """
    index_embs = []
    index_meta = []
    missing = 0

    for labels_json in tqdm(all_games, desc="Building index"):
        for half in [1, 2]:
            fp = labels_json.parent / f"{half}_qwen_features.npz"
            if not fp.exists():
                missing += 1
                continue
            embeddings, timestamps_ms = load_features(fp)
            window_embs = build_window_embeddings(embeddings, timestamps_ms, window_ms)

            selected = [0]
            for i in range(1, len(timestamps_ms)):
                if timestamps_ms[i] - timestamps_ms[selected[-1]] >= stride_ms:
                    selected.append(i)

            for i in selected:
                index_embs.append(window_embs[i].astype(np.float16))
                index_meta.append((labels_json, half, int(timestamps_ms[i])))

    if missing:
        log.warning(f"{missing} feature files missing")
    log.info(f"Global index: {len(index_meta):,} windows from {len(all_games)} games")
    return np.stack(index_embs), index_meta


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


def encode_label(model, processor, label: str, device: str) -> np.ndarray:
    """Encode an action class label with Qwen3-VL-Embedding. Returns [4096] float32."""
    query = TASK_INSTRUCTION + label
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]
    formatted = processor.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
    inputs = processor(text=formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    emb = outputs.hidden_states[-1][:, -1, :]          # EOS pooling [1, 4096]
    emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.cpu().numpy()[0]                        # [4096]


def compute_global_ap(sorted_meta, gts_by_game, delta_ms):
    """
    sorted_meta : list of (labels_json, half, time_ms) sorted by score desc
    gts_by_game : dict labels_json → [(half, time_ms), ...]
    Returns AP in [0, 1].
    """
    total_gts = sum(len(v) for v in gts_by_game.values())
    if total_gts == 0:
        return 0.0

    game_matched = {g: [False] * len(v) for g, v in gts_by_game.items()}
    n_tp = n_fp = 0
    ap = prev_recall = 0.0
    found = 0

    for game, half, time_ms in sorted_meta:
        is_tp = False
        if game in game_matched:
            gts = gts_by_game[game]
            for i, (gt_half, gt_ms) in enumerate(gts):
                if (not game_matched[game][i]
                        and gt_half == half
                        and abs(time_ms - gt_ms) <= delta_ms):
                    game_matched[game][i] = True
                    is_tp = True
                    found += 1
                    break

        n_tp += is_tp
        n_fp += not is_tp
        precision = n_tp / (n_tp + n_fp)
        recall    = n_tp / total_gts
        if recall > prev_recall:
            ap += precision * (recall - prev_recall)
            prev_recall = recall
        if found == total_gts:
            break

    return ap


def main():
    args = parse_args()
    feature_root = args.feature_root.expanduser()
    window_ms = args.window_s * 1000
    stride_ms = args.index_stride_s * 1000

    model, processor = load_qwen(args.qwen_model, args.device)

    all_games = discover_spotting_games(feature_root)
    if args.eval_split:
        from SoccerNet.utils import getListGames
        split_games = set(getListGames(args.eval_split, task="spotting"))
        all_games = [g for g in all_games if str(g.parent.relative_to(feature_root)) in split_games]
        log.info(f"Filtered to {args.eval_split} split: {len(all_games)} games")
    eval_games = all_games if args.max_games is None else all_games[:args.max_games]

    # Build global index from ALL games (not just eval — same as CLIP spotting)
    index_embs, index_meta = build_global_index(all_games, window_ms, stride_ms)
    # Keep float16 on GPU to fit alongside the 8B model (~16GB model + ~20GB index on A100)
    index_tensor = torch.from_numpy(index_embs).to(args.device)   # [N, 4096] float16

    log.info("Collecting ground truths ...")
    gts_by_label = defaultdict(lambda: defaultdict(list))
    for labels_json in eval_games:
        queries = load_spotting_annotations(labels_json)
        for label, instances in queries.items():
            gts_by_label[label][labels_json].extend(instances)

    all_labels = sorted(gts_by_label.keys())
    log.info(f"Found {len(all_labels)} action classes across {len(eval_games)} games")

    map_per_dt   = {dt: [] for dt in DELTA_T_LIST}
    per_class_ap = {dt: {} for dt in DELTA_T_LIST}

    for label in tqdm(all_labels, desc="Labels"):
        emb = encode_label(model, processor, label, args.device)
        label_tensor = torch.from_numpy(emb).to(args.device)

        scores = (index_tensor.float() @ label_tensor).cpu().numpy()   # [N]

        order = np.argsort(scores)[::-1]
        sorted_meta = [index_meta[i] for i in order]

        gts = gts_by_label[label]
        total_gts = sum(len(v) for v in gts.values())
        log.info(f"  {label}: {total_gts} instances across {len(gts)} games")

        for dt in DELTA_T_LIST:
            ap = compute_global_ap(sorted_meta, gts, dt * 1000)
            per_class_ap[dt][label] = float(ap)
            map_per_dt[dt].append(ap)

    print(f"\n{'='*65}")
    print(f"Zero-shot Qwen — Action Spotting  |  {len(eval_games)} games")
    print(f"Global index: {len(index_meta):,} windows")
    print(f"{'='*65}")
    print(f"{'Label':<25}  " + "  ".join(f"AP@{dt:.0f}s" for dt in DELTA_T_LIST))
    print(f"{'-'*65}")
    for label in all_labels:
        row = f"{label:<25}  "
        row += "  ".join(f"{per_class_ap[dt][label]*100:6.2f}%" for dt in DELTA_T_LIST)
        print(row)
    print(f"{'-'*65}")
    mAPs = {dt: float(np.mean(map_per_dt[dt])) for dt in DELTA_T_LIST}
    print("mAP" + " " * 22 + "  ".join(f"{mAPs[dt]*100:6.2f}%" for dt in DELTA_T_LIST))
    print(f"{'='*65}\n")

    results = {
        "timestamp":            datetime.now().isoformat(),
        "baseline":             "qwen_spotting_crossvideo",
        "model":                args.qwen_model,
        "feature_root":         str(args.feature_root),
        "window_s":             args.window_s,
        "index_stride_s":       args.index_stride_s,
        "delta_t_list":         DELTA_T_LIST,
        "games_in_index":       len(all_games),
        "games_evaluated":      len(eval_games),
        "total_index_windows":  len(index_meta),
        "mAP_per_delta_t":      {str(dt): mAPs[dt] for dt in DELTA_T_LIST},
        "per_class_AP":         {str(dt): per_class_ap[dt] for dt in DELTA_T_LIST},
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {args.results_file}")


if __name__ == "__main__":
    main()
