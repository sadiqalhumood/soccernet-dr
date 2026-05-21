"""
Pre-extract Qwen3-VL-Embedding-8B frame embeddings from SoccerNet .mkv videos.

Mirrors the exact pipeline of clip/extract_clip_features.py:
  - Same 2fps frame sampling
  - Same per-frame embeddings (no pooling at extraction time)
  - Same .npz output format alongside each .mkv
  - Same SLURM array job sharding

Output per video half:
    1_qwen_features.npz  →  embeddings [T, 4096] float16
                             timestamps_ms [T] int64

Pooling design (same as CLIP):
  Per-frame embeddings are saved. At inference time, baseline_clip.py's
  build_window_embeddings() mean-pools frames within ±window_s/2 seconds
  and L2-normalizes. The same inference code will work with these embeddings
  by swapping the .npz filename.

Model:
  Qwen/Qwen3-VL-Embedding-8B — EOS-token pooling, 4096-dim output.
  Uses flash_attention_2 for efficiency on A100.
  Requires an A100 (40 or 80GB) — too large for V100 (16GB).

Usage:
    python extract_qwen_features.py --dataset_root ~/SoccerNet/caption-2023
    python extract_qwen_features.py --dataset_root ~/SoccerNet/caption-2023 --dry_run

SLURM (submit via extract_qwen_features.sh with ARRAY 0-7):
    python extract_qwen_features.py \\
        --dataset_root /ibex/scratch/alhumosm/SoccerNet/soccernet_videos \\
        --job_id $SLURM_ARRAY_TASK_ID --num_jobs 8
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Qwen3-VL-Embedding frame embeddings from soccer videos."
    )
    p.add_argument("--dataset_root", required=True, type=Path,
                   help="Root directory containing *_224p.mkv files")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Frame sampling rate (default: 2.0, same as CLIP extraction)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Frames per forward pass (default: 16, lower than CLIP due to model size)")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL,
                   help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--job_id", type=int, default=0,
                   help="SLURM array task index (default: 0)")
    p.add_argument("--num_jobs", type=int, default=1,
                   help="Total SLURM array tasks (default: 1)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print pending videos without running")
    p.add_argument("--video_list", type=Path, default=None,
                   help="Text file with explicit video paths to process (one per line). "
                        "Bypasses sharding — use for resubmitting failed tasks.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# File helpers (mirrors clip/extract_clip_features.py)
# ---------------------------------------------------------------------------

def discover_videos(dataset_root: Path) -> list[Path]:
    """Find all *_224p.mkv files, sorted for deterministic sharding."""
    return sorted(dataset_root.rglob("*_224p.mkv"))


def get_output_path(video_path: Path) -> Path:
    """1_224p.mkv → 1_qwen_features.npz in the same directory."""
    stem = video_path.name.replace("_224p.mkv", "_qwen_features")
    return video_path.parent / f"{stem}.npz"


def compute_frame_indices(video_path: Path, target_fps: float):
    """Return (frame_indices, timestamps_ms) for uniform sampling at target_fps."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)

    stride = max(1, round(native_fps / target_fps))
    frame_indices = list(range(0, total_frames, stride))
    timestamps_ms = [round(idx / native_fps * 1000) for idx in frame_indices]

    return frame_indices, timestamps_ms


def save_features(output_path: Path, embeddings: np.ndarray, timestamps_ms: list[int]):
    """Save embeddings and timestamps as compressed .npz."""
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float16),
        timestamps_ms=np.array(timestamps_ms, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    """
    Load Qwen3-VL-Embedding model and processor.

    Uses flash_attention_2 for memory efficiency on A100.
    Left-padding is required so that the EOS token is always at position -1.
    """
    log.info(f"Loading {model_name} on {device} ...")
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
        padding_side="left",   # required: EOS token must be at the last position
        trust_remote_code=True,
    )
    return model, processor


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_frames(
    model,
    processor,
    frames: list[Image.Image],
    device: str,
) -> np.ndarray:
    """
    Embed a batch of PIL frames with Qwen3-VL-Embedding.

    Each frame is treated as a standalone visual document (no text instruction).
    EOS-token pooling extracts the embedding: because the tokenizer uses left-
    padding, the EOS token always sits at the final sequence position [:, -1, :].

    Returns L2-normalized float16 array of shape [B, 4096].
    """
    # Build one single-image message per frame
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": frame}]}]
        for frame in frames
    ]
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
    ]

    inputs = processor(
        text=texts,
        images=frames,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # EOS pooling: last token of the last hidden layer (left-padding ensures EOS is at -1)
    embeddings = outputs.hidden_states[-1][:, -1, :]   # [B, 4096]
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)

    return embeddings.cpu().half().numpy()             # [B, 4096] float16


# ---------------------------------------------------------------------------
# Per-video extraction
# ---------------------------------------------------------------------------

def extract_features(
    video_path: Path,
    model,
    processor,
    frame_indices: list[int],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Decode frames and encode with Qwen3-VL-Embedding.
    Returns L2-normalized float16 embeddings of shape [T, 4096].
    """
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=4)
    all_embeddings = []

    for start in tqdm(range(0, len(frame_indices), batch_size),
                      desc=f"  {video_path.name}", leave=False):
        batch_indices = frame_indices[start : start + batch_size]
        frames_np = vr.get_batch(batch_indices).asnumpy()   # [B, H, W, C] uint8
        frames = [Image.fromarray(f) for f in frames_np]

        batch_emb = embed_frames(model, processor, frames, device)
        all_embeddings.append(batch_emb)

    return np.concatenate(all_embeddings, axis=0)   # [T, 4096]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()   # never .resolve() on IBEX

    if not dataset_root.exists():
        log.error(f"dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    if args.video_list:
        # Resubmit mode: read explicit list of videos, shard across array tasks
        with open(args.video_list) as f:
            videos = [Path(line.strip()) for line in f if line.strip()]
        log.info(f"Video list mode: {len(videos)} videos from {args.video_list}")
        videos = videos[args.job_id :: args.num_jobs]
        log.info(f"Job {args.job_id}/{args.num_jobs}: {len(videos)} videos this task")
    else:
        videos = discover_videos(dataset_root)
        if not videos:
            log.error(f"No *_224p.mkv files found under {dataset_root}")
            sys.exit(1)
        # Shard across SLURM array tasks
        videos = videos[args.job_id :: args.num_jobs]
        log.info(f"Job {args.job_id}/{args.num_jobs}: {len(videos)} videos total")

    if args.dry_run:
        for v in videos:
            out = get_output_path(v)
            status = "EXISTS" if out.exists() else "PENDING"
            print(f"[{status}] {v}")
        return

    # Filter already-extracted videos BEFORE loading the model
    # (avoids wasting GPU time loading an 8B model just to skip everything)
    videos = [v for v in videos if not get_output_path(v).exists()]
    if not videos:
        log.info("All videos already processed, nothing to do.")
        return

    log.info(f"Processing {len(videos)} videos (skipping already-done)")

    t_load_start = time.time()
    model, processor = load_model(args.model_name, args.device)
    log.info(f"Model loaded in {time.time() - t_load_start:.1f}s")

    failed = []
    job_start = time.time()

    for video_path in tqdm(videos, desc="Videos"):
        output_path = get_output_path(video_path)

        t0 = time.time()
        try:
            frame_indices, timestamps_ms = compute_frame_indices(video_path, args.fps)
            embeddings = extract_features(
                video_path, model, processor, frame_indices, args.batch_size, args.device
            )
            save_features(output_path, embeddings, timestamps_ms)
            elapsed = time.time() - t0
            fps_processed = len(frame_indices) / elapsed
            log.info(
                f"DONE: {output_path.name} | shape={embeddings.shape} "
                f"| {elapsed:.1f}s ({fps_processed:.1f} frames/s)"
            )

        except Exception as e:
            log.error(f"FAILED: {video_path} | {time.time() - t0:.1f}s | {e}")
            failed.append(str(video_path))

    total_elapsed = time.time() - job_start
    log.info(f"All done in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if failed:
        log.warning(f"{len(failed)} video(s) failed:")
        for f in failed:
            log.warning(f"  {f}")


if __name__ == "__main__":
    main()
