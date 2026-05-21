"""
Pre-extract CLIP (ViT-L/14) frame embeddings from SoccerNet .mkv videos.

Saves per-half .npz files alongside each video:
    1_clip_features.npz  →  embeddings [T, 768] float16
                             timestamps_ms [T] int64

Usage:
    python extract_clip_features.py --dataset_root ~/SoccerNet/caption-2023
    python extract_clip_features.py --dataset_root ~/SoccerNet/caption-2023 --dry_run
"""

import argparse
import logging
import sys
from pathlib import Path

import clip
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from SoccerNet.utils import getListGames
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Extract CLIP frame embeddings from soccer videos.")
    p.add_argument("--dataset_root", required=True, type=Path,
                   help="Root of a dataset split, e.g. ~/SoccerNet/caption-2023")
    p.add_argument("--fps", type=float, default=2.0,
                   help="Target frame sampling rate (default: 2.0)")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Frames per CLIP forward pass (default: 256)")
    p.add_argument("--clip_model", type=str, default="ViT-L/14",
                   help="CLIP model variant (default: ViT-L/14)")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu (default: cuda)")
    p.add_argument("--job_id", type=int, default=0,
                   help="SLURM array task index (default: 0)")
    p.add_argument("--num_jobs", type=int, default=1,
                   help="Total SLURM array tasks for sharding (default: 1)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print videos that would be processed without running")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Fine-tuned CLIP checkpoint to load (default: use pretrained weights)")
    p.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"],
                   help="Only process videos from this split's games (default: all videos)")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Suffix added to output filename, e.g. '_ft' → 1_clip_ft_features.npz")
    return p.parse_args()


def discover_videos(dataset_root: Path) -> list[Path]:
    """Find all *_224p.mkv files under dataset_root, sorted for deterministic sharding."""
    videos = sorted(dataset_root.rglob("*_224p.mkv"))
    return videos


def get_output_path(video_path: Path, suffix: str = "") -> Path:
    """Replace e.g. 1_224p.mkv → 1_clip_features.npz (or 1_clip_ft_features.npz with suffix='_ft')."""
    stem = video_path.name.replace("_224p.mkv", f"_clip{suffix}_features")
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


def extract_features(
    video_path: Path,
    model,
    preprocess,
    frame_indices: list[int],
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Decode frames and encode with CLIP vision encoder.
    Returns L2-normalized float16 embeddings of shape [T, 768].
    """
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=4)
    all_embeddings = []

    for start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[start : start + batch_size]

        # Decode frames: [B, H, W, C] uint8 RGB
        frames = vr.get_batch(batch_indices).asnumpy()

        # Preprocess each frame individually (CLIP preprocess expects PIL images)
        tensors = [preprocess(Image.fromarray(frame)) for frame in frames]
        batch_tensor = torch.stack(tensors).to(device)  # [B, 3, 224, 224]

        with torch.no_grad():
            features = model.encode_image(batch_tensor)          # [B, 768]
            features = features / features.norm(dim=-1, keepdim=True)  # L2-normalize

        all_embeddings.append(features.cpu().half().numpy())

    return np.concatenate(all_embeddings, axis=0)  # [T, 768]


def save_features(output_path: Path, embeddings: np.ndarray, timestamps_ms: list[int]):
    """Save embeddings and timestamps to a compressed .npz file."""
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float16),
        timestamps_ms=np.array(timestamps_ms, dtype=np.int64),
    )


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()

    if not dataset_root.exists():
        log.error(f"dataset_root does not exist: {dataset_root}")
        sys.exit(1)

    videos = discover_videos(dataset_root)
    if not videos:
        log.error(f"No *_224p.mkv files found under {dataset_root}")
        sys.exit(1)

    # Filter to a specific split's games if requested
    if args.split:
        split_games = set(getListGames(args.split, task="caption"))
        videos = [v for v in videos if any(g in str(v) for g in split_games)]
        log.info(f"Filtered to {args.split} split: {len(videos)} videos")

    # Shard across SLURM array tasks
    videos = videos[args.job_id :: args.num_jobs]
    log.info(f"Job {args.job_id}/{args.num_jobs}: {len(videos)} videos to process")

    if args.dry_run:
        for v in videos:
            out = get_output_path(v, args.output_suffix)
            status = "EXISTS" if out.exists() else "PENDING"
            print(f"[{status}] {v}")
        return

    # Filter out already-processed videos before loading the model
    videos = [v for v in videos if not get_output_path(v, args.output_suffix).exists()]
    if not videos:
        log.info("All videos already processed, nothing to do.")
        return

    # Load CLIP model
    log.info(f"Loading CLIP model: {args.clip_model} on {args.device}")
    model, preprocess = clip.load(args.clip_model, device=args.device)

    if args.checkpoint:
        log.info(f"Loading fine-tuned weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Fine-tuned weights loaded.")

    model.eval()

    failed = []

    for video_path in tqdm(videos, desc="Videos"):
        output_path = get_output_path(video_path, args.output_suffix)
        try:
            frame_indices, timestamps_ms = compute_frame_indices(video_path, args.fps)
            embeddings = extract_features(
                video_path, model, preprocess, frame_indices, args.batch_size, args.device
            )
            save_features(output_path, embeddings, timestamps_ms)
            log.info(f"DONE: {output_path.name} | shape={embeddings.shape}")

        except Exception as e:
            log.error(f"FAILED: {video_path} | {e}")
            failed.append(str(video_path))

    if failed:
        log.warning(f"{len(failed)} video(s) failed:")
        for f in failed:
            log.warning(f"  {f}")


if __name__ == "__main__":
    main()
