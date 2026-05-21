"""
Simple CLIP fine-tuning: 1 positive + 1 negative per caption.

For each annotation:
  - Positive: text caption + its own video frames (±2.5s window) → push similarity to 1
  - Negative: text caption + one random other clip's frames       → push similarity to 0

Loss = (1 - sim_pos)^2 + sim_neg^2

No large batch of negatives, no InfoNCE. Safe for V100 (16GB).
Saves a checkpoint after each epoch. Auto-resumes on re-submit.

To train epoch 1:   sbatch run_finetune_clip_simple.sh  (--epochs 1)
To train epoch 2:   edit --epochs to 2, re-submit
To train epoch 3:   edit --epochs to 3, re-submit
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import clip
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from SoccerNet.utils import getListGames
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import get_half, load_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--caption_root", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/caption-2024"))
    p.add_argument("--video_root", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/soccernet_videos"))
    p.add_argument("--output_dir", type=Path,
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_simple"))
    p.add_argument("--clip_model", type=str, default="ViT-L/14")
    p.add_argument("--window_s", type=float, default=2.5,
                   help="Half-window in seconds — ±2.5s = 5s clip around each annotation")
    p.add_argument("--n_frames", type=int, default=4,
                   help="Frames sampled per clip (default: 4)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Annotations per batch (default: 4 — safe for V100)")
    p.add_argument("--epochs", type=int, default=1,
                   help="Total epochs to reach. Re-submit with higher value to continue.")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None,
                   help="Limit to N games for quick testing")
    return p.parse_args()


def get_video_path(game_rel: str, video_root: Path, half: int) -> Path:
    return video_root / game_rel / f"{half}_224p.mkv"


def sample_frames(video_path: Path, center_ms: int, window_s: float,
                  n_frames: int, preprocess) -> Optional[torch.Tensor]:
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    except Exception:
        return None
    fps = vr.get_avg_fps()
    total = len(vr)
    half_w = int(window_s * fps)
    center = int(center_ms / 1000 * fps)
    start = max(0, center - half_w)
    end = min(total - 1, center + half_w)
    if start >= end:
        return None
    indices = np.linspace(start, end, n_frames, dtype=int).tolist()
    frames_np = vr.get_batch(indices).asnumpy()
    return torch.stack([preprocess(Image.fromarray(f)) for f in frames_np])


class SoccerNetSimpleDataset(Dataset):
    """
    Each item returns:
      pos_frames [N, 3, 224, 224] — frames from the annotation's own timestamp
      neg_frames [N, 3, 224, 224] — frames from a random different annotation
      caption    str              — anonymized caption text
    """

    def __init__(self, caption_root, game_paths, video_root,
                 window_s, n_frames, preprocess, max_games=None):
        self.window_s = window_s
        self.n_frames = n_frames
        self.preprocess = preprocess

        if max_games:
            game_paths = game_paths[:max_games]

        self.samples = []
        skipped = 0
        for game_rel in game_paths:
            caption_json = caption_root / game_rel / "Labels-caption.json"
            if not caption_json.exists():
                skipped += 1
                continue
            for ann in load_annotations(caption_json):
                half = get_half(ann)
                if half not in [1, 2]:
                    skipped += 1
                    continue
                vp = get_video_path(game_rel, video_root, half)
                if not vp.exists():
                    skipped += 1
                    continue
                self.samples.append((vp, int(ann["position"]), ann["anonymized"]))

        log.info(f"Dataset: {len(self.samples)} samples ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pos_path, center_ms, caption = self.samples[idx]
        pos_frames = sample_frames(pos_path, center_ms, self.window_s,
                                   self.n_frames, self.preprocess)

        # Pick any random other annotation as the negative — O(1)
        neg_idx = random.randint(0, len(self.samples) - 2)
        if neg_idx >= idx:
            neg_idx += 1
        neg_path, neg_ms, _ = self.samples[neg_idx]
        neg_frames = sample_frames(neg_path, neg_ms, self.window_s,
                                   self.n_frames, self.preprocess)

        return pos_frames, neg_frames, caption


def collate_fn(batch):
    batch = [(p, n, c) for p, n, c in batch if p is not None and n is not None]
    if not batch:
        return None, None, None
    pos = torch.stack([p for p, _, _ in batch])      # [B, N, 3, 224, 224]
    neg = torch.stack([n for _, n, _ in batch])      # [B, N, 3, 224, 224]
    captions = [c for _, _, c in batch]
    return pos, neg, captions


def encode_visual(model, frames: torch.Tensor, device: str) -> torch.Tensor:
    """frames [B, N, 3, 224, 224] → L2-normalized [B, D]"""
    B, N, C, H, W = frames.shape
    flat = frames.view(B * N, C, H, W).to(device)
    emb = model.encode_image(flat)           # [B*N, D]
    emb = emb.view(B, N, -1).mean(dim=1)    # [B, D] — mean-pool over frames
    return F.normalize(emb.float(), dim=-1)


def run_epoch(model, loader, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    total_loss = total_sim_pos = total_sim_neg = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for pos_frames, neg_frames, captions in tqdm(
                loader, desc="train" if train else "val ", leave=False):
            if pos_frames is None:
                continue

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pos_emb  = encode_visual(model, pos_frames, device)   # [B, D]
                neg_emb  = encode_visual(model, neg_frames, device)   # [B, D]
                tokens   = clip.tokenize(captions, truncate=True).to(device)
                text_emb = model.encode_text(tokens)
                text_emb = F.normalize(text_emb.float(), dim=-1)      # [B, D]

                sim_pos = (text_emb * pos_emb).sum(dim=-1)            # [B] cosine sim
                sim_neg = (text_emb * neg_emb).sum(dim=-1)            # [B] cosine sim
                # Push sim_pos → 1 and sim_neg → 0
                loss = (1 - sim_pos).pow(2).mean() + sim_neg.pow(2).mean()

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss    += loss.item()
            total_sim_pos += sim_pos.detach().mean().item()
            total_sim_neg += sim_neg.detach().mean().item()
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_sim_pos / n, total_sim_neg / n


def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    video_root   = args.video_root.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_games = getListGames("train", task="caption")
    valid_games = getListGames("valid", task="caption")
    log.info(f"Split: {len(train_games)} train / {len(valid_games)} valid games")
    log.info(f"Window: ±{args.window_s}s  |  Frames/clip: {args.n_frames}  |  Batch: {args.batch_size}")
    log.info(f"LR: {args.lr}  |  Target epochs: {args.epochs}")

    log.info(f"Loading CLIP {args.clip_model} ...")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model = model.float()

    train_ds = SoccerNetSimpleDataset(caption_root, train_games, video_root,
                                      args.window_s, args.n_frames, preprocess, args.max_games)
    val_ds   = SoccerNetSimpleDataset(caption_root, valid_games, video_root,
                                      args.window_s, args.n_frames, preprocess, args.max_games)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler()

    start_epoch   = 1
    best_val_loss = float("inf")
    history       = []

    latest_ckpt = args.output_dir / "clip_simple_latest.pt"
    if latest_ckpt.exists():
        log.info(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_epoch   = ckpt["epoch"] + 1
        history       = ckpt.get("history", [])
        log.info(f"  Resumed from epoch {start_epoch - 1}, best_val_loss={best_val_loss:.4f}")
    else:
        log.info("No checkpoint found — starting from scratch.")

    if start_epoch > args.epochs:
        log.info(f"Already at epoch {start_epoch - 1}/{args.epochs}. "
                 f"Re-submit with --epochs {args.epochs + 1} to continue.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, tr_pos, tr_neg = run_epoch(
            model, train_loader, optimizer, scaler, args.device, train=True)
        val_loss, val_pos, val_neg = run_epoch(
            model, val_loader,   optimizer, scaler, args.device, train=False)
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}  sim_pos={tr_pos:.3f}  sim_neg={tr_neg:.3f} | "
            f"val   loss={val_loss:.4f}  sim_pos={val_pos:.3f}  sim_neg={val_neg:.3f} | "
            f"{elapsed/60:.1f} min"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_sim_pos": tr_pos, "train_sim_neg": tr_neg,
            "val_loss":   val_loss,   "val_sim_pos":   val_pos, "val_sim_neg":   val_neg,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss}, args.output_dir / "clip_simple_best.pt")
            log.info(f"  New best val loss → saved clip_simple_best.pt")

        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_val_loss":        best_val_loss,
            "history":              history,
        }, latest_ckpt)
        log.info(f"  Checkpoint saved → {latest_ckpt}")

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info("Done. Re-extract features and run baseline to evaluate:")
    log.info(f"  Checkpoint: {args.output_dir}/clip_simple_best.pt")


if __name__ == "__main__":
    main()
