"""
CLIP fine-tuning — best configuration addressing all identified failure modes:

  Issue 1 (weak loss):          InfoNCE with batch=16 → 15 negatives per positive per step
  Issue 2 (catastrophic forget): Visual encoder frozen; only text encoder is updated
  Issue 3 (window mismatch):    Default window=±5s matches eval inference window

Additional improvements over simple baseline:
  - LR warmup (first 10% of steps) + cosine decay → stable convergence
  - Gradient clipping (max_norm=1.0) → prevents large destabilizing updates
  - Higher LR (1e-5) safe now that only ~100M text params are being trained (vs 400M total)
  - Both anonymized AND description captions used for training (2x data, better text generalization)

Usage:
    python finetune_clip_best.py --epochs 1
    sbatch clip/run_finetune_best.sh
"""

import argparse
import json
import logging
import math
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
                   default=Path("/ibex/scratch/alhumosm/SoccerNet/checkpoints/clip_ft_best"))
    p.add_argument("--clip_model", type=str, default="ViT-L/14")
    p.add_argument("--window_s", type=float, default=5.0,
                   help="Half-window in seconds (default: 5.0 → ±5s = 10s clip, matches eval)")
    p.add_argument("--n_frames", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size (default: 16 → 15 negatives per positive per step)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="LR for text encoder only (higher than full fine-tune since vision is frozen)")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--warmup_frac", type=float, default=0.1,
                   help="Fraction of total steps used for LR warmup (default: 10%%)")
    p.add_argument("--max_norm", type=float, default=1.0,
                   help="Gradient clipping max norm (default: 1.0)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_games", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


class SoccerNetBestDataset(Dataset):
    """
    Each item: (frames [N,3,224,224], caption str).
    Uses BOTH anonymized and description captions — doubles the training set and
    teaches the text encoder to handle both domain-generic and named-entity text.
    """

    def __init__(self, caption_root, game_paths, video_root,
                 window_s, n_frames, preprocess, max_games=None):
        self.window_s  = window_s
        self.n_frames  = n_frames
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
                pos_ms = int(ann["position"])
                # anonymized caption (used in training split)
                self.samples.append((vp, pos_ms, ann["anonymized"]))
                # description caption (used at eval) — same video, different text
                if ann.get("description") and ann["description"] != ann["anonymized"]:
                    self.samples.append((vp, pos_ms, ann["description"]))

        log.info(f"Dataset: {len(self.samples)} samples ({skipped} annotations skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, center_ms, caption = self.samples[idx]
        frames = sample_frames(path, center_ms, self.window_s, self.n_frames, self.preprocess)
        return frames, caption


def collate_fn(batch):
    batch = [(f, c) for f, c in batch if f is not None]
    if not batch:
        return None, None
    return torch.stack([f for f, _ in batch]), [c for _, c in batch]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def encode_visual(model, frames: torch.Tensor, device: str) -> torch.Tensor:
    """frames [B, N, 3, 224, 224] → L2-normalized [B, D]"""
    B, N, C, H, W = frames.shape
    flat = frames.view(B * N, C, H, W).to(device)
    with torch.no_grad():                          # vision is frozen — no grad needed
        emb = model.encode_image(flat)
    emb = emb.view(B, N, -1).float().mean(dim=1)  # [B, D]
    return F.normalize(emb, dim=-1)


def encode_text(model, captions: list, device: str) -> torch.Tensor:
    """captions → L2-normalized [B, D]  (with grad for text fine-tuning)"""
    tokens = clip.tokenize(captions, truncate=True).to(device)
    emb = model.encode_text(tokens).float()        # [B, D]
    return F.normalize(emb, dim=-1)


def infonce_loss(vis_emb: torch.Tensor, txt_emb: torch.Tensor,
                 temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE. Diagonal = positives, off-diagonal = in-batch negatives."""
    B = vis_emb.shape[0]
    sim = (txt_emb @ vis_emb.T) / temperature      # [B, B]
    labels = torch.arange(B, device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def get_lr_lambda(total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, scheduler, scaler,
              device, temperature, max_norm, train=True):
    model.train() if train else model.eval()
    total_loss = total_diag_sim = 0.0
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames, captions in tqdm(loader, desc="train" if train else "val ", leave=False):
            if frames is None:
                continue

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                vis_emb = encode_visual(model, frames, device)              # [B, D], no grad
                txt_emb = encode_text(model, captions, device)              # [B, D], with grad
                loss    = infonce_loss(vis_emb, txt_emb, temperature)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            with torch.no_grad():
                diag_sim = (txt_emb * vis_emb).sum(dim=-1).mean().item()

            total_loss     += loss.item()
            total_diag_sim += diag_sim
            n += 1

    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_diag_sim / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    caption_root = args.caption_root.expanduser()
    video_root   = args.video_root.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_games = getListGames("train", task="caption")
    valid_games = getListGames("valid", task="caption")

    log.info(f"Loading CLIP {args.clip_model} ...")
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model = model.float()

    # Freeze visual encoder — preserve pretrained visual features
    for p in model.visual.parameters():
        p.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    log.info(f"Visual encoder frozen. Trainable params: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M total")

    train_ds = SoccerNetBestDataset(caption_root, train_games, video_root,
                                    args.window_s, args.n_frames, preprocess, args.max_games)
    val_ds   = SoccerNetBestDataset(caption_root, valid_games, video_root,
                                    args.window_s, args.n_frames, preprocess, args.max_games)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(total_steps * args.warmup_frac)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda")

    log.info(f"Window: ±{args.window_s}s  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info(f"Steps/epoch: {steps_per_epoch}  |  Warmup: {warmup_steps} steps  |  Total: {total_steps}")

    start_epoch   = 1
    best_val_loss = float("inf")
    history       = []

    latest_ckpt = args.output_dir / "latest.pt"
    if latest_ckpt.exists():
        log.info(f"Resuming from {latest_ckpt} ...")
        ckpt = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_loss = ckpt["best_val_loss"]
        start_epoch   = ckpt["epoch"] + 1
        history       = ckpt.get("history", [])
        log.info(f"  Resumed from epoch {start_epoch - 1}")
    else:
        log.info("Starting from scratch.")

    if start_epoch > args.epochs:
        log.info(f"Already at epoch {start_epoch - 1}/{args.epochs}. "
                 f"Re-submit with --epochs {args.epochs + 1} to continue.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_sim = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            args.device, args.temperature, args.max_norm, train=True)
        val_loss, val_sim = run_epoch(
            model, val_loader, optimizer, scheduler, scaler,
            args.device, args.temperature, args.max_norm, train=False)
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}  diag_sim={train_sim:.3f} | "
            f"val loss={val_loss:.4f}  diag_sim={val_sim:.3f} | "
            f"{elapsed/60:.1f} min"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_diag_sim": train_sim,
            "val_loss":   val_loss,   "val_diag_sim":   val_sim,
            "elapsed_min": round(elapsed / 60, 1),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss}, args.output_dir / "best.pt")
            log.info("  New best → saved best.pt")

        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_val_loss":        best_val_loss,
            "history":              history,
        }, latest_ckpt)

    with open(args.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Done. Best checkpoint: {args.output_dir}/best.pt")
    log.info("Next steps:")
    log.info(f"  Extract features: python extract_clip_features.py "
             f"--checkpoint {args.output_dir}/best.pt --output_suffix _best")
    log.info("  Evaluate:         python baseline_clip.py --eval_split test "
             "--feature_suffix _best --results_file .../clip_best_test_results.json")


if __name__ == "__main__":
    main()
