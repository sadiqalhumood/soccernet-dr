"""
Shared utilities for moment retrieval baselines.
Imported by baseline_clip.py and baseline_qwen.py.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def discover_games(caption_root: Path) -> list[Path]:
    """Find all Labels-caption.json files under caption_root."""
    return sorted(caption_root.rglob("Labels-caption.json"))


def load_annotations(caption_json: Path) -> list[dict]:
    """Load annotations filtered to visibility == 'shown'."""
    with open(caption_json) as f:
        data = json.load(f)
    return [a for a in data["annotations"] if a.get("visibility") == "shown"]


def get_half(annotation: dict) -> int:
    """Parse the half number from gameTime, e.g. '2 - 47:31' → 2."""
    return int(annotation["gameTime"].split(" - ")[0])


def smooth_scores(scores: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Apply Gaussian smoothing along the time axis."""
    return gaussian_filter1d(scores, sigma=sigma_frames)


def nms_topk(timestamps_ms, scores: np.ndarray, min_distance_ms: float, k: int) -> list[int]:
    """
    Greedy NMS: rank all positions by score, then suppress any position
    within min_distance_ms of an already-selected position.
    Returns up to k timestamps (in ms), sorted by score descending.
    """
    timestamps_ms = np.asarray(timestamps_ms)
    order = np.argsort(scores)[::-1]
    selected = []

    for idx in order:
        t = int(timestamps_ms[idx])
        if all(abs(t - s) >= min_distance_ms for s in selected):
            selected.append(t)
        if len(selected) >= k:
            break

    return selected


def is_hit(predictions: list[int], ground_truth_ms: int, delta_ms: float, k: int) -> bool:
    """True if any of the top-k predictions is within delta_ms of ground truth."""
    return any(abs(p - ground_truth_ms) <= delta_ms for p in predictions[:k])


# ---------------------------------------------------------------------------
# Action spotting utilities
# ---------------------------------------------------------------------------

def discover_spotting_games(feature_root: Path) -> list[Path]:
    """Find all Labels-v2.json files under feature_root."""
    return sorted(feature_root.rglob("Labels-v2.json"))


def load_spotting_annotations(labels_json: Path) -> dict:
    """
    Load Labels-v2.json and return {label: [(half, position_ms), ...]} for all annotations.
    Skips annotations for half > 2 (extra time — no features extracted).
    """
    with open(labels_json) as f:
        data = json.load(f)
    result: dict[str, list] = {}
    for ann in data["annotations"]:
        half = int(ann["gameTime"].split(" - ")[0])
        if half > 2:
            continue
        label = ann["label"]
        pos_ms = int(ann["position"])
        result.setdefault(label, []).append((half, pos_ms))
    return result
