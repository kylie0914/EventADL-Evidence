"""Trigger score functions.

Scores are normalized to 0..1 when possible. Event formation is intentionally kept in
`events.py`, separating visual evidence from temporal policy.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .video import iter_sampled_frames


def motion_scores(video: str | Path, sample_fps: float, resize_width: int) -> pd.DataFrame:
    """Cheap baseline: mean absolute grayscale difference between sampled frames."""
    rows: list[dict[str, float | int]] = []
    previous: np.ndarray | None = None
    for time_sec, frame_index, frame in iter_sampled_frames(video, sample_fps, resize_width):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        score = 0.0 if previous is None else float(cv2.absdiff(gray, previous).mean() / 255.0)
        rows.append({"time_sec": time_sec, "frame_index": frame_index, "motion_score": score})
        previous = gray
    return pd.DataFrame(rows)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def bbox_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ac = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return math.dist(ac, bc) / math.sqrt(2)


def active_object_score(hand_boxes: list[tuple[float, float, float, float]], object_boxes: list[tuple[float, float, float, float]], max_distance: float) -> float:
    """Heuristic interaction score from normalized hand/object boxes.

    This is not a learned contact detector. It is an interpretable baseline to compare
    against an optional 100DOH adapter later.
    """
    if not hand_boxes or not object_boxes:
        return 0.0
    best = 0.0
    for hand in hand_boxes:
        for obj in object_boxes:
            overlap = bbox_iou(hand, obj)
            distance = bbox_distance(hand, obj)
            proximity = max(0.0, 1.0 - distance / max_distance) if max_distance > 0 else 0.0
            best = max(best, overlap, proximity)
    return best


def fuse_scores(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    result = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column not in frame:
            raise KeyError(f"Missing score column: {column}")
        result += frame[column].fillna(0).clip(0, 1) * weight / total
    return result

