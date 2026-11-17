from __future__ import annotations

from pathlib import Path

import pandas as pd


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    """겹치는 구간을 한 번만 세어 전체 저장 시간을 계산한다."""
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    merged = [list(valid[0])]
    for start, end in valid[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def interval_overlap_fraction(gt_start: float, gt_end: float, predictions: pd.DataFrame) -> float:
    intervals = []
    for row in predictions.itertuples():
        start, end = max(gt_start, row.start_sec), min(gt_end, row.end_sec)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    return covered / (gt_end - gt_start)


def storage_metrics(predictions: pd.DataFrame, video_duration: float) -> dict[str, float]:
    """Full-video logging 대비 저장 비율과 감소율을 반환한다."""
    saved = merged_duration(list(predictions[["start_sec", "end_sec"]].itertuples(index=False, name=None)))
    ratio = min(saved / video_duration, 1.0) if video_duration > 0 else 0.0
    return {
        "video_duration_sec": video_duration,
        "saved_duration_sec": saved,
        "storage_ratio": ratio,
        "storage_reduction": 1.0 - ratio,
    }


def evaluate_events(gt_csv: str | Path, predicted_csv: str | Path, threshold: float = 0.8) -> dict[str, float]:
    gt = pd.read_csv(gt_csv)
    predicted = pd.read_csv(predicted_csv)
    fractions = []
    for row in gt.itertuples():
        candidates = predicted[predicted.video_id == row.video_id]
        fractions.append(interval_overlap_fraction(row.start_sec, row.stop_sec, candidates))
    recall = sum(value >= threshold for value in fractions) / len(fractions) if fractions else 0.0
    return {"event_count": float(len(fractions)), "event_recall": recall, "mean_gt_coverage": sum(fractions) / len(fractions) if fractions else 0.0}

