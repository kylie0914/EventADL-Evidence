"""Create 100DOH-only or Motion+100DOH Fusion v1 temporal events."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adl_event_logger.events import events_as_dicts, expand_and_merge, scores_to_events
from adl_event_logger.io_utils import load_yaml
from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", required=True)
parser.add_argument("--scores", required=True)
parser.add_argument("--motion-scores")
parser.add_argument(
    "--method", choices=["interaction_100doh", "fusion_100doh"], required=True
)
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--interaction-threshold", type=float, default=0.5)
parser.add_argument("--output", required=True)
args = parser.parse_args()

config = load_yaml(args.config)
scores = pd.read_csv(args.scores).sort_values("time_sec")
score_column = "interaction_score"
threshold = args.interaction_threshold

if args.method == "fusion_100doh":
    if not args.motion_scores:
        raise ValueError("--motion-scores is required for fusion_100doh")
    motion = pd.read_csv(args.motion_scores).sort_values("time_sec")
    scores = pd.merge_asof(
        scores,
        motion[["time_sec", "motion_score"]],
        on="time_sec",
        direction="nearest",
        tolerance=0.26,
    )
    if scores.motion_score.isna().any():
        missing = int(scores.motion_score.isna().sum())
        raise ValueError(f"Could not align {missing} motion samples with 100DOH")
    # Fusion v1 is an interpretable OR rule: save when either motion passes its
    # validated threshold or 100DOH detects hand-object contact.  Encoding the
    # motion decision as 0/1 lets both cues share the same temporal postprocess.
    motion_active = (
        scores.motion_score >= float(config["triggers"]["motion_threshold"])
    ).astype(float)
    scores["fusion_score"] = np.maximum(scores.interaction_score, motion_active)
    score_column = "fusion_score"

raw = scores_to_events(
    list(scores[["time_sec", score_column]].itertuples(index=False, name=None)),
    threshold,
    args.method,
    int(config["events"]["min_active_samples"]),
    int(config["events"]["min_inactive_samples"]),
)
metadata = read_metadata(args.video)
events = expand_and_merge(
    raw,
    metadata.duration_seconds,
    float(config["events"]["pre_seconds"]),
    float(config["events"]["post_seconds"]),
    float(config["events"]["merge_gap_seconds"]),
)
table = pd.DataFrame(events_as_dicts(events))
if table.empty:
    table = pd.DataFrame(columns=["start_sec", "end_sec", "peak_score", "method"])
table.insert(0, "video_id", Path(args.video).stem)
table.insert(
    1, "event_id", [f"{args.method}_{index:04d}" for index in range(len(table))]
)
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(destination, index=False)
print(f"method={args.method}, samples={len(scores)}, events={len(table)}, output={destination}")
