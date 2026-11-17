from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adl_event_logger.events import events_as_dicts, expand_and_merge, scores_to_events
from adl_event_logger.io_utils import load_yaml
from adl_event_logger.triggers import motion_scores
from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--output-dir", default="outputs/motion")
args = parser.parse_args()
config = load_yaml(args.config)
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
scores = motion_scores(args.video, config["video"]["detection_fps"], config["video"]["resize_width"])
scores.to_csv(output / "scores.csv", index=False)
rows = list(scores[["time_sec", "motion_score"]].itertuples(index=False, name=None))
raw = scores_to_events(rows, config["triggers"]["motion_threshold"], "motion", config["events"]["min_active_samples"], config["events"]["min_inactive_samples"])
meta = read_metadata(args.video)
events = expand_and_merge(raw, meta.duration_seconds, config["events"]["pre_seconds"], config["events"]["post_seconds"], config["events"]["merge_gap_seconds"])
table = pd.DataFrame(events_as_dicts(events))
if not table.empty:
    table.insert(0, "video_id", Path(args.video).stem)
    table.insert(1, "event_id", [f"motion_{i:04d}" for i in range(len(table))])
table.to_csv(output / "events.csv", index=False)
print(f"samples={len(scores)}, events={len(table)}")

