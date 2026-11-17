"""Evaluate Fusion v2 candidates at multiple GT-coverage requirements."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adl_event_logger.evaluation import interval_overlap_fraction, storage_metrics
from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", required=True, type=Path)
parser.add_argument("--events-dir", required=True, type=Path)
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

gt = pd.read_csv(args.gt)
duration = read_metadata(args.video).duration_seconds
rows = []
for path in sorted(args.events_dir.glob("fusion_v2_*.csv")):
    predictions = pd.read_csv(path)
    coverage = []
    for event in gt.itertuples():
        candidates = predictions[predictions.video_id == event.video_id]
        coverage.append(interval_overlap_fraction(event.start_sec, event.stop_sec, candidates))
    storage = storage_metrics(predictions, duration)
    rows.append({
        "method": path.stem,
        "clips": len(predictions),
        "recall_at_20pct_coverage": sum(x >= 0.2 for x in coverage) / len(coverage),
        "recall_at_50pct_coverage": sum(x >= 0.5 for x in coverage) / len(coverage),
        "recall_at_80pct_coverage": sum(x >= 0.8 for x in coverage) / len(coverage),
        "mean_gt_coverage": sum(coverage) / len(coverage),
        **storage,
    })

table = pd.DataFrame(rows).sort_values("method")
args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {args.output}")
