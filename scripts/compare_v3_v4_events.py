"""Compare v3 change windows and v4 complete episodes on the same GT timeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adl_event_logger.evaluation import evaluate_events, storage_metrics
from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", required=True, type=Path)
parser.add_argument("--v3-events", required=True, type=Path)
parser.add_argument("--v4-events", required=True, type=Path)
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

duration = read_metadata(args.video).duration_seconds
rows = []
for method, path in (("v3_change_evidence", args.v3_events), ("v4_complete_episode", args.v4_events)):
    events = pd.read_csv(path)
    storage = storage_metrics(events, duration)
    row = {
        "method": method,
        "clips_or_episodes": len(events),
        "saved_duration_sec": storage["saved_duration_sec"],
        "storage_ratio": storage["storage_ratio"],
        "storage_reduction": storage["storage_reduction"],
    }
    for threshold in (0.2, 0.5, 0.8):
        metrics = evaluate_events(args.gt, path, threshold)
        row[f"recall_at_{int(threshold * 100)}pct_coverage"] = metrics["event_recall"]
        if threshold == 0.8:
            row["mean_gt_coverage"] = metrics["mean_gt_coverage"]
    rows.append(row)

table = pd.DataFrame(rows)
args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(table.to_string(index=False))
print(f"Saved: {args.output}")
