"""Build a compact comparison table for the P04_01 Fusion v1 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metrics-dir", required=True, type=Path)
parser.add_argument("--events-dir", required=True, type=Path)
parser.add_argument("--motion-events", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

methods = {
    "full": args.events_dir / "P04_01_full.csv",
    "periodic": args.events_dir / "P04_01_periodic.csv",
    "motion": args.motion_events,
    "interaction_100doh": args.events_dir / "P04_01_interaction_100doh.csv",
    "fusion_100doh": args.events_dir / "P04_01_fusion_100doh.csv",
}
rows = []
for method, events_path in methods.items():
    event_metrics_path = args.metrics_dir / f"P04_01_{method}_event.json"
    storage_metrics_path = args.metrics_dir / f"P04_01_{method}_storage.json"
    if not event_metrics_path.exists() or not storage_metrics_path.exists():
        continue
    event_metrics = json.loads(event_metrics_path.read_text(encoding="utf-8"))
    storage_metrics = json.loads(storage_metrics_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "method": method,
            "saved_event_clips": len(pd.read_csv(events_path)),
            "event_recall_at_0.8": event_metrics["event_recall"],
            "mean_gt_coverage": event_metrics["mean_gt_coverage"],
            "saved_duration_sec": storage_metrics["saved_duration_sec"],
            "storage_ratio": storage_metrics["storage_ratio"],
            "storage_reduction": storage_metrics["storage_reduction"],
        }
    )

table = pd.DataFrame(rows)
args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(table.to_string(index=False))
print(f"\nSaved: {args.output}")
