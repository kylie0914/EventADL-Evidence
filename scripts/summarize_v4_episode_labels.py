"""Collect per-chunk Qwen labels into one inspectable record per episode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", required=True, type=Path)
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

episodes = pd.read_csv(args.episodes)
predictions = pd.read_json(args.predictions, lines=True)
rows = []
for episode in episodes.itertuples():
    parts = predictions[predictions.episode_id == episode.event_id].sort_values("chunk_index")
    labels = parts[["chunk_index", "verb", "object", "description", "uncertain"]].to_dict("records")
    rows.append({
        "episode_id": episode.event_id,
        "start_sec": episode.start_sec,
        "end_sec": episode.end_sec,
        "duration_sec": episode.end_sec - episode.start_sec,
        "vlm_segment_count": len(parts),
        "vlm_labels_json": json.dumps(labels, ensure_ascii=False),
    })
args.output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(args.output, index=False)
print(f"episodes={len(rows)}, output={args.output}")
