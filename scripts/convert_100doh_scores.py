"""Convert full-video 100DOH JSONL output to temporal trigger-score CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def maximum_score(rows: list[dict]) -> float:
    return max((float(row.get("score", 0.0)) for row in rows), default=0.0)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()

records = []
with args.input.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "time_sec" not in row or "frame_index" not in row:
            raise ValueError(
                f"Line {line_number} has no video timestamp. "
                "Run infer_json.py with --video."
            )
        hands = row.get("hands", [])
        interactions = [
            hand for hand in hands if int(hand.get("contact_state_id", 0)) > 0
        ]
        records.append(
            {
                "time_sec": float(row["time_sec"]),
                "frame_index": int(row["frame_index"]),
                "hand_score": maximum_score(hands),
                "interaction_score": maximum_score(interactions),
                # The event code historically calls the interaction cue
                # active_object_score. Keep both explicit and compatible names.
                "active_object_score": maximum_score(interactions),
                "hand_count": len(hands),
                "interaction_count": len(interactions),
                "target_object_count": len(row.get("objects", [])),
                "inference_seconds": float(row.get("inference_seconds", 0.0)),
            }
        )

table = pd.DataFrame(records).sort_values("time_sec")
if table.empty:
    raise ValueError(f"No records found in {args.input}")
if table.time_sec.duplicated().any():
    raise ValueError("Duplicate timestamps found in 100DOH output")

args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
print(
    f"samples={len(table)}, hand_frames={(table.hand_score > 0).sum()}, "
    f"interaction_frames={(table.interaction_score > 0).sum()}, "
    f"mean_inference_seconds={table.inference_seconds.mean():.4f}, "
    f"output={args.output}"
)
