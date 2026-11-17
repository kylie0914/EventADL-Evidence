"""Build complete hand-object interaction episodes from 100DOH detections.

Unlike the v2 change-window method, this module opens one episode when an
interaction becomes stable and closes it only after the interaction has been
absent for a configurable grace period.  Object/contact changes are retained as
sub-events instead of automatically becoming separate video clips.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from adl_event_logger.video import read_metadata


def interacting_hands(row: dict) -> list[dict]:
    """Return confident hands that 100DOH classifies as touching something."""
    return [
        hand for hand in row.get("hands", [])
        if int(hand.get("contact_state_id", 0)) > 0
    ]


def signature(row: dict) -> tuple:
    """Compact state used only to record interpretable within-episode changes."""
    result = []
    objects = row.get("objects", [])
    for hand in sorted(interacting_hands(row), key=lambda item: item.get("side", "")):
        linked = hand.get("linked_object_index")
        box = None
        if isinstance(linked, int) and 0 <= linked < len(objects):
            raw = objects[linked].get("bbox")
            if raw and len(raw) == 4:
                # Quantization prevents tiny detector jitter from becoming a change.
                width, height = max(1, row["width"]), max(1, row["height"])
                box = tuple(round(value / scale, 1) for value, scale in zip(raw, (width, height, width, height)))
        result.append((hand.get("side", "unknown"), hand.get("contact_state", "unknown"), box))
    return tuple(result)


def make_episode(
    index: int,
    video_id: str,
    active_start: float,
    active_end: float,
    video_duration: float,
    pre: float,
    post: float,
    sub_events: list[dict],
) -> dict:
    return {
        "video_id": video_id,
        "event_id": f"episode_{index:04d}",
        "start_sec": max(0.0, active_start - pre),
        "end_sec": min(video_duration, active_end + post),
        "active_start_sec": active_start,
        "active_end_sec": active_end,
        "active_duration_sec": max(0.0, active_end - active_start),
        "method": "interaction_episode_v4",
        "sub_event_count": len(sub_events),
        "sub_events_json": json.dumps(sub_events, ensure_ascii=False),
    }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--detections", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--start-confirm-samples", type=int, default=2)
parser.add_argument("--end-confirm-seconds", type=float, default=2.0)
parser.add_argument("--pre-seconds", type=float, default=2.0)
parser.add_argument("--post-seconds", type=float, default=2.0)
parser.add_argument("--minimum-active-seconds", type=float, default=0.5)
args = parser.parse_args()

rows = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise ValueError(f"No detections in {args.detections}")
rows.sort(key=lambda row: float(row["time_sec"]))
metadata = read_metadata(args.video)
video_id = args.video.stem
sample_period = float(pd.Series([row["time_sec"] for row in rows]).diff().median())
if not math.isfinite(sample_period) or sample_period <= 0:
    raise ValueError("Could not determine detector sampling period")
end_confirm_samples = max(1, math.ceil(args.end_confirm_seconds / sample_period))

episodes: list[dict] = []
active = False
positive_run = 0
negative_run = 0
pending_start = 0.0
last_positive = 0.0
last_signature: tuple | None = None
sub_events: list[dict] = []

for row in rows:
    time_sec = float(row["time_sec"])
    present = bool(interacting_hands(row))
    current_signature = signature(row) if present else ()

    if not active:
        if present:
            if positive_run == 0:
                pending_start = time_sec
            positive_run += 1
            if positive_run >= args.start_confirm_samples:
                active = True
                negative_run = 0
                last_positive = time_sec
                last_signature = current_signature
                sub_events = [{"time_sec": pending_start, "type": "interaction_start"}]
        else:
            positive_run = 0
        continue

    if present:
        negative_run = 0
        last_positive = time_sec
        if current_signature != last_signature:
            sub_events.append({
                "time_sec": time_sec,
                "type": "hand_object_or_contact_change",
                "state": current_signature,
            })
            last_signature = current_signature
        continue

    negative_run += 1
    if negative_run >= end_confirm_samples:
        sub_events.append({"time_sec": last_positive, "type": "interaction_end"})
        if last_positive - pending_start >= args.minimum_active_seconds:
            episodes.append(make_episode(
                len(episodes), video_id, pending_start, last_positive,
                metadata.duration_seconds, args.pre_seconds, args.post_seconds, sub_events,
            ))
        active = False
        positive_run = 0
        negative_run = 0
        last_signature = None
        sub_events = []

if active:
    sub_events.append({"time_sec": last_positive, "type": "interaction_end_at_video_boundary"})
    if last_positive - pending_start >= args.minimum_active_seconds:
        episodes.append(make_episode(
            len(episodes), video_id, pending_start, last_positive,
            metadata.duration_seconds, args.pre_seconds, args.post_seconds, sub_events,
        ))

table = pd.DataFrame(episodes)
args.output.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(args.output, index=False)
saved = float((table.end_sec - table.start_sec).sum()) if not table.empty else 0.0
print(
    f"samples={len(rows)}, approximate_fps={1/sample_period:.3f}, episodes={len(table)}, "
    f"saved_seconds={saved:.2f}, output={args.output}"
)
