"""Build short evidence clips from 100DOH interaction-change triggers.

Fusion v2 does not save every frame while contact remains active.  It detects
changes (interaction lifecycle, linked-object geometry, contact state, and
motion onset during interaction) and opens a fixed evidence window around each
change.  This is intentionally separate from Fusion v1's continuous OR rule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from adl_event_logger.video import read_metadata


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def center_distance(a: list[float], b: list[float], width: int, height: int) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(ax - bx, ay - by) / math.hypot(width, height)


def summarize_interactions(row: dict) -> dict[str, dict]:
    """Keep the highest-score interacting hand for each side."""
    result: dict[str, dict] = {}
    objects = row.get("objects", [])
    for hand in row.get("hands", []):
        if int(hand.get("contact_state_id", 0)) <= 0:
            continue
        side = str(hand.get("side", "unknown"))
        linked = hand.get("linked_object_index")
        bbox = None
        if isinstance(linked, int) and 0 <= linked < len(objects):
            bbox = objects[linked].get("bbox")
        candidate = {
            "score": float(hand.get("score", 0.0)),
            "state_id": int(hand.get("contact_state_id", 0)),
            "state": str(hand.get("contact_state", "unknown")),
            "object_bbox": bbox,
        }
        if side not in result or candidate["score"] > result[side]["score"]:
            result[side] = candidate
    return result


def apply_cooldown(triggers: list[dict], seconds: float) -> list[dict]:
    kept, last_by_kind = [], {}
    for trigger in sorted(triggers, key=lambda x: (x["time_sec"], x["candidate"])):
        previous = last_by_kind.get(trigger["candidate"], -math.inf)
        if trigger["time_sec"] - previous >= seconds:
            kept.append(trigger)
            last_by_kind[trigger["candidate"]] = trigger["time_sec"]
    return kept


def windows_from_triggers(
    triggers: list[dict], duration: float, pre: float, post: float,
    merge_gap: float, max_duration: float, method: str,
) -> pd.DataFrame:
    windows = []
    for trigger in sorted(triggers, key=lambda x: x["time_sec"]):
        start = max(0.0, trigger["time_sec"] - pre)
        end = min(duration, trigger["time_sec"] + post)
        if windows:
            previous = windows[-1]
            merged_end = max(previous["end_sec"], end)
            if start - previous["end_sec"] <= merge_gap and merged_end - previous["start_sec"] <= max_duration:
                previous["end_sec"] = merged_end
                previous["trigger_count"] += 1
                previous["trigger_reasons"].append(trigger["reason"])
                continue
        windows.append({
            "start_sec": start,
            "end_sec": end,
            "peak_score": float(trigger.get("score", 1.0)),
            "method": method,
            "trigger_count": 1,
            "trigger_reasons": [trigger["reason"]],
        })
    rows = []
    for index, window in enumerate(windows):
        window["trigger_reasons"] = "|".join(sorted(set(window["trigger_reasons"])))
        rows.append({"video_id": args.video.stem, "event_id": f"{method}_{index:04d}", **window})
    return pd.DataFrame(rows, columns=[
        "video_id", "event_id", "start_sec", "end_sec", "peak_score",
        "method", "trigger_count", "trigger_reasons",
    ])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", required=True, type=Path)
parser.add_argument("--detections", required=True, type=Path)
parser.add_argument("--motion-scores", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--motion-threshold", type=float, default=0.08)
parser.add_argument("--pre-seconds", type=float, default=4.0)
parser.add_argument("--post-seconds", type=float, default=6.0)
parser.add_argument("--merge-gap-seconds", type=float, default=2.0)
parser.add_argument("--max-clip-seconds", type=float, default=20.0)
parser.add_argument("--trigger-cooldown-seconds", type=float, default=2.0)
parser.add_argument(
    "--union-cooldown-seconds", type=float, default=20.0,
    help="Global refractory period for the final A-D union evidence logger",
)
parser.add_argument("--object-iou-threshold", type=float, default=0.15)
parser.add_argument("--object-center-distance", type=float, default=0.10)
args = parser.parse_args()

rows = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line.strip()]
motion = pd.read_csv(args.motion_scores).sort_values("time_sec")
motion_by_frame = dict(zip(motion.frame_index.astype(int), motion.motion_score.astype(float)))
metadata = read_metadata(args.video)
triggers = []
previous_interactions: dict[str, dict] = {}
previous_motion_active = False
stable_active = False
active_run = inactive_run = 0
last_visible_interactions: dict[str, dict] = {}

for row in rows:
    time_sec = float(row["time_sec"])
    frame_index = int(row["frame_index"])
    raw_current = summarize_interactions(row)
    if raw_current:
        active_run += 1
        inactive_run = 0
        last_visible_interactions = raw_current
        if not stable_active and active_run >= 2:
            stable_active = True
    else:
        inactive_run += 1
        active_run = 0
        if stable_active and inactive_run >= 2:
            stable_active = False
    # Bridge a single missing 2-FPS detection. This prevents detector jitter
    # from becoming a false interaction end followed by a false restart.
    current = (raw_current or last_visible_interactions) if stable_active else {}
    current_active = bool(current)
    previous_active = bool(previous_interactions)
    motion_score = float(motion_by_frame.get(frame_index, 0.0))
    motion_active = motion_score >= args.motion_threshold

    # A: lifecycle change. Both start and end are evidence-bearing boundaries.
    if current_active != previous_active:
        transition = "start" if current_active else "end"
        triggers.append({"time_sec": time_sec, "candidate": "a", "reason": f"interaction_{transition}", "score": 1.0})

    # B: the same hand is still interacting but the linked target-object box
    # jumps enough that it is unlikely to be the same physical object.
    for side in set(current).intersection(previous_interactions):
        now_box = current[side]["object_bbox"]
        old_box = previous_interactions[side]["object_bbox"]
        if now_box and old_box:
            overlap = iou(now_box, old_box)
            displacement = center_distance(now_box, old_box, int(row["width"]), int(row["height"]))
            if overlap < args.object_iou_threshold and displacement >= args.object_center_distance:
                triggers.append({"time_sec": time_sec, "candidate": "b", "reason": f"{side}_linked_object_change", "score": 1.0 - overlap})

    # C: contact type changes while the same side remains interacting.
    for side in set(current).intersection(previous_interactions):
        if current[side]["state_id"] != previous_interactions[side]["state_id"]:
            triggers.append({
                "time_sec": time_sec, "candidate": "c",
                "reason": f"{side}_contact_{previous_interactions[side]['state']}_to_{current[side]['state']}",
                "score": current[side]["score"],
            })

    # D: motion begins while a hand-object interaction is already active.
    if current_active and motion_active and not previous_motion_active:
        triggers.append({"time_sec": time_sec, "candidate": "d", "reason": "motion_onset_during_interaction", "score": motion_score})

    previous_interactions = current
    previous_motion_active = motion_active

triggers = apply_cooldown(triggers, args.trigger_cooldown_seconds)
args.output_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(triggers).to_csv(args.output_dir / "trigger_audit.csv", index=False)

candidate_names = {
    "a": "fusion_v2_a_interaction_lifecycle",
    "b": "fusion_v2_b_object_change",
    "c": "fusion_v2_c_contact_state_change",
    "d": "fusion_v2_d_motion_during_interaction",
    "union": "fusion_v2_union",
}
for candidate, method in candidate_names.items():
    selected = triggers if candidate == "union" else [t for t in triggers if t["candidate"] == candidate]
    if candidate == "union":
        # A lifecycle boundary is the primary evidence. B/C/D are secondary
        # cues that fill time regions not already represented by A. This avoids
        # a noisy motion onset suppressing a nearby interaction start merely
        # because it occurred a fraction of a second earlier.
        union_selected = []
        for priority in ("a", "b", "c", "d"):
            for trigger in (t for t in selected if t["candidate"] == priority):
                if all(
                    abs(trigger["time_sec"] - kept["time_sec"])
                    >= args.union_cooldown_seconds
                    for kept in union_selected
                ):
                    union_selected.append(trigger)
        selected = sorted(union_selected, key=lambda x: x["time_sec"])
    table = windows_from_triggers(
        selected, metadata.duration_seconds, args.pre_seconds, args.post_seconds,
        args.merge_gap_seconds, args.max_clip_seconds, method,
    )
    destination = args.output_dir / f"{method}.csv"
    table.to_csv(destination, index=False)
    saved = float((table.end_sec - table.start_sec).sum()) if not table.empty else 0.0
    print(f"{method}: triggers={len(selected)}, clips={len(table)}, summed_clip_sec={saved:.2f}")
