"""Connect event clips to EPIC GT and extract identical frames for every VLM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clips-dir", required=True, type=Path)
parser.add_argument("--events", required=True, type=Path)
parser.add_argument("--epic-gt", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--frames", type=int, default=8)
parser.add_argument("--resize-width", type=int, default=640)
args = parser.parse_args()

events = pd.read_csv(args.events)
gt = pd.read_csv(args.epic_gt)
args.output_dir.mkdir(parents=True, exist_ok=True)
frames_root = args.output_dir / "frames"
frames_root.mkdir(exist_ok=True)
rows = []

for event in events.itertuples():
    clip = args.clips_dir / f"{event.event_id}.mp4"
    if not clip.is_file():
        raise FileNotFoundError(clip)

    matches = gt[(gt.start_sec < event.end_sec) & (gt.stop_sec > event.start_sec)].copy()
    if not matches.empty:
        matches["overlap_sec"] = (
            np.minimum(matches.stop_sec, event.end_sec)
            - np.maximum(matches.start_sec, event.start_sec)
        ).clip(lower=0)
        matches = matches[matches.overlap_sec > 0].sort_values(
            ["overlap_sec", "start_sec"], ascending=[False, True]
        )
        primary = matches.iloc[0]
        verb, noun = str(primary.verb), str(primary.noun)
        actions = [
            {
                "verb": str(row.verb), "noun": str(row.noun),
                "narration": str(row.narration),
                "overlap_sec": round(float(row.overlap_sec), 3),
            }
            for row in matches.itertuples()
        ]
    else:
        verb = noun = "no_action"
        actions = []

    destination = frames_root / event.event_id
    destination.mkdir(exist_ok=True)
    existing = sorted(destination.glob("frame_*.jpg"))
    dimensions_match = False
    if len(existing) == args.frames:
        probe = cv2.imread(str(existing[0]))
        dimensions_match = probe is not None and (
            args.resize_width <= 0 or probe.shape[1] == args.resize_width
        )
    if len(existing) != args.frames or not dimensions_match:
        capture = cv2.VideoCapture(str(clip))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or count <= 0:
            raise RuntimeError(f"Cannot decode clip: {clip}")
        # Bin centers avoid fragile first/last frames while covering the clip.
        indices = np.linspace(0, count, args.frames + 2, dtype=int)[1:-1]
        for old in existing:
            old.unlink()
        for number, index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read frame {index} from {clip}")
            if args.resize_width > 0 and image.shape[1] != args.resize_width:
                scale = args.resize_width / image.shape[1]
                image = cv2.resize(
                    image, (args.resize_width, round(image.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            output = destination / f"frame_{number:02d}.jpg"
            if not cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Cannot write frame: {output}")
        capture.release()
        existing = sorted(destination.glob("frame_*.jpg"))

    rows.append({
        "event_id": event.event_id,
        "clip": str(clip.resolve()),
        "start_sec": float(event.start_sec),
        "end_sec": float(event.end_sec),
        "verb_gt": verb,
        "object_gt": noun,
        "adl_gt": f"{verb}|{noun}",
        "all_gt_actions_json": json.dumps(actions, ensure_ascii=False),
        "frame_paths_json": json.dumps([str(path.resolve()) for path in existing]),
    })

manifest = pd.DataFrame(rows)
manifest.to_csv(args.output_dir / "clip_gt.csv", index=False)
vocabulary = {
    "verbs": sorted(set(gt.verb.astype(str)) | {"no_action", "unknown"}),
    "objects": sorted(set(gt.noun.astype(str)) | {"no_action", "unknown"}),
    "gt_policy": "Primary action is the EPIC annotation with maximum temporal overlap.",
    "input_policy": (
        f"{args.frames} identical uniformly sampled frames per clip for all models; "
        f"resized to width {args.resize_width}px."
    ),
}
(args.output_dir / "vocabulary.json").write_text(
    json.dumps(vocabulary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(f"clips={len(manifest)}, frames_per_clip={args.frames}, output={args.output_dir}")
