"""Sample every episode at a target FPS and split long inputs into safe chunks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--clips-dir", required=True, type=Path)
parser.add_argument("--episodes", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
parser.add_argument("--sample-fps", type=float, default=3.0)
parser.add_argument("--chunk-seconds", type=float, default=10.0)
parser.add_argument("--overlap-seconds", type=float, default=1.0)
parser.add_argument("--resize-width", type=int, default=640)
args = parser.parse_args()

if args.sample_fps <= 0 or args.chunk_seconds <= 0:
    raise ValueError("sample FPS and chunk seconds must be positive")
if not 0 <= args.overlap_seconds < args.chunk_seconds:
    raise ValueError("overlap must be >= 0 and shorter than one chunk")

episodes = pd.read_csv(args.episodes)
frames_root = args.output_dir / "frames"
frames_root.mkdir(parents=True, exist_ok=True)
records = []

for episode in episodes.itertuples():
    clip = args.clips_dir / f"{episode.event_id}.mp4"
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip}")
    native_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / native_fps
    step_seconds = args.chunk_seconds - args.overlap_seconds
    starts = [0.0]
    next_start = step_seconds
    # Do not create a nearly empty tail chunk that contains only the overlap.
    while next_start + args.overlap_seconds < duration:
        starts.append(next_start)
        next_start += step_seconds

    for chunk_index, local_start in enumerate(starts):
        local_end = min(duration, local_start + args.chunk_seconds)
        segment_id = f"{episode.event_id}_part_{chunk_index:02d}"
        destination = frames_root / segment_id
        destination.mkdir(exist_ok=True)
        desired = max(2, int(math.ceil((local_end - local_start) * args.sample_fps)))
        times = np.linspace(local_start, local_end, desired, endpoint=False)
        paths = []
        for number, time_sec in enumerate(times):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {clip} near {time_sec:.3f}s")
            if args.resize_width > 0 and image.shape[1] != args.resize_width:
                scale = args.resize_width / image.shape[1]
                image = cv2.resize(image, (args.resize_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            path = destination / f"frame_{number:03d}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Could not write {path}")
            paths.append(str(path.resolve()))
        records.append({
            "event_id": segment_id,
            "episode_id": episode.event_id,
            "chunk_index": chunk_index,
            "clip": str(clip.resolve()),
            "start_sec": float(episode.start_sec) + local_start,
            "end_sec": float(episode.start_sec) + local_end,
            "frame_count": len(paths),
            "frame_paths_json": json.dumps(paths),
        })
    capture.release()

manifest = pd.DataFrame(records)
args.output_dir.mkdir(parents=True, exist_ok=True)
manifest.to_csv(args.output_dir / "manifest.csv", index=False)
policy = {
    "sampling_fps": args.sample_fps,
    "chunk_seconds": args.chunk_seconds,
    "overlap_seconds": args.overlap_seconds,
    "resize_width": args.resize_width,
    "note": "Long episodes are chunked so target FPS is preserved instead of reducing all episodes to 8 frames.",
}
(args.output_dir / "sampling_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
print(f"episodes={len(episodes)}, VLM_segments={len(manifest)}, frames={manifest.frame_count.sum()}, output={args.output_dir}")
