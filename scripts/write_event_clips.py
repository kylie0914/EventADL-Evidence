from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--events", required=True)
parser.add_argument("--output-dir", default="outputs/clips")
parser.add_argument(
    "--ffmpeg",
    default="/usr/bin/ffmpeg",
    help="FFmpeg executable. Use the system build to avoid incompatible Conda builds.",
)
parser.add_argument("--preset", default="medium", help="libx264 speed/size preset")
args = parser.parse_args()
ffmpeg = Path(args.ffmpeg)
if not ffmpeg.is_file():
    raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")
events = pd.read_csv(args.events)
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
for row in events.itertuples():
    destination = output / f"{row.event_id}.mp4"
    duration = float(row.end_sec) - float(row.start_sec)
    if duration <= 0:
        raise ValueError(f"잘못된 event 구간: {row.event_id}")
    subprocess.run([
        str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(row.start_sec), "-i", args.video, "-t", str(duration),
        "-c:v", "libx264", "-crf", "23", "-preset", args.preset, "-an", str(destination),
    ], check=True)
    print(destination)

