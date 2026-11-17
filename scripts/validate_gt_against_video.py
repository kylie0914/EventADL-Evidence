from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--split-csv", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
metadata = read_metadata(args.video)
video_id = Path(args.video).stem
events = pd.read_csv(args.split_csv)
events = events[events.video_id == video_id].copy()
events["inside_video"] = (events.start_sec >= 0) & (events.stop_sec <= metadata.duration_seconds + 0.1)
# 뒤 단계는 모든 방법에서 end_sec/event_id라는 동일한 열 이름을 사용한다.
events["end_sec"] = events["stop_sec"]
if "event_id" not in events:
    events.insert(1, "event_id", [f"gt_{index:04d}" for index in range(len(events))])
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
events.to_csv(destination, index=False)
print(f"video={video_id}, duration={metadata.duration_seconds:.3f}, events={len(events)}, valid={int(events.inside_video.sum())}")

