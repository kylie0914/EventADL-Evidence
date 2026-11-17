from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adl_event_logger.events import events_as_dicts, expand_and_merge, scores_to_events
from adl_event_logger.io_utils import load_yaml
from adl_event_logger.triggers import fuse_scores
from adl_event_logger.video import read_metadata


"""동일한 event 후처리로 모든 trigger 방법을 공정하게 비교한다.

Full/Periodic는 detector가 없는 기준선이고, 나머지는 score CSV를 thresholding한 뒤
동일한 hysteresis, 앞뒤 buffer, event merge 규칙을 적용한다.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--method", choices=["full", "periodic", "hand", "object", "active_object", "fusion"], required=True)
parser.add_argument("--scores")
parser.add_argument("--motion-scores", help="Motion CSV required by the fusion method")
parser.add_argument("--config", default="configs/pilot.yaml")
parser.add_argument("--period-seconds", type=float, default=10.0)
parser.add_argument("--clip-seconds", type=float, default=8.0)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = load_yaml(args.config)
meta = read_metadata(args.video)

if args.method == "full":
    events = [{"start_sec": 0.0, "end_sec": meta.duration_seconds, "peak_score": 1.0, "method": "full"}]
elif args.method == "periodic":
    events = []
    start = 0.0
    while start < meta.duration_seconds:
        events.append({"start_sec": start, "end_sec": min(meta.duration_seconds, start + args.clip_seconds), "peak_score": 1.0, "method": "periodic"})
        start += args.period_seconds
else:
    if not args.scores:
        raise ValueError("--scores is required for detector-based methods")
    scores = pd.read_csv(args.scores)
    if args.method == "fusion":
        if not args.motion_scores:
            raise ValueError("--motion-scores is required for fusion")
        motion = pd.read_csv(args.motion_scores)
        scores = pd.merge_asof(scores.sort_values("time_sec"), motion.sort_values("time_sec"), on="time_sec", direction="nearest")
        scores["fusion_score"] = fuse_scores(scores.rename(columns={"motion_score": "motion", "hand_score": "hand", "active_object_score": "active_object"}), config["triggers"]["fusion_weights"])
        column, threshold = "fusion_score", config["triggers"]["active_object_threshold"]
    else:
        column = f"{args.method}_score"
        threshold = config["triggers"][f"{args.method}_threshold"]
    # 순간적인 한 프레임 오검출을 event로 만들지 않도록 연속 sample 조건을 둔다.
    raw = scores_to_events(list(scores[["time_sec", column]].itertuples(index=False, name=None)), threshold, args.method, config["events"]["min_active_samples"], config["events"]["min_inactive_samples"])
    # 실제 행동의 시작/끝이 잘리지 않도록 buffer를 붙이고 가까운 event는 합친다.
    events = events_as_dicts(expand_and_merge(raw, meta.duration_seconds, config["events"]["pre_seconds"], config["events"]["post_seconds"], config["events"]["merge_gap_seconds"]))

table = pd.DataFrame(events)
table.insert(0, "video_id", Path(args.video).stem)
table.insert(1, "event_id", [f"{args.method}_{index:04d}" for index in range(len(table))])
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(destination, index=False)
print(f"method={args.method}, events={len(table)}, output={destination}")

