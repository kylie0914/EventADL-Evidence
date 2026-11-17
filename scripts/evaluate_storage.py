"""Event CSV가 원본 영상보다 저장량을 얼마나 줄였는지 계산한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from adl_event_logger.evaluation import storage_metrics
from adl_event_logger.video import read_metadata


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", required=True)
parser.add_argument("--events", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

metrics = storage_metrics(pd.read_csv(args.events), read_metadata(args.video).duration_seconds)
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
