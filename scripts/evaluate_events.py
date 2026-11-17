from __future__ import annotations

import argparse
import json
from pathlib import Path

from adl_event_logger.evaluation import evaluate_events
from adl_event_logger.io_utils import write_json


parser = argparse.ArgumentParser()
parser.add_argument("--gt", required=True)
parser.add_argument("--predictions", required=True)
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--output", required=True)
args = parser.parse_args()
metrics = evaluate_events(args.gt, args.predictions, args.threshold)
write_json(args.output, metrics)
print(json.dumps(metrics, indent=2))

