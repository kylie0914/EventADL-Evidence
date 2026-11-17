from __future__ import annotations

import argparse
import json

from adl_event_logger.annotations import analyze_annotations


parser = argparse.ArgumentParser()
parser.add_argument("--annotations-dir", required=True)
parser.add_argument("--output-dir", default="data/manifests/epic")
args = parser.parse_args()
print(json.dumps(analyze_annotations(args.annotations_dir, args.output_dir), indent=2))

