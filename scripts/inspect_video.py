from __future__ import annotations

import argparse
import json
from pathlib import Path

from adl_event_logger.io_utils import write_json
from adl_event_logger.video import metadata_dict, sample_uniform_rgb
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--output-dir", default="outputs/video_check")
parser.add_argument("--frames", type=int, default=5)
args = parser.parse_args()
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
metadata = metadata_dict(args.video)
write_json(output / "metadata.json", metadata)
for index, frame in enumerate(sample_uniform_rgb(args.video, args.frames)):
    Image.fromarray(frame).save(output / f"frame_{index:02d}.jpg")
print(json.dumps(metadata, indent=2))

