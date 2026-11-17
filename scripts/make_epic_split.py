from __future__ import annotations

import argparse

from adl_event_logger.annotations import make_participant_split


parser = argparse.ArgumentParser()
parser.add_argument("--annotations-dir", required=True)
parser.add_argument("--output-dir", default="data/manifests/epic")
parser.add_argument("--dev-participants", type=int, default=2)
parser.add_argument("--test-participants", type=int, default=1)
args = parser.parse_args()
dev, test = make_participant_split(
    args.annotations_dir, args.output_dir, args.dev_participants, args.test_participants
)
print(f"dev events={len(dev)}, test events={len(test)}")

