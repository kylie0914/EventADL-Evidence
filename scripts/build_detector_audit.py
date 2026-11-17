from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--scores", required=True)
parser.add_argument("--count", type=int, default=250)
parser.add_argument("--output-dir", default="data/audit")
args = parser.parse_args()
scores = pd.read_csv(args.scores)
chosen = scores.iloc[np.linspace(0, len(scores) - 1, min(args.count, len(scores)), dtype=int)].copy()
output = Path(args.output_dir)
images = output / "frames"
images.mkdir(parents=True, exist_ok=True)
capture = cv2.VideoCapture(args.video)
paths = []
for row in chosen.itertuples():
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(row.frame_index))
    ok, frame = capture.read()
    if not ok:
        paths.append("")
        continue
    path = images / f"frame_{int(row.frame_index):08d}.jpg"
    cv2.imwrite(str(path), frame)
    paths.append(str(path))
capture.release()
chosen["image_path"] = paths
for column in ("gt_hand_visible", "gt_object_visible", "gt_interaction", "gt_active_object", "review_notes"):
    chosen[column] = ""
chosen.to_csv(output / "audit_labels.csv", index=False)
print(f"Label {len(chosen)} rows in {output / 'audit_labels.csv'}")

