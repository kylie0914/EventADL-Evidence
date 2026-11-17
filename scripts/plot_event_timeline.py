from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--gt", required=True)
parser.add_argument("--predictions", required=True)
parser.add_argument("--video-id", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
gt = pd.read_csv(args.gt)
pred = pd.read_csv(args.predictions)
gt = gt[gt.video_id == args.video_id]
pred = pred[pred.video_id == args.video_id]
fig, ax = plt.subplots(figsize=(14, 3))
for row in gt.itertuples():
    ax.broken_barh([(row.start_sec, row.stop_sec - row.start_sec)], (10, 6), facecolors="tab:green")
for row in pred.itertuples():
    ax.broken_barh([(row.start_sec, row.end_sec - row.start_sec)], (20, 6), facecolors="tab:blue")
ax.set_yticks([13, 23], labels=["GT", "Predicted"])
ax.set_xlabel("Seconds")
ax.set_title(args.video_id)
fig.tight_layout()
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=160)
print(destination)

