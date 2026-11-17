"""Create portfolio figures for the final v4 episode pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("outputs/v4/P04_01"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/v4"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1) v3-v4 temporal-selection trade-off.
table = pd.read_csv(args.root / "v3_vs_v4_event_comparison.csv")
labels = ["v3\nchange evidence", "v4\ncomplete episodes"]
recall = table.recall_at_80pct_coverage.to_numpy() * 100
reduction = table.storage_reduction.to_numpy() * 100
x = np.arange(2)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
width = 0.34
b1 = ax.bar(x - width / 2, recall, width, label="Event recall @ 80% coverage", color="#2563eb")
b2 = ax.bar(x + width / 2, reduction, width, label="Storage reduction", color="#f59e0b")
ax.bar_label(b1, fmt="%.1f%%", padding=3)
ax.bar_label(b2, fmt="%.1f%%", padding=3)
ax.set_xticks(x, labels)
ax.set_ylim(0, 100)
ax.set_ylabel("Percent")
ax.set_title("P04_01: v3 vs v4 temporal-selection trade-off")
ax.legend(loc="upper center", ncol=2, frameon=False)
ax.grid(axis="y", alpha=.22)
fig.tight_layout()
fig.savefig(args.output_dir / "v3_vs_v4_tradeoff.png", dpi=220)
plt.close(fig)

# 2) Episode duration distribution.
episodes = pd.read_csv(args.root / "events/interaction_episodes.csv")
durations = episodes.end_sec - episodes.start_sec
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
axes[0].hist(durations, bins=[0, 5, 10, 20, 40, 80, 160], color="#14b8a6", edgecolor="white")
axes[0].set_xlabel("Episode duration (seconds)")
axes[0].set_ylabel("Episode count")
axes[0].set_title("37 detected interaction episodes")
axes[0].grid(axis="y", alpha=.2)
longest = episodes.assign(duration=durations).nlargest(5, "duration")
axes[1].barh(longest.event_id[::-1], longest.duration[::-1], color="#7c3aed")
axes[1].set_xlabel("Seconds")
axes[1].set_title("Five longest episodes")
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1fs", padding=3)
fig.suptitle("v4 keeps long interactions intact, then chunks only the VLM input", fontsize=12)
fig.tight_layout()
fig.savefig(args.output_dir / "v4_episode_durations.png", dpi=220)
plt.close(fig)

# 3) Final workflow figure.
fig, ax = plt.subplots(figsize=(14, 5.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.2)
ax.axis("off")
steps = [
    (0.3, 2.7, 1.8, 1.25, "Untrimmed video", "P04_01 / GoPro\noriginal FPS"),
    (2.45, 2.7, 1.8, 1.25, "100DOH + ego", "hand, contact, target\n3 FPS monitoring"),
    (4.6, 2.7, 1.8, 1.25, "Episode builder", "stable start → end\n2 s missing grace"),
    (6.75, 2.7, 1.8, 1.25, "Episode clips", "±2 s context\noriginal FPS MP4"),
    (8.9, 2.7, 1.8, 1.25, "Adaptive sampler", "3 FPS\n10 s + 1 s overlap"),
    (11.05, 2.7, 1.8, 1.25, "Qwen3-VL-4B", "verb / object /\ndescription JSON"),
]
colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#ede9fe", "#cffafe", "#fee2e2"]
for (x0, y0, w, h, title, body), color in zip(steps, colors):
    box = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor="#334155", linewidth=1.2)
    ax.add_patch(box)
    ax.text(x0 + w / 2, y0 + .85, title, ha="center", va="center", weight="bold", fontsize=10)
    ax.text(x0 + w / 2, y0 + .38, body, ha="center", va="center", fontsize=8.5, color="#334155")
for index in range(len(steps) - 1):
    x1 = steps[index][0] + steps[index][2]
    x2 = steps[index + 1][0]
    ax.annotate("", xy=(x2 - .05, 3.33), xytext=(x1 + .05, 3.33), arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"})

ax.text(5.5, 1.55, "Within each episode", ha="center", weight="bold", color="#475569")
ax.text(5.5, .95, "object/contact changes → sub-events (do not split the saved episode)", ha="center", bbox={"boxstyle":"round,pad=.35", "facecolor":"#f8fafc", "edgecolor":"#94a3b8"}, fontsize=9)
ax.annotate("", xy=(5.5, 2.65), xytext=(5.5, 1.35), arrowprops={"arrowstyle":"->", "lw":1.3, "color":"#64748b"})
ax.text(10.8, 1.55, "Outputs", ha="center", weight="bold", color="#475569")
ax.text(10.8, .82, "episode CSV + MP4 clips + chunk JSONL + episode summary\n+ storage metrics + v3/v4 comparison", ha="center", bbox={"boxstyle":"round,pad=.4", "facecolor":"#f8fafc", "edgecolor":"#94a3b8"}, fontsize=9)
ax.annotate("", xy=(11.95, 2.65), xytext=(11.1, 1.35), arrowprops={"arrowstyle":"->", "lw":1.3, "color":"#64748b"})
ax.set_title("Final v4 — Interaction-Episode ADL Video Logging with Adaptive VLM Sampling", fontsize=15, weight="bold", pad=14)
fig.tight_layout()
fig.savefig(args.output_dir / "workflow_v4.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"Saved figures to {args.output_dir}")
