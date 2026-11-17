"""Create publication-ready result figures from the final P04_01 CSV files."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/portfolio"))
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})

# 1) Detector audit: v0 distance heuristic vs learned 100DOH.
audit = pd.read_csv("outputs/reports/P04_01_100doh/v0_vs_100doh.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, target in zip(axes, ["hand", "interaction"]):
    table = audit[audit.target == target].set_index("method")
    x = np.arange(len(table))
    width = 0.25
    for offset, metric, color in [
        (-width, "precision", "#2563a6"),
        (0, "recall", "#e87524"),
        (width, "f1", "#159a8c"),
    ]:
        bars = ax.bar(x + offset, table[metric], width, label=metric.title(), color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, table.index)
    ax.set_title(f"{target.title()} detection")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Score")
axes[1].legend(loc="lower right")
fig.suptitle("Detector audit on 245 labeled P04_01 frames")
fig.tight_layout()
fig.savefig(args.output_dir / "detector_audit_v0_vs_100doh.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 2) Recall-storage trade-off: baselines/Fusion v1 plus v2 candidates.
v1 = pd.read_csv("outputs/tables/P04_01_fusion_v1_comparison.csv")
v2 = pd.read_csv("outputs/tables/P04_01_fusion_v2_comparison.csv")
v1_names = {
    "full": "Full", "periodic": "Periodic", "motion": "Motion",
    "interaction_100doh": "100DOH continuous", "fusion_100doh": "Fusion v1 OR",
}
v2_names = {
    "fusion_v2_a_interaction_lifecycle": "v2-A lifecycle",
    "fusion_v2_b_object_change": "v2-B object change",
    "fusion_v2_c_contact_state_change": "v2-C contact state",
    "fusion_v2_d_motion_during_interaction": "v2-D motion+interaction",
    "fusion_v2_union": "v2 budgeted union",
}
fig, ax = plt.subplots(figsize=(9, 6))
for _, row in v1.iterrows():
    recall = row["event_recall_at_0.8"]
    ax.scatter(row.storage_reduction, recall, s=70, color="#667085", marker="o")
    ax.annotate(v1_names[row.method], (row.storage_reduction, recall), xytext=(5, 5), textcoords="offset points", fontsize=8)
for _, row in v2.iterrows():
    recall = row["recall_at_80pct_coverage"]
    ax.scatter(row.storage_reduction, recall, s=90, color="#1677b8", marker="D")
    ax.annotate(v2_names[row.method], (row.storage_reduction, recall), xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Storage reduction (higher is better)")
ax.set_ylabel("Event recall at 80% GT coverage (higher is better)")
ax.set_title("P04_01 recall–storage trade-off")
ax.set_xlim(-0.03, 0.82)
ax.set_ylim(0.15, 1.05)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(args.output_dir / "event_recall_storage_tradeoff_final.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 3) VLM accuracy, speed and VRAM dashboard.
vlm = pd.read_csv("outputs/vlm_benchmark/P04_01_fusion_v2_union/reports/model_comparison.csv")
labels = {
    "smolvlm2_2b": "SmolVLM2-2.2B",
    "qwen3_vl_4b": "Qwen3-VL-4B",
    "internvl3_5_4b": "InternVL3.5-4B",
}
vlm["label"] = vlm.model.map(labels)
order = ["SmolVLM2-2.2B", "Qwen3-VL-4B", "InternVL3.5-4B"]
vlm = vlm.set_index("label").loc[order].reset_index()
colors = ["#7a8aa0", "#e87524", "#1677b8"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
metrics = [
    ("object_accuracy", "Object exact accuracy", "Accuracy", lambda x: f"{x:.1%}"),
    ("mean_inference_sec", "Mean inference time", "Seconds / clip", lambda x: f"{x:.2f}"),
    ("peak_vram_mib", "Peak allocated VRAM", "MiB", lambda x: f"{x:.0f}"),
]
for ax, (column, title, ylabel, formatter) in zip(axes, metrics):
    bars = ax.bar(vlm.label, vlm[column], color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    ax.bar_label(bars, labels=[formatter(x) for x in vlm[column]], padding=3, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(vlm[column]) * 1.18)
fig.suptitle("VLM pilot comparison on the same 43 clips and 8 frames")
fig.tight_layout()
fig.savefig(args.output_dir / "vlm_model_comparison_final.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(args.output_dir)
