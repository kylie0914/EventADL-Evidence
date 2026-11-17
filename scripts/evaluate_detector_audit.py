"""사람이 채운 audit 표를 이용해 detector 성능을 계산한다.

audit_labels.csv의 빈 gt 열을 0/1로 채운 뒤 실행한다. Detector 자체가 약하면
후단 VLM 비교가 불공정해지므로, 본 실험 전에 이 검사를 통과시킨다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


TARGETS = {
    "hand": ("gt_hand_visible", "hand_score"),
    "object": ("gt_object_visible", "object_score"),
    # active_object_score가 실제로 평가하는 것은 물체 이름/box가 아니라
    # 이 프레임에 손-물체 interaction이 있는지 여부다.
    "interaction": ("gt_interaction", "active_object_score"),
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit-csv", required=True)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--output", required=True)
args = parser.parse_args()

table = pd.read_csv(args.audit_csv)
results: dict[str, object] = {"frames": len(table), "threshold": args.threshold}
for name, (gt_column, score_column) in TARGETS.items():
    if gt_column not in table or score_column not in table:
        continue
    labeled = table.dropna(subset=[gt_column, score_column])
    truth = labeled[gt_column].astype(int)
    predicted = (labeled[score_column] >= args.threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else None
    balanced_accuracy = (recall + specificity) / 2 if specificity is not None else None
    results[name] = {
        "labeled_frames": len(labeled),
        "positive_gt": int(truth.sum()),
        "negative_gt": int((truth == 0).sum()),
        "true_positive": int(((truth == 1) & (predicted == 1)).sum()),
        "false_positive": fp,
        "false_negative": int(((truth == 1) & (predicted == 0)).sum()),
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }
    if truth.nunique() < 2:
        results[name]["warning"] = (
            "Ground truth contains only one class; this sample cannot support a "
            "trustworthy precision/specificity claim."
        )

destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
