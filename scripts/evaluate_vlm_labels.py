"""VLM JSONL의 verb/object 예측을 EPIC GT와 비교한다.

정확한 비교를 위해 GT event clip에 VLM을 실행하고, clip 파일명(event_id)을 키로
사용한다. Macro-F1은 자주 나오는 행동만 잘 맞히는 모델에 유리하지 않다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def normalize(value: object) -> str:
    """대소문자·구두점 차이를 제거하되 의미는 임의로 바꾸지 않는다."""
    return re.sub(r"[^a-z0-9 ]+", "", str(value).lower()).strip()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", required=True)
parser.add_argument("--predictions", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

gt = pd.read_csv(args.gt)
pred = pd.read_json(args.predictions, lines=True)
object_gt = "noun" if "noun" in gt.columns else "object"
required = {"event_id", "verb", object_gt}
missing = required - set(gt.columns)
if missing:
    raise ValueError(f"GT CSV에 필요한 열이 없습니다: {sorted(missing)}")

gt_labels = gt[["event_id", "verb", object_gt]].rename(
    columns={"verb": "verb_gt", object_gt: "object_gt"}
)
pred_labels = pred[["event_id", "verb", "object"]].rename(
    columns={"verb": "verb_pred", "object": "object_pred"}
)
joined = gt_labels.merge(pred_labels, on="event_id")
metrics: dict[str, object] = {"matched_events": len(joined)}
for target, gt_column, pred_column in (
    ("verb", "verb_gt", "verb_pred"),
    ("object", "object_gt", "object_pred"),
):
    truth = joined[gt_column].map(normalize)
    predicted = joined[pred_column].map(normalize)
    metrics[target] = {
        "accuracy": accuracy_score(truth, predicted) if len(joined) else 0.0,
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0) if len(joined) else 0.0,
    }

destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
