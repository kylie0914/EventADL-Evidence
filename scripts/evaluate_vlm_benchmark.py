"""Compare VLM verb, object and verb-object ADL predictions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def normalize(value: object) -> str:
    text = str(value).lower().replace("_", "-").strip()
    return re.sub(r"\s+", "-", text)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gt", required=True, type=Path)
parser.add_argument("--predictions-dir", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path)
args = parser.parse_args()
gt = pd.read_csv(args.gt)
args.output_dir.mkdir(parents=True, exist_ok=True)
summary = []

for path in sorted(args.predictions_dir.glob("*.jsonl")):
    predictions = pd.read_json(path, lines=True)
    if predictions.empty:
        continue
    model_key = str(predictions.model_key.iloc[0])
    joined = gt.merge(predictions, on="event_id", how="inner", validate="one_to_one")
    joined["verb_gt_norm"] = joined.verb_gt.map(normalize)
    joined["verb_pred_norm"] = joined.verb.map(normalize)
    joined["object_gt_norm"] = joined.object_gt.map(normalize)
    joined["object_pred_norm"] = joined.object.map(normalize)
    joined["adl_gt_norm"] = joined.verb_gt_norm + "|" + joined.object_gt_norm
    joined["adl_pred_norm"] = joined.verb_pred_norm + "|" + joined.object_pred_norm
    row = {
        "model": model_key,
        "model_id": str(predictions.model_id.iloc[0]),
        "clips": len(joined),
        "completion_rate": len(joined) / len(gt),
        "json_valid_rate": float((joined.parse_status == "valid_json").mean()),
        "uncertain_rate": float(joined.uncertain.astype(bool).mean()),
        "verb_accuracy": accuracy_score(joined.verb_gt_norm, joined.verb_pred_norm),
        "verb_macro_f1": f1_score(joined.verb_gt_norm, joined.verb_pred_norm, average="macro", zero_division=0),
        "object_accuracy": accuracy_score(joined.object_gt_norm, joined.object_pred_norm),
        "object_macro_f1": f1_score(joined.object_gt_norm, joined.object_pred_norm, average="macro", zero_division=0),
        "adl_accuracy": accuracy_score(joined.adl_gt_norm, joined.adl_pred_norm),
        "adl_macro_f1": f1_score(joined.adl_gt_norm, joined.adl_pred_norm, average="macro", zero_division=0),
        "mean_inference_sec": float(joined.inference_sec.mean()),
        "median_inference_sec": float(joined.inference_sec.median()),
        "peak_vram_mib": float(joined.peak_vram_mib.max()),
    }
    summary.append(row)
    joined.to_csv(args.output_dir / f"{model_key}_per_clip.csv", index=False)

table = pd.DataFrame(summary).sort_values("model")
table.to_csv(args.output_dir / "model_comparison.csv", index=False)
(args.output_dir / "model_comparison.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(table.to_string(index=False))
