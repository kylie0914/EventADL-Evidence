"""100DOH JSONL predictions를 P04 detector-audit GT와 비교한다.

이미지 basename으로 두 파일을 연결하며, GT가 미정(NaN)인 프레임은 해당 binary
metric에서 제외한다. 결과 JSON, 프레임별 판정 CSV, 기존 v0 비교 CSV를 저장한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


TARGETS = {
    "hand": ("gt_hand_visible", "has_hand"),
    "interaction": ("gt_interaction", "has_interaction"),
}


def binary_metrics(truth: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="binary", zero_division=0
    )
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "labeled_frames": int(len(truth)),
        "positive_gt": int(truth.sum()),
        "negative_gt": int((truth == 0).sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity) if specificity is not None else None,
        "balanced_accuracy": float((recall + specificity) / 2) if specificity is not None else None,
        "f1": float(f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit = pd.read_csv(args.audit_csv)
    predictions = [
        json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
    ]
    by_name = {Path(row["image"]).name: row for row in predictions}
    if len(by_name) != len(predictions):
        raise ValueError("Duplicate image basenames in prediction JSONL")

    audit_names = audit["image_path"].map(lambda value: Path(value).name)
    missing = sorted(set(audit_names) - set(by_name))
    extra = sorted(set(by_name) - set(audit_names))
    if missing or extra:
        raise ValueError(f"Image join mismatch: missing={missing[:5]}, extra={extra[:5]}")

    joined = audit.copy()
    joined["image_name"] = audit_names
    joined["pred_hand"] = audit_names.map(lambda name: int(by_name[name]["has_hand"]))
    joined["pred_interaction"] = audit_names.map(
        lambda name: int(by_name[name]["has_interaction"])
    )
    joined["pred_hand_count"] = audit_names.map(lambda name: len(by_name[name]["hands"]))
    joined["pred_object_count"] = audit_names.map(lambda name: len(by_name[name]["objects"]))
    joined["inference_seconds"] = audit_names.map(
        lambda name: by_name[name]["inference_seconds"]
    )

    result: dict[str, object] = {
        "model": "100DOH+ego",
        "frames_in_jsonl": len(predictions),
        "mean_inference_seconds": float(joined["inference_seconds"].mean()),
        "frames_per_second": float(1 / joined["inference_seconds"].mean()),
        "total_hands": int(joined["pred_hand_count"].sum()),
        "total_target_objects": int(joined["pred_object_count"].sum()),
    }
    comparison_rows = []
    baseline = (
        json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if args.baseline_json and args.baseline_json.exists()
        else {}
    )

    for target, (gt_column, prediction_column) in TARGETS.items():
        labeled = joined.dropna(subset=[gt_column]).copy()
        truth = labeled[gt_column].astype(int)
        predicted = labeled[f"pred_{target}"].astype(int)
        result[target] = binary_metrics(truth, predicted)

        labeled["error_type"] = "correct"
        labeled.loc[(truth == 1) & (predicted == 0), "error_type"] = "false_negative"
        labeled.loc[(truth == 0) & (predicted == 1), "error_type"] = "false_positive"
        joined.loc[labeled.index, f"{target}_error"] = labeled["error_type"]

        if target in baseline:
            comparison_rows.append(
                {"method": "v0", "target": target, **{
                    key: baseline[target].get(key)
                    for key in ["precision", "recall", "specificity", "balanced_accuracy", "f1"]
                }}
            )
        comparison_rows.append(
            {"method": "100DOH+ego", "target": target, **{
                key: result[target][key]
                for key in ["precision", "recall", "specificity", "balanced_accuracy", "f1"]
            }}
        )

    contact_counts: dict[str, int] = {}
    for row in predictions:
        for hand in row["hands"]:
            name = hand["contact_state"]
            contact_counts[name] = contact_counts.get(name, 0) + 1
    result["contact_state_counts"] = contact_counts

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    joined.to_csv(args.output_dir / "frame_results.csv", index=False)
    errors = joined[
        joined["hand_error"].isin(["false_positive", "false_negative"])
        | joined["interaction_error"].isin(["false_positive", "false_negative"])
    ]
    errors.to_csv(args.output_dir / "failure_frames.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "v0_vs_100doh.csv", index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Failure frames: {len(errors)} -> {args.output_dir / 'failure_frames.csv'}")


if __name__ == "__main__":
    main()
