"""여러 방법의 JSON metric을 한 표와 trade-off 그래프로 모은다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metrics-dir", required=True)
parser.add_argument("--output-csv", required=True)
parser.add_argument("--output-figure", required=True)
args = parser.parse_args()

rows = []
for path in sorted(Path(args.metrics_dir).glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 같은 stem의 event/storage 결과를 합치기 쉽도록 파일명 앞부분을 방법명으로 쓴다.
    method = path.stem.removesuffix("_event").removesuffix("_storage")
    rows.append({"method": method, **payload})
table = pd.DataFrame(rows).groupby("method", as_index=False).first()
csv_path = Path(args.output_csv)
csv_path.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(csv_path, index=False)

if {"event_recall", "storage_reduction"}.issubset(table.columns):
    figure_path = Path(args.output_figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(table.storage_reduction, table.event_recall)
    for row in table.itertuples():
        plt.annotate(row.method, (row.storage_reduction, row.event_recall))
    plt.xlabel("Storage reduction (higher is better)")
    plt.ylabel("Event recall (higher is better)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
print(table.to_string(index=False))
