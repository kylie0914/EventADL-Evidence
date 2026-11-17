"""EPIC-KITCHENS annotation loading, validation, summaries, and participant splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io_utils import timestamp_to_seconds


REQUIRED_COLUMNS = {"participant_id", "video_id", "start_timestamp", "stop_timestamp"}


def find_split_csv(root: str | Path, split: str) -> Path:
    candidates = list(Path(root).rglob(f"EPIC_100_{split}.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one EPIC_100_{split}.csv under {root}; found {candidates}")
    return candidates[0]


def load_split(root: str | Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(find_split_csv(root, split))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {split}: {sorted(missing)}")
    frame = frame.copy()
    frame["split"] = split
    frame["start_sec"] = frame["start_timestamp"].map(timestamp_to_seconds)
    frame["stop_sec"] = frame["stop_timestamp"].map(timestamp_to_seconds)
    frame["duration_sec"] = frame["stop_sec"] - frame["start_sec"]
    if (frame["duration_sec"] <= 0).any():
        bad = frame.loc[frame["duration_sec"] <= 0, ["video_id", "start_timestamp", "stop_timestamp"]]
        raise ValueError(f"Non-positive event durations found:\n{bad.head()}")
    return frame


def analyze_annotations(root: str | Path, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.concat([load_split(root, "train"), load_split(root, "validation")], ignore_index=True)

    participant = (
        data.groupby(["split", "participant_id"], as_index=False)
        .agg(action_count=("video_id", "size"), video_count=("video_id", "nunique"), action_seconds=("duration_sec", "sum"))
        .sort_values(["split", "action_count"], ascending=[True, False])
    )
    video = (
        data.groupby(["split", "participant_id", "video_id"], as_index=False)
        .agg(action_count=("video_id", "size"), first_action_sec=("start_sec", "min"), last_action_sec=("stop_sec", "max"), labeled_seconds=("duration_sec", "sum"))
        .sort_values(["split", "participant_id", "video_id"])
    )
    participant.to_csv(output / "participant_summary.csv", index=False)
    video.to_csv(output / "video_summary.csv", index=False)
    data.head(200).to_csv(output / "sample_actions.csv", index=False)
    summary = {
        "rows": int(len(data)),
        "participants": int(data.participant_id.nunique()),
        "videos": int(data.video_id.nunique()),
        "splits": data.groupby("split").size().astype(int).to_dict(),
        "columns": list(data.columns),
    }
    (output / "annotation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def make_participant_split(root: str | Path, output_dir: str | Path, dev_count: int = 2, test_count: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a leakage-safe pilot split using disjoint participants.

    This is deliberately deterministic: participants with the most labeled actions are
    selected so the first pilot has enough events. The resulting CSVs become fixed inputs.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = load_split(root, "train")
    validation = load_split(root, "validation")
    dev_participants = (
        train.groupby("participant_id").size().sort_values(ascending=False).head(dev_count).index.tolist()
    )

    # EPIC의 공식 train/validation은 participant-held-out split이 아니다. 따라서
    # validation의 상위 참가자가 dev와 겹칠 수 있다. 사람별 생활 습관이 양쪽에 섞이는
    # 데이터 누수를 막기 위해 dev 참가자를 제외한 뒤 test 참가자를 선택한다.
    validation_counts = validation.groupby("participant_id").size().sort_values(ascending=False)
    eligible_test = validation_counts[~validation_counts.index.isin(dev_participants)]
    test_participants = eligible_test.head(test_count).index.tolist()
    if len(test_participants) < test_count:
        raise ValueError(
            f"겹치지 않는 test participant가 부족합니다: requested={test_count}, "
            f"available={len(eligible_test)}"
        )
    if set(dev_participants) & set(test_participants):
        raise AssertionError("Participant leakage between dev and test")
    dev = train[train.participant_id.isin(dev_participants)].copy()
    test = validation[validation.participant_id.isin(test_participants)].copy()
    keep = ["split", "participant_id", "video_id", "start_sec", "stop_sec", "duration_sec"]
    for optional in ("verb", "noun", "verb_class", "noun_class", "narration"):
        if optional in dev.columns:
            keep.append(optional)
    dev[keep].to_csv(output / "split_dev.csv", index=False)
    test[keep].to_csv(output / "split_test.csv", index=False)
    manifests = pd.concat([
        dev[["split", "participant_id", "video_id"]],
        test[["split", "participant_id", "video_id"]],
    ]).drop_duplicates().sort_values(["split", "participant_id", "video_id"])
    manifests.to_csv(output / "pilot_videos.csv", index=False)
    return dev, test

