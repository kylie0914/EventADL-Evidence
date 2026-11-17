"""P04_01의 250 audit frame을 시각 검토한 GT 판정을 CSV에 적용한다.

2026-11-16에 25-frame contact sheet 10장과 애매한 원본 확대 시트 9장을
사람이 직접 검토해 만든 고정 판정이다. 모델 score/box는 판정에 사용하지 않았다.
정지 화면만으로 확인할 수 없는 프레임은 NaN으로 두어 평가에서 제외한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def expand(spec: str) -> set[int]:
    """'2-9,12,15' 같은 범위 표기를 정수 집합으로 바꾼다."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


# 화면에서 손이 실제로 보이는 frame-row. 나머지는 0이며 UNCERTAIN은 평가 제외.
HAND_POSITIVE = expand(
    "2-9,12-13,15,19-31,38-54,56,58,60-66,68-70,72-77,79-87,"
    "89-97,99-117,119-122,124-146,151-175,178,181-187,189-191,"
    "194-199,201-208,210-239,243-247"
)
UNCERTAIN = {59, 88, 98, 118, 150}

# 손은 보이지만 정지 화면에서 물체 접촉이 확인되지 않는 frame-row.
VISIBLE_NOT_INTERACTING = {13, 22, 77, 174}
INTERACTION_POSITIVE = HAND_POSITIVE - VISIBLE_NOT_INTERACTING


ACTIVE_GROUPS = {
    "jar": "2-7",
    "pot": "8",
    "pot_lid": "9,15",
    "knife": "12,83-87",
    "can_opener": "19-20,23-31,46",
    "sponge": "21",
    "cloth": "38",
    "can": "39",
    "food_package": "40-45,60-66,79-82,89-90,92-97,134",
    "spoon": "47-52,184-186,190-191,195,199,201,205-208",
    "faucet_handle": "53,175,239",
    "drawer_handle": "58,178,181,196",
    "refrigerator_door": "68-69,130-131,135,151,202",
    "milk_carton": "70",
    "food_container": "72-76,182-183,187",
    "wrapper": "91",
    "food_mixture": "99-117,124-129,132,136-145,215-220,223-238",
    "pan": "119-120,133,146,244",
    "cooktop_control": "121,203",
    "bottle": "122,189,197-198,212-213",
    "egg_carton": "152-153",
    "egg": "154-173",
    "seasoning_bottle": "194",
    "utensil": "204",
    "spatula": "210-211,222",
    "paper_towel": "214,221,243,245-247",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/audit/P04_01/audit_labels.csv")
    parser.add_argument("--confirm-overwrite", action="store_true")
    args = parser.parse_args()
    if not args.confirm_overwrite:
        raise SystemExit("Refusing to overwrite without --confirm-overwrite")

    path = Path(args.csv)
    table = pd.read_csv(path)
    if len(table) != 250:
        raise ValueError(f"Expected 250 rows, found {len(table)}")

    # 기존 사용자 입력은 요청에 따라 판정에 사용하지 않고 전부 새로 기록한다.
    table["gt_hand_visible"] = 0.0
    table.loc[list(HAND_POSITIVE), "gt_hand_visible"] = 1.0
    table.loc[list(UNCERTAIN), "gt_hand_visible"] = pd.NA

    # 이 영상은 모든 표본에 생활 물체가 하나 이상 보인다. 따라서 이 열만으로는
    # object detector precision/mAP을 평가할 수 없고 presence recall만 확인 가능하다.
    table["gt_object_visible"] = 1.0

    table["gt_interaction"] = 0.0
    table.loc[list(INTERACTION_POSITIVE), "gt_interaction"] = 1.0
    table.loc[list(UNCERTAIN), "gt_interaction"] = pd.NA

    table["gt_active_object"] = ""
    for label, spec in ACTIVE_GROUPS.items():
        indices = expand(spec)
        invalid = indices - INTERACTION_POSITIVE
        if invalid:
            raise ValueError(f"{label} assigned to non-interaction rows: {sorted(invalid)}")
        table.loc[list(indices), "gt_active_object"] = label

    # 접촉은 확실하지만 단일 정지 화면에서 물체 종류를 특정하기 어려운 소수 표본.
    missing_name = table.index[(table["gt_interaction"] == 1) & (table["gt_active_object"] == "")]
    table.loc[missing_name, "gt_active_object"] = "unknown"

    table["review_notes"] = ""
    table.loc[list(UNCERTAIN), "review_notes"] = "motion_blur; excluded_from_binary_metrics"
    table.loc[list(VISIBLE_NOT_INTERACTING), "review_notes"] = "hand_visible_without_confirmed_object_contact"
    table["gt_source"] = "codex_visual_review_v1_2026-11-16"
    table["gt_review_status"] = "reviewed"
    table.loc[list(UNCERTAIN), "gt_review_status"] = "uncertain"

    # 논리 검증: interaction이면 hand가 보여야 하고 active-object 이름이 있어야 한다.
    assert not ((table["gt_interaction"] == 1) & (table["gt_hand_visible"] != 1)).any()
    assert not ((table["gt_interaction"] == 1) & (table["gt_active_object"] == "")).any()
    assert table.loc[table["gt_interaction"] == 0, "gt_active_object"].eq("").all()

    temp = path.with_suffix(".csv.tmp")
    table.to_csv(temp, index=False)
    temp.replace(path)
    print(f"Wrote {len(table)} reviewed rows to {path}")
    print(f"hand: {int((table.gt_hand_visible == 1).sum())} positive, "
          f"{int((table.gt_hand_visible == 0).sum())} negative, {len(UNCERTAIN)} uncertain")
    print(f"interaction: {int((table.gt_interaction == 1).sum())} positive, "
          f"{int((table.gt_interaction == 0).sum())} negative, {len(UNCERTAIN)} uncertain")
    print(f"active object unknown: {int((table.gt_active_object == 'unknown').sum())}")


if __name__ == "__main__":
    main()
