#!/usr/bin/env bash
# 기존 P04_01 100DOH JSONL을 재사용해 change-triggered Fusion v2를 실행한다.

set -Eeuo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="$PROJECT_ROOT/data/raw/epic/P04_01.MP4"
GT="$PROJECT_ROOT/data/manifests/epic/P04_01_gt.csv"
DETECTIONS="$PROJECT_ROOT/outputs/100doh/P04_01_full_2fps.jsonl"
MOTION="$PROJECT_ROOT/outputs/motion/P04_01/scores.csv"
EVENTS_DIR="$PROJECT_ROOT/outputs/events/P04_01_fusion_v2"
UNION_EVENTS="$EVENTS_DIR/fusion_v2_union.csv"
CLIPS_DIR="$PROJECT_ROOT/outputs/clips/P04_01_fusion_v2_union"
RESULTS="$PROJECT_ROOT/outputs/tables/P04_01_fusion_v2_comparison.csv"
PYTHON="${ADL_MAIN_PY:-$HOME/miniconda3/envs/adl-event-logger/bin/python}"
LOG="$PROJECT_ROOT/outputs/logs/P04_01_fusion_v2_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$EVENTS_DIR" "$CLIPS_DIR" "$PROJECT_ROOT/outputs/tables" \
  "$PROJECT_ROOT/outputs/figures" "$PROJECT_ROOT/outputs/logs"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG") 2>&1

on_error() {
  local code=$?
  echo "[실패] line=$1, exit_code=$code"
  echo "수정 후 같은 명령을 다시 실행하면 됩니다. 로그: $LOG"
  exit "$code"
}
trap 'on_error $LINENO' ERR

echo "============================================================"
echo "P04_01 change-triggered Fusion v2: $(date --iso-8601=seconds)"
echo "기존 100DOH 결과를 재사용하므로 detector 추론은 하지 않습니다."
echo "============================================================"

echo "[1/6] 입력 검증"
for file in "$VIDEO" "$GT" "$DETECTIONS" "$MOTION" "$PYTHON" /usr/bin/ffmpeg; do
  [[ -e "$file" ]] || { echo "[오류] 파일 없음: $file"; exit 2; }
done
[[ "$(wc -l < "$DETECTIONS")" -eq 2308 ]] || { echo "[오류] 100DOH JSONL은 2308행이어야 합니다."; exit 3; }
"$PYTHON" -c "import pandas, cv2, adl_event_logger; print('environment: OK')"

echo "[2/6] A·B·C·D 변화 trigger 및 10초 증거 window 생성"
"$PYTHON" scripts/build_fusion_v2_events.py \
  --video "$VIDEO" --detections "$DETECTIONS" --motion-scores "$MOTION" \
  --output-dir "$EVENTS_DIR" \
  --pre-seconds 4 --post-seconds 6 \
  --merge-gap-seconds 2 --max-clip-seconds 20 \
  --trigger-cooldown-seconds 2 --union-cooldown-seconds 20 \
  --motion-threshold 0.08 \
  --object-iou-threshold 0.15 --object-center-distance 0.10

echo "[3/6] 후보별 Coverage 20%·50%·80%와 저장량 평가"
"$PYTHON" scripts/evaluate_fusion_v2.py \
  --gt "$GT" --events-dir "$EVENTS_DIR" --video "$VIDEO" --output "$RESULTS"

echo "[4/6] A~D 합집합의 짧은 MP4 증거 clip 저장"
EVENT_COUNT="$("$PYTHON" -c "import pandas as pd; print(len(pd.read_csv('$UNION_EVENTS')))")"
CLIPS_COMPLETE="$("$PYTHON" -c "from pathlib import Path; import pandas as pd; t=pd.read_csv('$UNION_EVENTS'); d=Path('$CLIPS_DIR'); print(int(len(t)>0 and all((d/f'{x}.mp4').is_file() for x in t.event_id)))")"
if [[ "$CLIPS_COMPLETE" -eq 1 ]]; then
  echo "현재 event ID의 clip이 모두 있어 건너뜁니다: $EVENT_COUNT clips"
else
  "$PYTHON" scripts/write_event_clips.py \
    --video "$VIDEO" --events "$UNION_EVENTS" --output-dir "$CLIPS_DIR" \
    --ffmpeg /usr/bin/ffmpeg --preset veryfast
fi

echo "[5/6] GT와 Fusion v2 타임라인 생성"
"$PYTHON" scripts/plot_event_timeline.py \
  --gt "$GT" --predictions "$UNION_EVENTS" --video-id P04_01 \
  --output "$PROJECT_ROOT/outputs/figures/P04_01_fusion_v2_timeline.png"

echo "[6/6] 산출물 검증"
"$PYTHON" -c "import pandas as pd; t=pd.read_csv('$UNION_EVENTS'); assert len(t)==$EVENT_COUNT; assert ((t.end_sec-t.start_sec)<=20.0001).all(); print('union clips:',len(t)); print('maximum clip seconds:',round((t.end_sec-t.start_sec).max(),2))"

echo "============================================================"
echo "Fusion v2 완료"
echo "비교표: $RESULTS"
echo "Trigger audit: $EVENTS_DIR/trigger_audit.csv"
echo "최종 event: $UNION_EVENTS"
echo "최종 clips: $CLIPS_DIR"
echo "로그: $LOG"
echo "결과를 확인한 뒤 VLM 입력 방법을 결정합니다."
echo "============================================================"
