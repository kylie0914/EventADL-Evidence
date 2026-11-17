#!/usr/bin/env bash
# P04_01 전체 영상에서 100DOH와 Motion+100DOH Fusion v1을 끝까지 실행한다.
# 완료된 산출물은 검증 후 건너뛰므로 오류 수정 뒤 같은 명령으로 재실행할 수 있다.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_ID="P04_01"
VIDEO="$PROJECT_ROOT/data/raw/epic/${VIDEO_ID}.MP4"
GT="$PROJECT_ROOT/data/manifests/epic/${VIDEO_ID}_gt.csv"
MOTION_SCORES="$PROJECT_ROOT/outputs/motion/${VIDEO_ID}/scores.csv"
MOTION_EVENTS="$PROJECT_ROOT/outputs/motion/${VIDEO_ID}/events.csv"
CHECKPOINT="$PROJECT_ROOT/tools/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"
RAW_JSONL="$PROJECT_ROOT/outputs/100doh/${VIDEO_ID}_full_2fps.jsonl"
SCORES_CSV="$PROJECT_ROOT/outputs/detectors/${VIDEO_ID}_100doh_scores.csv"
INTERACTION_EVENTS="$PROJECT_ROOT/outputs/events/${VIDEO_ID}_interaction_100doh.csv"
FUSION_EVENTS="$PROJECT_ROOT/outputs/events/${VIDEO_ID}_fusion_100doh.csv"
CLIPS_DIR="$PROJECT_ROOT/outputs/clips/${VIDEO_ID}_fusion_100doh"
MAIN_PY="${ADL_MAIN_PY:-$HOME/miniconda3/envs/adl-event-logger/bin/python}"
DOH_PY="${ADL_100DOH_PY:-$HOME/miniconda3/envs/handobj-100doh/bin/python}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/outputs/logs/${VIDEO_ID}_100doh_fusion_v1_${RUN_STAMP}.log"

mkdir -p "$PROJECT_ROOT/outputs/logs" "$PROJECT_ROOT/outputs/100doh" \
  "$PROJECT_ROOT/outputs/detectors" "$PROJECT_ROOT/outputs/events" \
  "$PROJECT_ROOT/outputs/metrics" "$PROJECT_ROOT/outputs/clips" "$CLIPS_DIR"
cd "$PROJECT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[실패] line=$1, exit_code=$exit_code"
  echo "문제를 수정한 뒤 같은 명령을 다시 실행하면 완료 단계부터 이어집니다."
  echo "로그: $LOG_FILE"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

echo "============================================================"
echo "P04_01 100DOH + Fusion v1 시작: $(date --iso-8601=seconds)"
echo "메인 Python: $MAIN_PY"
echo "100DOH Python: $DOH_PY"
echo "로그: $LOG_FILE"
echo "============================================================"

echo "[1/10] 입력·환경·checkpoint 검증"
for required in "$VIDEO" "$GT" "$MOTION_SCORES" "$MOTION_EVENTS" "$CHECKPOINT" "$MAIN_PY" "$DOH_PY"; do
  if [[ ! -e "$required" ]]; then
    echo "[오류] 필요한 파일이 없습니다: $required"
    exit 2
  fi
done
EXPECTED_SHA="3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03"
ACTUAL_SHA="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "[오류] 100DOH checkpoint SHA256이 다릅니다: $ACTUAL_SHA"
  exit 3
fi
"$MAIN_PY" -c "import cv2, pandas, yaml, adl_event_logger; print('main environment: OK')"
(cd "$PROJECT_ROOT/tools/hand_object_detector" && "$DOH_PY" -c "import cv2, torch, torchvision; print('100DOH CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available()")

echo "[2/10] 영상 2 FPS 예상 표본 수 계산"
EXPECTED_SAMPLES="$("$MAIN_PY" -c "from adl_event_logger.video import read_metadata; m=read_metadata('$VIDEO'); step=max(1, round(m.fps/2.0)); print((m.frame_count+step-1)//step)")"
echo "예상 표본 수: $EXPECTED_SAMPLES"

echo "[3/10] P04_01 전체에 100DOH GPU 추론"
CURRENT_LINES=0
if [[ -f "$RAW_JSONL" ]]; then CURRENT_LINES="$(wc -l < "$RAW_JSONL")"; fi
if [[ "$CURRENT_LINES" -eq "$EXPECTED_SAMPLES" ]]; then
  echo "검증된 line 수가 있어 추론을 건너뜁니다: $RAW_JSONL ($CURRENT_LINES lines)"
else
  echo "현재 line=$CURRENT_LINES, expected=$EXPECTED_SAMPLES: 처음부터 안전하게 다시 생성합니다."
  (cd "$PROJECT_ROOT/tools/hand_object_detector" && \
    "$DOH_PY" infer_json.py \
      --video "$VIDEO" --sample-fps 2.0 \
      --checkpoint "$CHECKPOINT" --device cuda \
      --hand-threshold 0.5 --object-threshold 0.5 \
      --output "$RAW_JSONL")
fi
[[ "$(wc -l < "$RAW_JSONL")" -eq "$EXPECTED_SAMPLES" ]]

echo "[4/10] 100DOH JSONL을 시간축 score CSV로 변환"
"$MAIN_PY" scripts/convert_100doh_scores.py --input "$RAW_JSONL" --output "$SCORES_CSV"

echo "[5/10] 100DOH 상호작용 단독 이벤트 생성"
"$MAIN_PY" scripts/run_100doh_event_method.py \
  --video "$VIDEO" --scores "$SCORES_CSV" \
  --method interaction_100doh --interaction-threshold 0.5 \
  --output "$INTERACTION_EVENTS"

echo "[6/10] Motion OR 100DOH Fusion v1 이벤트 생성"
"$MAIN_PY" scripts/run_100doh_event_method.py \
  --video "$VIDEO" --scores "$SCORES_CSV" --motion-scores "$MOTION_SCORES" \
  --method fusion_100doh --interaction-threshold 0.5 \
  --output "$FUSION_EVENTS"

echo "[7/10] Event Recall·GT coverage·Storage Reduction 평가"
for method in interaction_100doh fusion_100doh; do
  "$MAIN_PY" scripts/evaluate_events.py \
    --gt "$GT" --predictions "$PROJECT_ROOT/outputs/events/${VIDEO_ID}_${method}.csv" \
    --threshold 0.8 --output "$PROJECT_ROOT/outputs/metrics/${VIDEO_ID}_${method}_event.json"
  "$MAIN_PY" scripts/evaluate_storage.py \
    --video "$VIDEO" --events "$PROJECT_ROOT/outputs/events/${VIDEO_ID}_${method}.csv" \
    --output "$PROJECT_ROOT/outputs/metrics/${VIDEO_ID}_${method}_storage.json"
done

echo "[8/10] Fusion v1 이벤트 MP4 clip 저장"
EVENT_COUNT="$("$MAIN_PY" -c "import pandas as pd; print(len(pd.read_csv('$FUSION_EVENTS')))")"
CLIPS_COMPLETE="$("$MAIN_PY" -c "from pathlib import Path; import pandas as pd; t=pd.read_csv('$FUSION_EVENTS'); d=Path('$CLIPS_DIR'); print(int(len(t)>0 and all((d/f'{x}.mp4').is_file() for x in t.event_id)))")"
if [[ "$CLIPS_COMPLETE" -eq 1 ]]; then
  echo "현재 event ID의 clip이 모두 있어 인코딩을 건너뜁니다: $EVENT_COUNT clips"
else
  "$MAIN_PY" scripts/write_event_clips.py \
    --video "$VIDEO" --events "$FUSION_EVENTS" --output-dir "$CLIPS_DIR" \
    --ffmpeg /usr/bin/ffmpeg
fi

echo "[9/10] Fusion v1 timeline 그림 생성"
"$MAIN_PY" scripts/plot_event_timeline.py \
  --gt "$GT" --predictions "$FUSION_EVENTS" --video-id "$VIDEO_ID" \
  --output "$PROJECT_ROOT/outputs/figures/${VIDEO_ID}_fusion_100doh_timeline.png"

echo "[10/10] 핵심 방법 비교표 생성"
"$MAIN_PY" scripts/summarize_fusion_v1.py \
  --metrics-dir "$PROJECT_ROOT/outputs/metrics" \
  --events-dir "$PROJECT_ROOT/outputs/events" --motion-events "$MOTION_EVENTS" \
  --output "$PROJECT_ROOT/outputs/tables/${VIDEO_ID}_fusion_v1_comparison.csv"

echo "============================================================"
echo "완료: $(date --iso-8601=seconds)"
echo "100DOH scores: $SCORES_CSV"
echo "Fusion events: $FUSION_EVENTS"
echo "Fusion clips: $CLIPS_DIR"
echo "비교표: $PROJECT_ROOT/outputs/tables/${VIDEO_ID}_fusion_v1_comparison.csv"
echo "로그: $LOG_FILE"
echo "다음 단계: 생성된 Fusion v1 clip에 SmolVLM2/Qwen/InternVL을 비교합니다."
echo "============================================================"
