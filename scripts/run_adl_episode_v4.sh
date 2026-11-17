#!/usr/bin/env bash
# Final v4 pipeline for one arbitrary egocentric video.
# Usage: bash scripts/run_adl_episode_v4.sh /absolute/or/relative/video.mp4

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "사용법: bash scripts/run_adl_episode_v4.sh VIDEO_PATH"
  exit 2
fi
VIDEO_INPUT="$1"
if [[ "$VIDEO_INPUT" = /* ]]; then VIDEO="$VIDEO_INPUT"; else VIDEO="$PWD/$VIDEO_INPUT"; fi
[[ -f "$VIDEO" ]] || { echo "[오류] 영상이 없습니다: $VIDEO"; exit 2; }

RAW_NAME="$(basename "$VIDEO")"
VIDEO_ID="${RAW_NAME%.*}"
# Output paths remain shell-safe even when the source filename contains spaces.
SAFE_ID="$(printf '%s' "$VIDEO_ID" | tr -cs '[:alnum:]_-' '_')"
MAIN_PY="${ADL_MAIN_PY:-$HOME/miniconda3/envs/adl-event-logger/bin/python}"
DOH_PY="${ADL_100DOH_PY:-$HOME/miniconda3/envs/handobj-100doh/bin/python}"
CHECKPOINT="$ROOT/tools/hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"
RUN_ROOT="$ROOT/outputs/v4/$SAFE_ID"
DETECTIONS="$RUN_ROOT/detections/100doh_3fps.jsonl"
EPISODES="$RUN_ROOT/events/interaction_episodes.csv"
CLIPS="$RUN_ROOT/clips"
VLM_DATA="$RUN_ROOT/vlm_dataset_3fps"
PREDICTIONS="$RUN_ROOT/qwen3_vl_predictions.jsonl"
SUMMARY="$RUN_ROOT/episode_summary.csv"
STORAGE="$RUN_ROOT/storage_metrics.json"
LOG="$ROOT/outputs/logs/${SAFE_ID}_v4_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RUN_ROOT/detections" "$RUN_ROOT/events" "$CLIPS" "$VLM_DATA" "$ROOT/outputs/logs"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

on_error() {
  local code=$?
  echo "[실패] line=$1, exit_code=$code"
  echo "완료된 detector/VLM 결과는 보존됩니다. 같은 명령을 다시 실행하면 이어서 진행합니다."
  echo "로그: $LOG"
  exit "$code"
}
trap 'on_error $LINENO' ERR

echo "============================================================"
echo "ADL Episode Pipeline v4"
echo "영상: $VIDEO"
echo "Detector: 100DOH at 3 FPS"
echo "VLM: Qwen3-VL at 3 FPS, 10초 chunk, 1초 overlap"
echo "출력: $RUN_ROOT"
echo "============================================================"

echo "[1/9] 환경·영상·checkpoint 검증"
for item in "$MAIN_PY" "$DOH_PY" "$CHECKPOINT" /usr/bin/ffmpeg; do
  [[ -e "$item" ]] || { echo "[오류] 필수 파일 없음: $item"; exit 3; }
done
"$MAIN_PY" -c "import cv2,pandas,torch,transformers,accelerate; assert torch.cuda.is_available(); print('main GPU:',torch.cuda.get_device_name(0))"
(cd "$ROOT/tools/hand_object_detector" && "$DOH_PY" -c "import cv2,torch,torchvision; assert torch.cuda.is_available(); print('100DOH GPU:',torch.cuda.get_device_name(0))")
"$MAIN_PY" scripts/inspect_video.py --video "$VIDEO" --output-dir "$RUN_ROOT/video_check" --frames 5

echo "[2/9] 3 FPS 예상 표본 수 계산"
EXPECTED="$("$MAIN_PY" -c "import sys; from adl_event_logger.video import read_metadata; m=read_metadata(sys.argv[1]); step=max(1,round(m.fps/3.0)); print((m.frame_count+step-1)//step)" "$VIDEO")"
echo "예상 detector samples: $EXPECTED"

echo "[3/9] 100DOH를 원본 전체에 3 FPS로 실행"
CURRENT=0
[[ -f "$DETECTIONS" ]] && CURRENT="$(wc -l < "$DETECTIONS")"
if [[ "$CURRENT" -eq "$EXPECTED" ]]; then
  echo "완료된 100DOH 결과 재사용: $CURRENT lines"
else
  echo "100DOH 실행: current=$CURRENT, expected=$EXPECTED"
  (cd "$ROOT/tools/hand_object_detector" && "$DOH_PY" infer_json.py \
    --video "$VIDEO" --sample-fps 3.0 \
    --checkpoint "$CHECKPOINT" --device cuda \
    --hand-threshold 0.5 --object-threshold 0.5 \
    --output "$DETECTIONS")
fi
[[ "$(wc -l < "$DETECTIONS")" -eq "$EXPECTED" ]]

echo "[4/9] 시작~종료를 하나로 묶은 interaction episode 생성"
"$MAIN_PY" scripts/build_interaction_episodes_v4.py \
  --video "$VIDEO" --detections "$DETECTIONS" --output "$EPISODES" \
  --start-confirm-samples 2 --end-confirm-seconds 2 \
  --pre-seconds 2 --post-seconds 2 --minimum-active-seconds 0.5
EPISODE_COUNT="$($MAIN_PY -c "import pandas as pd; print(len(pd.read_csv('$EPISODES')))")"
[[ "$EPISODE_COUNT" -gt 0 ]] || { echo "[중단] 안정적인 손-물체 상호작용 episode가 검출되지 않았습니다."; exit 4; }

echo "[5/9] episode 전체 MP4 저장 — 원본 FPS 유지"
CLIPS_COMPLETE="$("$MAIN_PY" -c "import sys,pandas as pd; from pathlib import Path; t=pd.read_csv(sys.argv[1]); d=Path(sys.argv[2]); print(int(len(t)>0 and all((d/f'{x}.mp4').is_file() for x in t.event_id)))" "$EPISODES" "$CLIPS")"
if [[ "$CLIPS_COMPLETE" -eq 1 ]]; then
  echo "현재 episode ID의 MP4가 모두 있어 인코딩을 건너뜁니다."
else
  "$MAIN_PY" scripts/write_event_clips.py \
    --video "$VIDEO" --events "$EPISODES" --output-dir "$CLIPS" \
    --ffmpeg /usr/bin/ffmpeg --preset veryfast
fi

echo "[6/9] 원본 대비 저장 시간 계산"
"$MAIN_PY" scripts/evaluate_storage.py --video "$VIDEO" --events "$EPISODES" --output "$STORAGE"

echo "[7/9] episode를 3 FPS로 표본화 — 긴 동작은 10초 chunk로 분할"
"$MAIN_PY" scripts/build_adaptive_vlm_dataset_v4.py \
  --clips-dir "$CLIPS" --episodes "$EPISODES" --output-dir "$VLM_DATA" \
  --sample-fps 3 --chunk-seconds 10 --overlap-seconds 1 --resize-width 640

echo "[8/9] Qwen3-VL-4B로 각 episode 구간 JSON 라벨 생성"
"$MAIN_PY" scripts/run_qwen_episode_v4.py \
  --manifest "$VLM_DATA/manifest.csv" --output "$PREDICTIONS" \
  --model-id Qwen/Qwen3-VL-4B-Instruct --max-new-tokens 384

echo "[9/9] chunk 결과를 episode별로 정리하고 검증"
"$MAIN_PY" scripts/summarize_v4_episode_labels.py \
  --episodes "$EPISODES" --predictions "$PREDICTIONS" --output "$SUMMARY"
"$MAIN_PY" -c "import pandas as pd; e=pd.read_csv('$EPISODES'); s=pd.read_csv('$SUMMARY'); assert len(e)==len(s)==$EPISODE_COUNT; print(s[['episode_id','start_sec','end_sec','vlm_segment_count']].to_string(index=False))"

GT_CANDIDATE="$ROOT/data/manifests/epic/${VIDEO_ID}_gt.csv"
V3_CANDIDATE="$ROOT/outputs/events/${VIDEO_ID}_fusion_v2/fusion_v2_union.csv"
COMPARISON="$RUN_ROOT/v3_vs_v4_event_comparison.csv"
if [[ -f "$GT_CANDIDATE" && -f "$V3_CANDIDATE" ]]; then
  echo "[추가 평가] 동일 EPIC GT에서 v3 change evidence와 v4 episode 비교"
  "$MAIN_PY" scripts/compare_v3_v4_events.py \
    --gt "$GT_CANDIDATE" --v3-events "$V3_CANDIDATE" \
    --v4-events "$EPISODES" --video "$VIDEO" --output "$COMPARISON"
else
  echo "[추가 평가 건너뜀] 이 영상에 연결된 EPIC GT 또는 v3 event가 없습니다."
fi

echo "============================================================"
echo "v4 완료"
echo "episodes: $EPISODES"
echo "clips: $CLIPS"
echo "Qwen predictions: $PREDICTIONS"
echo "episode summary: $SUMMARY"
echo "storage metrics: $STORAGE"
[[ -f "$COMPARISON" ]] && echo "v3 vs v4 comparison: $COMPARISON"
echo "log: $LOG"
echo "============================================================"
