#!/usr/bin/env bash
# P04_01 한 영상으로 자동 실행 가능한 파이프라인을 밤새 순서대로 수행한다.
#
# 안전 원칙:
# - 기존 .part 파일은 삭제하지 않고 공식 downloader가 이어받게 한다.
# - 한 단계라도 실패하면 즉시 멈춘다(set -Eeuo pipefail).
# - 터미널 출력은 화면과 로그 파일에 동시에 저장한다.
# - 사람이 직접 채워야 하는 detector audit 평가는 실행하지 않는다.

set -Eeuo pipefail

# 이 파일이 어느 경로에 clone되어도 repository root를 자동으로 찾는다.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_ID="P04_01"
DOWNLOAD_ROOT="$PROJECT_ROOT/data/raw"
DOWNLOADED_VIDEO="$DOWNLOAD_ROOT/EPIC-KITCHENS/P04/videos/$VIDEO_ID.MP4"
STANDARD_VIDEO="$DOWNLOAD_ROOT/epic/$VIDEO_ID.MP4"
GT_CSV="$PROJECT_ROOT/data/manifests/epic/${VIDEO_ID}_gt.csv"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_ROOT/outputs/logs/${VIDEO_ID}_overnight_${RUN_STAMP}.log"
EXPECTED_MD5="b2d35b957da99abd9f37a14cc39184a8"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs outputs/downloads outputs/cache/matplotlib data/raw/epic
export MPLCONFIGDIR="$PROJECT_ROOT/outputs/cache/matplotlib"

# 이후 모든 stdout/stderr를 로그에도 복사한다.
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
  local exit_code=$?
  echo "[실패] line=$1, exit_code=$exit_code"
  echo "[로그] $LOG_FILE"
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

echo "============================================================"
echo "P04_01 overnight pipeline 시작: $(date --iso-8601=seconds)"
echo "프로젝트: $PROJECT_ROOT"
echo "Conda 환경: ${CONDA_DEFAULT_ENV:-활성화되지 않음}"
echo "기존 .part 파일은 있으면 이어받는다."
echo "============================================================"

if [[ "${CONDA_DEFAULT_ENV:-}" != "adl-event-logger" ]]; then
  echo "[오류] 먼저 conda activate adl-event-logger 를 실행하세요."
  exit 2
fi

echo "[1/15] 프로젝트 package 연결"
if python -c "import adl_event_logger" 2>/dev/null; then
  echo "adl_event_logger가 이미 import되므로 editable install을 건너뜁니다."
else
  # 빌드 격리 환경이 setuptools를 다시 내려받지 않도록 현재 Conda 환경의
  # setuptools를 사용한다.
  python -m pip install --no-build-isolation -e .
fi

echo "[2/15] Detector와 평가용 package 설치/확인"
if python -c "import mediapipe as mp, matplotlib, seaborn, sklearn; assert hasattr(mp, 'solutions')" 2>/dev/null; then
  echo "필요 package와 MediaPipe Solutions API가 이미 준비되어 있습니다."
else
  # MediaPipe 0.10.35는 legacy mp.solutions API를 노출하지 않으므로 현재 hand
  # baseline 코드와 호환되는 0.10.21을 고정한다.
  python -m pip install --upgrade --force-reinstall \
    "mediapipe==0.10.21" \
    "matplotlib>=3.9" \
    "seaborn>=0.13" \
    "scikit-learn>=1.5"
fi
python -c "import cv2, mediapipe as mp, sklearn; print('cv2=', cv2.__version__); print('mediapipe=', mp.__version__, 'solutions=', hasattr(mp, 'solutions')); print('sklearn=', sklearn.__version__)"

echo "[3/15] P04_01 한 파일 다운로드 또는 .part 이어받기"
if [[ -s "$DOWNLOADED_VIDEO" ]] && echo "$EXPECTED_MD5  $DOWNLOADED_VIDEO" | md5sum --check --status; then
  echo "이미 다운로드되어 MD5 검증된 파일을 사용합니다: $DOWNLOADED_VIDEO"
else
  python tools/epic-kitchens-download-scripts/epic_video_downloader_verify.py \
    --specific-videos "$VIDEO_ID" \
    --output-path "$DOWNLOAD_ROOT" \
    --epic55-splits tools/epic-kitchens-download-scripts/data/epic_55_splits.csv \
    --epic100-splits tools/epic-kitchens-download-scripts/data/epic_100_splits.csv \
    --md5 tools/epic-kitchens-download-scripts/data/md5.csv \
    --errata tools/epic-kitchens-download-scripts/data/errata.csv \
    --max-retries 3 \
    --report outputs/downloads/P04_01_report.csv
fi

if [[ ! -s "$DOWNLOADED_VIDEO" ]]; then
  echo "[오류] 다운로드 완료 파일이 없습니다: $DOWNLOADED_VIDEO"
  exit 3
fi

echo "[4/15] 코드가 사용하는 표준 경로에 symbolic link 생성"
ln -sfn "$DOWNLOADED_VIDEO" "$STANDARD_VIDEO"
ls -lh "$DOWNLOADED_VIDEO" "$STANDARD_VIDEO"

echo "[5/15] 영상 metadata와 표본 frame 확인"
python scripts/inspect_video.py \
  --video "$STANDARD_VIDEO" \
  --output-dir "outputs/video_check/$VIDEO_ID"

echo "[6/15] EPIC GT event와 원본 영상 timestamp 연결"
python scripts/validate_gt_against_video.py \
  --video "$STANDARD_VIDEO" \
  --split-csv data/manifests/epic/split_dev.csv \
  --output "$GT_CSV"

echo "[7/15] Full·Periodic baseline event 생성"
python scripts/run_event_method.py \
  --video "$STANDARD_VIDEO" --method full \
  --output "outputs/events/${VIDEO_ID}_full.csv"
python scripts/run_event_method.py \
  --video "$STANDARD_VIDEO" --method periodic \
  --period-seconds 10 --clip-seconds 8 \
  --output "outputs/events/${VIDEO_ID}_periodic.csv"

echo "[8/15] Motion score와 event 생성"
python scripts/run_motion_trigger.py \
  --video "$STANDARD_VIDEO" \
  --output-dir "outputs/motion/$VIDEO_ID"

echo "[9/15] Hand·Object·거리 기반 Active-object score 생성"
python scripts/run_detector_scores.py \
  --video "$STANDARD_VIDEO" \
  --output "outputs/detectors/${VIDEO_ID}_scores.csv"

echo "[10/15] Hand·Object·Active-object·Fusion event 생성"
for method in hand object active_object; do
  python scripts/run_event_method.py \
    --video "$STANDARD_VIDEO" \
    --method "$method" \
    --scores "outputs/detectors/${VIDEO_ID}_scores.csv" \
    --output "outputs/events/${VIDEO_ID}_${method}.csv"
done
python scripts/run_event_method.py \
  --video "$STANDARD_VIDEO" \
  --method fusion \
  --scores "outputs/detectors/${VIDEO_ID}_scores.csv" \
  --motion-scores "outputs/motion/${VIDEO_ID}/scores.csv" \
  --output "outputs/events/${VIDEO_ID}_fusion.csv"

echo "[11/15] 사람이 나중에 라벨할 detector audit 250 frame 생성"
python scripts/build_detector_audit.py \
  --video "$STANDARD_VIDEO" \
  --scores "outputs/detectors/${VIDEO_ID}_scores.csv" \
  --count 250 \
  --output-dir "data/audit/$VIDEO_ID"

echo "[12/15] Fusion event를 MP4 clip으로 저장"
python scripts/write_event_clips.py \
  --video "$STANDARD_VIDEO" \
  --events "outputs/events/${VIDEO_ID}_fusion.csv" \
  --output-dir "outputs/clips/${VIDEO_ID}_fusion"

echo "[13/15] Fusion clip을 SmolVLM2 JSONL로 설명"
python scripts/annotate_clips_vlm.py \
  --clips-dir "outputs/clips/${VIDEO_ID}_fusion" \
  --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --output "outputs/labels/${VIDEO_ID}_smolvlm2.jsonl"

echo "[14/15] 각 방법의 Event Recall과 저장 시간 감소율 계산"
for method in full periodic hand object active_object fusion; do
  python scripts/evaluate_events.py \
    --gt "$GT_CSV" \
    --predictions "outputs/events/${VIDEO_ID}_${method}.csv" \
    --output "outputs/metrics/${VIDEO_ID}_${method}_event.json"
  python scripts/evaluate_storage.py \
    --video "$STANDARD_VIDEO" \
    --events "outputs/events/${VIDEO_ID}_${method}.csv" \
    --output "outputs/metrics/${VIDEO_ID}_${method}_storage.json"
done

# Motion은 run_motion_trigger.py가 별도 폴더에 events.csv를 쓰므로 따로 평가한다.
python scripts/evaluate_events.py \
  --gt "$GT_CSV" \
  --predictions "outputs/motion/${VIDEO_ID}/events.csv" \
  --output "outputs/metrics/${VIDEO_ID}_motion_event.json"
python scripts/evaluate_storage.py \
  --video "$STANDARD_VIDEO" \
  --events "outputs/motion/${VIDEO_ID}/events.csv" \
  --output "outputs/metrics/${VIDEO_ID}_motion_storage.json"

echo "[15/15] 타임라인과 방법 비교표 생성"
python scripts/plot_event_timeline.py \
  --gt "$GT_CSV" \
  --predictions "outputs/events/${VIDEO_ID}_fusion.csv" \
  --video-id "$VIDEO_ID" \
  --output "outputs/figures/${VIDEO_ID}_timeline.png"
python scripts/compare_methods.py \
  --metrics-dir outputs/metrics \
  --output-csv outputs/tables/method_comparison.csv \
  --output-figure outputs/figures/recall_storage_tradeoff.png

echo "============================================================"
echo "자동 단계 완료: $(date --iso-8601=seconds)"
echo "로그: $LOG_FILE"
echo "다음 수작업: data/audit/$VIDEO_ID/audit_labels.csv의 gt_* 열 채우기"
echo "주의: 현재 active-object는 거리 기반 v0이며 100DOH 최종 조건이 아니다."
echo "============================================================"
