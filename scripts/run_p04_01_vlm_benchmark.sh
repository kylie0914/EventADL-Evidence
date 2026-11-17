#!/usr/bin/env bash
# Fusion v2 Union 43개 clip을 동일 8-frame 입력으로 3개 VLM에 비교한다.

set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ADL_MAIN_PY:-$HOME/miniconda3/envs/adl-event-logger/bin/python}"
CLIPS="$ROOT/outputs/clips/P04_01_fusion_v2_union"
EVENTS="$ROOT/outputs/events/P04_01_fusion_v2/fusion_v2_union.csv"
EPIC_GT="$ROOT/data/manifests/epic/P04_01_gt.csv"
BENCHMARK="$ROOT/outputs/vlm_benchmark/P04_01_fusion_v2_union"
DATASET="$BENCHMARK/dataset"
PREDICTIONS="$BENCHMARK/predictions"
REPORTS="$BENCHMARK/reports"
LOG="$ROOT/outputs/logs/P04_01_vlm_benchmark_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$DATASET" "$PREDICTIONS" "$REPORTS" "$ROOT/outputs/logs"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1

on_error() {
  local code=$?
  echo "[실패] line=$1, exit_code=$code"
  echo "모델 다운로드와 완료된 JSONL은 보존됩니다. 수정 후 같은 명령으로 이어갑니다."
  echo "로그: $LOG"
  exit "$code"
}
trap 'on_error $LINENO' ERR

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "P04_01 Fusion v2 VLM benchmark: $(date --iso-8601=seconds)"
echo "모델은 한 번에 하나만 GPU에 올리며 완료 행은 재실행 시 건너뜁니다."
echo "============================================================"

echo "[1/7] 입력·GPU·공통 package 확인"
for file in "$PYTHON" "$EVENTS" "$EPIC_GT"; do
  [[ -e "$file" ]] || { echo "[오류] 파일 없음: $file"; exit 2; }
done
[[ "$(find "$CLIPS" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 43 ]] || {
  echo "[오류] Fusion v2 MP4 clip이 43개가 아닙니다: $CLIPS"; exit 3;
}
"$PYTHON" -c "import torch, transformers, accelerate, cv2, pandas, sklearn; print('torch',torch.__version__,'transformers',transformers.__version__); assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"
if ! "$PYTHON" -c "import timm" 2>/dev/null; then
  echo "InternVL 호환 package timm을 설치합니다."
  "$PYTHON" -m pip install "timm>=1.0,<2"
fi

echo "[2/7] 43개 clip과 EPIC GT 연결 + 공통 8-frame 입력 생성"
"$PYTHON" scripts/build_vlm_clip_dataset.py \
  --clips-dir "$CLIPS" --events "$EVENTS" --epic-gt "$EPIC_GT" \
  --output-dir "$DATASET" --frames 8 --resize-width 640
[[ "$(($(wc -l < "$DATASET/clip_gt.csv") - 1))" -eq 43 ]]

run_model() {
  local key="$1"
  local model_id="$2"
  local output="$PREDICTIONS/${key}.jsonl"
  local completed=0
  [[ -f "$output" ]] && completed="$(wc -l < "$output")"
  echo "------------------------------------------------------------"
  echo "모델=$key | Hugging Face=$model_id | 기존 완료=$completed/43"
  echo "첫 실행에서는 model weight 다운로드 시간이 추가됩니다."
  if [[ "$completed" -eq 43 ]]; then
    echo "$key 결과가 이미 43행이므로 model loading과 추론을 건너뜁니다."
    return
  fi
  "$PYTHON" scripts/run_vlm_clip_model.py \
    --manifest "$DATASET/clip_gt.csv" \
    --vocabulary "$DATASET/vocabulary.json" \
    --model-id "$model_id" --model-key "$key" \
    --output "$output" --max-new-tokens 128
  [[ "$(wc -l < "$output")" -eq 43 ]] || {
    echo "[오류] $key 결과가 43행이 아닙니다."; exit 4;
  }
}

echo "[3/7] SmolVLM2-2.2B 추론"
run_model "smolvlm2_2b" "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

echo "[4/7] Qwen3-VL-4B 추론"
run_model "qwen3_vl_4b" "Qwen/Qwen3-VL-4B-Instruct"

echo "[5/7] InternVL3.5-4B 표준 Transformers checkpoint 추론"
run_model "internvl3_5_4b" "OpenGVLab/InternVL3_5-4B-HF"

echo "[6/7] Verb·Object·ADL Macro-F1와 비용 비교"
"$PYTHON" scripts/evaluate_vlm_benchmark.py \
  --gt "$DATASET/clip_gt.csv" --predictions-dir "$PREDICTIONS" \
  --output-dir "$REPORTS"

echo "[7/7] 최종 산출물 검증"
"$PYTHON" -c "import pandas as pd; p='$REPORTS/model_comparison.csv'; t=pd.read_csv(p); assert len(t)==3; assert (t.clips==43).all(); print(t.to_string(index=False))"

echo "============================================================"
echo "VLM 3종 비교 완료: $(date --iso-8601=seconds)"
echo "비교표: $REPORTS/model_comparison.csv"
echo "클립별 결과: $REPORTS/*_per_clip.csv"
echo "원본 예측: $PREDICTIONS/*.jsonl"
echo "GT와 공통 프레임: $DATASET"
echo "로그: $LOG"
echo "============================================================"
