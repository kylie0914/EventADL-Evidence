# 전체 실행 순서

모든 명령은 프로젝트 루트에서 실행한다. 한 단계가 성공한 뒤 다음 단계로 이동한다.
이 문서는 앞으로 실행할 계획이다. 실제로 실행된 명령과 변경 이력은
`docs/actual_execution_log.md`에 별도로 누적한다.

```bash
cd /path/to/egocentric_CAM_VLM
conda activate adl-event-logger
```

## 1. 환경 확인

```bash
python scripts/check_system.py
```

## 2. VLM 공통 패키지 확인

```bash
python -c "import torch, transformers, accelerate, decord, av; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, accelerate.__version__, decord.__version__, av.__version__)"
```

## 3. SmolVLM2 단일 이미지 smoke test

공식 예제 이미지를 직접 받는다.

```bash
mkdir -p data/smoke && curl -L https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg -o data/smoke/bee.jpg
```

처음 실행할 때 모델 약 5GB가 Hugging Face cache에 다운로드된다.

```bash
python scripts/smoke_test_smolvlm2_image.py --image data/smoke/bee.jpg
```

## 4. SmolVLM2 비디오 smoke test

직접 촬영 영상이 아직 없으면 Hugging Face의 작은 공개 `archery.mp4` 한 파일만
다운로드한다. 이것은 EPIC-KITCHENS 데이터셋 다운로드가 아니라 decoder 확인용이다.

```bash
hf download nateraw/video-demo archery.mp4 --repo-type dataset --local-dir data/smoke
```

```bash
mv data/smoke/archery.mp4 data/smoke/test.mp4
```

```bash
ls -lh data/smoke/test.mp4
```

Transformers 5.x는 비디오 decoding에 TorchCodec을 사용한다. PyTorch/CUDA와 같은
공식 wheel 저장소에서 설치한다.

```bash
python -m pip install torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cu130
```

```bash
python -c "import torchcodec; print('torchcodec:', torchcodec.__version__)"
```

그 뒤 비디오 smoke test를 실행한다.

```bash
python scripts/smoke_test_smolvlm2_video.py --video data/smoke/test.mp4
```

## 5. EPIC-KITCHENS annotations만 다운로드

이 명령은 annotation Git 저장소만 받으며 원본 영상은 받지 않는다.

```bash
mkdir -p data/annotations && git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git data/annotations/epic-kitchens-100-annotations
```

## 6. Annotation 분석

프로젝트 패키지를 editable mode로 연결한다.

```bash
python -m pip install -e .
```

```bash
python scripts/analyze_epic_annotations.py --annotations-dir data/annotations/epic-kitchens-100-annotations
```

분석 결과 확인:

```bash
cat data/manifests/epic/annotation_summary.json
```

## 7. 참가자별 dev/test split과 video ID 선정

EPIC 공식 train/validation에 같은 참가자가 있을 수 있으므로, 코드는 dev로 선택된
참가자를 validation 후보에서 제외한 뒤 test 참가자를 고른다.

```bash
python scripts/make_epic_split.py --annotations-dir data/annotations/epic-kitchens-100-annotations --dev-participants 2 --test-participants 1
```

선정 목록 확인:

```bash
column -s, -t data/manifests/epic/pilot_videos.csv | less -S
```

## 8. 선택된 원본 영상만 다운로드

먼저 공식 downloader를 받는다.

```bash
git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git tools/epic-kitchens-download-scripts
```

다운로더 옵션은 저장소 버전에 따라 달라질 수 있으므로 먼저 확인한다.

```bash
python tools/epic-kitchens-download-scripts/epic_downloader.py --help
```

그 출력과 `pilot_videos.csv`를 확인한 뒤 participant/video subset 명령을 확정한다. 전체
1TB를 받는 명령은 실행하지 않는다.

현재 pilot의 첫 dev 영상 `P04_01` 하나만 받는 확정 명령은 다음과 같다. 진행률과 MD5
검사를 지원하는 video 전용 downloader를 사용한다.

```bash
python tools/epic-kitchens-download-scripts/epic_video_downloader_verify.py \
  --specific-videos P04_01 \
  --output-path data/raw \
  --epic55-splits tools/epic-kitchens-download-scripts/data/epic_55_splits.csv \
  --epic100-splits tools/epic-kitchens-download-scripts/data/epic_100_splits.csv \
  --md5 tools/epic-kitchens-download-scripts/data/md5.csv \
  --errata tools/epic-kitchens-download-scripts/data/errata.csv \
  --max-retries 3 \
  --report outputs/downloads/P04_01_report.csv
```

metadata 경로를 명시하는 이유는 downloader의 기본 상대 경로 `data/...`가 프로젝트의
`data/`를 가리키기 때문이다. 실제 metadata는 `tools/epic-kitchens-download-scripts/data/`
아래에 있다.

공식 downloader는 다음 중 데이터 버전에 맞는 중첩 경로에 파일을 저장한다.

```bash
find data/raw/EPIC-KITCHENS -type f -name 'P04_01.MP4' -ls
```

다운로드가 확인되면 이후 명령을 단순하게 유지하기 위해 표준 위치에 symbolic link를
만든다. 원본을 복사하지 않으므로 디스크를 두 배로 사용하지 않는다.

```bash
ln -s "$(pwd)/data/raw/EPIC-KITCHENS/P04/videos/P04_01.MP4" data/raw/epic/P04_01.MP4
```

## 9. 영상 metadata와 timestamp 검증

다운로드한 첫 pilot 파일은 `data/raw/epic/P04_01.MP4`로 연결한다.

```bash
python scripts/inspect_video.py --video data/raw/epic/PXX_XX.MP4
```

## 10. GT event와 영상 연결

```bash
python scripts/validate_gt_against_video.py --video data/raw/epic/PXX_XX.MP4 --split-csv data/manifests/epic/split_dev.csv --output data/manifests/epic/PXX_XX_gt.csv
```

## 11. Full·Periodic baseline

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method full --output outputs/events/PXX_XX_full.csv
```

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method periodic --period-seconds 10 --clip-seconds 8 --output outputs/events/PXX_XX_periodic.csv
```

## 12. Motion trigger

```bash
python scripts/run_motion_trigger.py --video data/raw/epic/PXX_XX.MP4 --output-dir outputs/motion/PXX_XX
```

## 13~15. Hand/Object/Active-object detector

추가 패키지:

```bash
python -m pip install mediapipe matplotlib seaborn scikit-learn
```

첫 실행 때 TorchVision object detector weight가 다운로드된다.

```bash
python scripts/run_detector_scores.py --video data/raw/epic/PXX_XX.MP4 --output outputs/detectors/PXX_XX_scores.csv
```

각 method의 event 생성:

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method hand --scores outputs/detectors/PXX_XX_scores.csv --output outputs/events/PXX_XX_hand.csv
```

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method object --scores outputs/detectors/PXX_XX_scores.csv --output outputs/events/PXX_XX_object.csv
```

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method active_object --scores outputs/detectors/PXX_XX_scores.csv --output outputs/events/PXX_XX_active_object.csv
```

## 16. Detector audit 200~300프레임

```bash
python scripts/build_detector_audit.py --video data/raw/epic/PXX_XX.MP4 --scores outputs/detectors/PXX_XX_scores.csv --count 250 --output-dir data/audit/PXX_XX
```

`data/audit/PXX_XX/audit_labels.csv`의 `gt_*` 열을 사람이 채운다.

라벨을 모두 채운 뒤 detector 자체 성능을 계산한다.

```bash
python scripts/evaluate_detector_audit.py --audit-csv data/audit/PXX_XX/audit_labels.csv --output outputs/metrics/PXX_XX_detector_audit.json
```

## 17. Fusion event와 clip 저장

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 --method fusion --scores outputs/detectors/PXX_XX_scores.csv --motion-scores outputs/motion/PXX_XX/scores.csv --output outputs/events/PXX_XX_fusion.csv
```

```bash
python scripts/write_event_clips.py --video data/raw/epic/PXX_XX.MP4 --events outputs/events/PXX_XX_fusion.csv --output-dir outputs/clips/PXX_XX_fusion
```

## 18. VLM JSON 라벨 생성

```bash
python scripts/annotate_clips_vlm.py --clips-dir outputs/clips/PXX_XX_fusion --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct --output outputs/labels/PXX_XX_smolvlm2.jsonl
```

## 19. Qwen3-VL과 InternVL3.5 비교

Qwen3-VL:

```bash
python scripts/annotate_clips_vlm.py --clips-dir outputs/clips/PXX_XX_fusion --model-id Qwen/Qwen3-VL-4B-Instruct --output outputs/labels/PXX_XX_qwen3vl.jsonl
```

InternVL은 모델의 custom API 호환성을 smoke test한 뒤 같은 JSONL 계약으로 adapter를
고정한다. 오류가 나면 해당 traceback을 공유하고 adapter만 수정한다.

## 20. Event Recall 평가

```bash
python scripts/evaluate_events.py --gt data/manifests/epic/PXX_XX_gt.csv --predictions outputs/events/PXX_XX_fusion.csv --output outputs/metrics/PXX_XX_fusion.json
```

저장량 감소율도 별도로 계산한다.

```bash
python scripts/evaluate_storage.py --video data/raw/epic/PXX_XX.MP4 --events outputs/events/PXX_XX_fusion.csv --output outputs/metrics/PXX_XX_fusion_storage.json
```

VLM F1은 **trigger clip이 아니라 GT event clip**을 만든 뒤 비교해야 trigger 누락과
VLM 오분류가 섞이지 않는다. GT clip을 생성하고 각 모델을 실행한 뒤 평가한다.

```bash
python scripts/write_event_clips.py --video data/raw/epic/PXX_XX.MP4 --events data/manifests/epic/PXX_XX_gt.csv --output-dir outputs/clips/PXX_XX_gt
```

```bash
python scripts/annotate_clips_vlm.py --clips-dir outputs/clips/PXX_XX_gt --model-id HuggingFaceTB/SmolVLM2-2.2B-Instruct --output outputs/labels/PXX_XX_gt_smolvlm2.jsonl
```

```bash
python scripts/evaluate_vlm_labels.py --gt data/manifests/epic/PXX_XX_gt.csv --predictions outputs/labels/PXX_XX_gt_smolvlm2.jsonl --output outputs/metrics/PXX_XX_smolvlm2_vlm.json
```

## 21. 타임라인·실패 사례·README

```bash
python scripts/plot_event_timeline.py --gt data/manifests/epic/PXX_XX_gt.csv --predictions outputs/events/PXX_XX_fusion.csv --video-id PXX_XX --output outputs/figures/PXX_XX_timeline.png
```

Full/Periodic/Motion/Hand/Object/Active/Fusion 모두 20번 평가를 반복한 뒤 비교한다.

```bash
python scripts/compare_methods.py --metrics-dir outputs/metrics --output-csv outputs/tables/method_comparison.csv --output-figure outputs/figures/recall_storage_tradeoff.png
```

각 명령의 실제 출력과 오류를 `docs/experiment_log.md`에 누적한다.

