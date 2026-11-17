# 코드 구조 가이드

## 설계 원칙

이 프로젝트는 세 계층으로 나눈다.

1. `src/adl_event_logger/`: 여러 실험에서 재사용하는 핵심 로직
2. `scripts/`: 사용자가 터미널에서 실행하는 작은 진입점
3. `configs/`, `data/`, `outputs/`: 설정, 입력, 결과

스크립트끼리 함수를 복사하지 않는다. 예를 들어 영상 timestamp 계산은 항상
`video.py`, trigger score를 event 구간으로 바꾸는 처리는 항상 `events.py`를 사용한다.

## 데이터 흐름

```text
EPIC annotation CSV
  → annotations.py
  → split_dev.csv / split_test.csv / pilot_videos.csv

원본 MP4
  → video.py
  → motion / hand / object / active-object scores.csv
  → events.py
  → events.csv
  → write_event_clips.py
  → event MP4 files
  → annotate_clips_vlm.py
  → labels.jsonl
  → evaluation.py / plotting scripts
```

## 핵심 모듈

### `video.py`

- MP4 metadata 읽기
- frame index를 초 단위 timestamp로 변환
- 지정 FPS로 frame sampling
- VLM용 균등 프레임 추출

### `annotations.py`

- EPIC train/validation CSV 탐색
- 필수 열과 시간 구간 검증
- participant/video 요약
- participant가 겹치지 않는 dev/test split 생성

### `triggers.py`

- motion score
- bounding-box IoU와 중심 거리
- 손과 물체가 가까운 정도를 active-object heuristic으로 계산
- 여러 score의 가중합

`active_object_score()`는 학습된 접촉 검출기가 아니라 해석 가능한 기준선이다. 추후
100DOH 결과를 같은 score CSV 형식으로 넣어 교체한다.

### `events.py`

- score threshold 적용
- N회 연속 활성/비활성 hysteresis
- pre/post buffer 추가
- 가까운 event 병합

### `evaluation.py`

- GT event 중 저장 구간이 80% 이상 보존된 비율 계산
- 겹친 구간의 합집합을 사용해 중복 저장을 두 번 세지 않음

## 주요 실행 스크립트

- `smoke_test_smolvlm2_image.py`: 모델 다운로드와 단일 이미지 GPU 추론 확인
- `smoke_test_smolvlm2_video.py`: 로컬 MP4 decoding과 비디오 추론 확인
- `analyze_epic_annotations.py`: annotations만으로 데이터 규모 파악
- `make_epic_split.py`: participant-held-out pilot split 생성
- `inspect_video.py`: metadata와 균등 프레임 5장 확인
- `run_motion_trigger.py`: 가장 단순한 event trigger
- `run_detector_scores.py`: MediaPipe hand + TorchVision object + interaction heuristic
- `run_event_method.py`: Full/Periodic/Hand/Object/Active/Fusion의 공통 event 생성
- `write_event_clips.py`: event CSV를 실제 MP4로 저장
- `build_detector_audit.py`: 사람이 검토할 200~300프레임 표본 생성
- `annotate_clips_vlm.py`: event clip별 JSONL 라벨 생성
- `evaluate_events.py`: Event Recall과 GT coverage 계산
- `evaluate_storage.py`: Full recording 대비 저장량 감소율 계산
- `evaluate_detector_audit.py`: 손·물체·상호작용 detector Precision/Recall/F1
- `evaluate_vlm_labels.py`: VLM verb/object Accuracy와 Macro-F1
- `compare_methods.py`: 방법별 recall-storage trade-off 표와 그래프
- `plot_event_timeline.py`: GT와 저장 구간 시각화

## 표준 파일 계약

### Score CSV

```text
time_sec,frame_index,motion_score,hand_score,object_score,active_object_score
```

### Event CSV

```text
video_id,event_id,start_sec,end_sec,peak_score,method
```

### VLM JSONL

```json
{"event_id":"motion_0001","verb":"take","object":"cup","uncertain":false}
```

각 단계는 이 계약만 지키면 내부 detector나 VLM을 교체할 수 있다.

