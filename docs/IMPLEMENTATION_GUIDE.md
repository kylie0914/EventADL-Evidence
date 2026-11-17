# 실제 구현 계획서: Offline Pipeline

## 1. 환경 분리

Main VLM과 평가 코드는 Python 3.11 `adl-event-logger` 환경에서 실행한다. Legacy 100DOH는 Python 3.8 `handobj-100doh` 환경에서 실행한다. 두 환경을 섞지 않는다.

```bash
conda env create -n adl-event-logger -f environment.yml
conda activate adl-event-logger
python -m pip install -e .
python scripts/check_system.py
```

100DOH repository와 checkpoint는 `tools/` 아래에 사용자가 별도로 설치하며 Git에 포함하지 않는다.

## 2. 영상과 GT 검증

**무엇:** FPS, frame 수, 길이, 표본 frame, annotation timestamp를 검사한다.

**왜:** 시간축이 다르면 event recall과 coverage 계산이 모두 잘못된다.

**어떻게:** `inspect_video.py`와 `validate_gt_against_video.py`를 실행한다.

## 3. 100DOH frame inference

**무엇:** hand box, contact state, contacted target-object를 3 FPS로 검출한다.

**왜:** 보이는 물체와 실제 조작 물체를 구분하기 위해서다.

**출력:** frame별 JSONL. 각 행은 timestamp와 detector 결과를 포함한다.

## 4. Interaction episode

**무엇:** 연속 interaction을 하나의 variable-length episode로 묶는다.

**왜:** 고정 길이 clip이 하나의 행동을 여러 조각으로 나누는 문제를 줄인다.

**어떻게:** 두 표본 연속 활성에서 시작하고, 2초 연속 비활성에서 종료하며 앞뒤 context를 추가한다.

```bash
bash scripts/run_adl_episode_v4.sh /absolute/path/to/video.mp4
```

## 5. Adaptive VLM input

**무엇:** episode 내부를 3 FPS로 표본화한다.

**왜:** 길이에 관계없이 8 frame만 보는 방식은 긴 행동 중간을 놓친다.

**어떻게:** 긴 episode는 VLM 입력에서만 10초 chunk와 1초 overlap으로 나눈다. 저장된 episode MP4는 전체 길이를 유지한다.

## 6. VLM comparison

같은 clip과 prompt로 SmolVLM2, Qwen3-VL, InternVL을 실행한다. JSON validity, latency, peak VRAM, normalized label score를 기록한다. 모델별 패키지와 output JSONL을 분리한다.

## 7. Two-pass action verification

1차 Qwen3-VL은 최대 네 개의 visible action 후보와 상대 시간을 생성한다. `verify_vlm_actions_v5.py`는 후보 시간 주변 frame을 다시 보고 `verified`, `rejected`, `uncertain`을 출력한다.

cut/slice/chop은 칼날이 재료에 닿아 통과하는 연속 frame 증거가 있어야 한다. 봉지를 누르거나 여는 행동을 cutting으로 통과시키지 않는다.

## 8. 평가 파일

- events CSV: 시작, 종료, method, peak score
- detector JSONL: frame timestamp, hands, contacted objects
- candidate JSONL: verb, object, description, candidate time
- verifier JSONL: verdict, evidence, corrected label, confidence
- comparison CSV: recall, coverage, saved duration, storage reduction

## 9. 재실행 정책

긴 detector와 VLM 작업은 JSONL을 행 단위로 flush한다. 같은 ID의 완료 결과는 재사용하고, candidate verb/object가 바뀌면 verifier를 다시 실행한다.

## 10. 공개 범위

이 저장소는 offline 연구 코드까지만 공개한다. 사용자 영상, model weight, upload server, web application, authentication, account integration, API key와 내부 운영 로그는 포함하지 않는다.
