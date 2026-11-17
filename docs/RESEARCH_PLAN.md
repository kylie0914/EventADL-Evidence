# 연구 계획서: Offline Event-Based ADL Video Labeling

## 1. 목표

긴 1인칭 생활 영상에서 손–물체 상호작용 구간을 자동으로 찾고, 저장 효율과 행동 보존 성능을 비교하며, VLM이 각 구간의 행동과 물체를 구조화된 형식으로 설명하도록 구현한다.

## 2. 연구 질문

1. Full 또는 Periodic 방식보다 event 기반 선택이 저장량을 줄이면서 행동을 보존하는가?
2. 단순 hand/object presence보다 learned hand–object interaction이 조작 구간을 더 잘 구분하는가?
3. RTX 3090에서 실행 가능한 VLM 중 어떤 모델이 JSON 안정성, 속도, VRAM, 행동 라벨 성능의 균형이 좋은가?
4. 2차 시각 검증이 1차 VLM 행동 후보의 오탐을 줄이는가?

## 3. 기존 방법과 한계

- **Full:** 행동 누락은 적지만 저장량과 검토 비용이 줄지 않는다.
- **Periodic:** 구현은 쉽지만 행동 경계와 짧은 행동을 무시한다.
- **Motion:** 카메라 흔들림과 조명 변화에도 반응한다.
- **Hand/Object presence:** passive object가 많은 환경에서 너무 많은 영상을 선택한다.
- **Distance heuristic:** 해석은 쉽지만 실제 접촉을 학습한 detector가 아니다.

## 4. 제안 방법

```text
연속 영상
  -> 100DOH 3 FPS interaction detection
  -> 시작과 종료를 묶은 variable-length episode
  -> 앞뒤 context 추가
  -> episode 내부 3 FPS adaptive VLM sampling
  -> 1차 VLM action candidates
  -> 2차 visual verification
  -> 검증 결과와 정량 평가 파일 저장
```

## 5. 데이터와 비교 모델

- EPIC-KITCHENS selected untrimmed video와 공식 action interval annotations
- SmolVLM2-2.2B-Instruct
- Qwen3-VL-4B-Instruct
- InternVL3.5-4B-HF
- 향후 외부 검증: 무행동 시간이 포함된 개인 GoPro 영상. 영상 자체는 공개하지 않는다.

## 6. 평가

- Event Recall at 20/50/80% GT coverage
- Mean GT coverage
- Unique saved duration과 storage reduction
- Detector precision, recall, F1
- VLM JSON validity, verb/object/ADL score
- 평균 inference time과 peak allocated VRAM
- Verifier의 verified/rejected/uncertain 분포

## 7. 해석 원칙

저장 감소와 행동 보존을 함께 보고한다. 한 영상에서 선택한 threshold를 일반화된 성능으로 주장하지 않는다. VLM이 생성한 설명은 사람 검토 없이 임상 판단에 사용하지 않는다.

## 8. 후속 연구

- participant-held-out EPIC 영상 평가
- 무행동 구간이 긴 GoPro 외부 검증
- 짧은 행동의 후보 recall을 높이는 sampling 방법
- 서로 다른 VLM을 사용한 교차 검증
- human-reviewed detector와 action label audit 확대
