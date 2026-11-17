# 최종 구현 보고서: Offline Event-Based ADL Pipeline

## 1. 구현 결과

연속 1인칭 영상에서 100DOH interaction을 검출하고 variable-length episode를 생성한 뒤, 여러 VLM으로 행동 JSON을 만들고 정량 평가하는 offline pipeline을 구현했다.

## 2. 구현 변화

- **v0:** MediaPipe hand, Faster R-CNN object, 거리 heuristic으로 전체 연결을 확인했다.
- **v1:** 100DOH learned interaction detector를 별도 환경에 설치했다.
- **v2/v3:** lifecycle, object change, contact-state change, motion 조건으로 짧은 evidence clip을 만들었다.
- **v4:** interaction 시작부터 종료까지 완전한 episode를 저장하고 3 FPS adaptive VLM input을 적용했다.
- **Verifier:** 실제 cutting 오탐을 발견한 뒤 action candidate별 2차 시각 검증을 추가했다.

## 3. Temporal selection 결과

| Method | Recall@80% | Mean GT coverage | Storage reduction |
| --- | ---: | ---: | ---: |
| v3 change evidence | 35.50% | 43.41% | 62.76% |
| v4 complete episode | 83.59% | 85.97% | 12.60% |

v3는 저장 효율이 높지만 행동 전체를 잃었다. v4는 행동 보존을 크게 높였지만 action-dense kitchen video에서 저장 감소가 작았다.

## 4. VLM 비교 결과

43개 v3 evidence clips의 pilot에서 Qwen3-VL-4B는 JSON valid 100%, 평균 1.12초, peak VRAM 약 8.8 GiB를 기록했다. 이 protocol에서는 SmolVLM2와 InternVL보다 정확도와 속도의 균형이 좋았다.

Exact label score는 낮았다. 한 clip에 여러 GT action이 겹치지만 primary GT 하나만 정답으로 사용했고, VLM 용어와 EPIC taxonomy의 문자열 차이도 모두 오답으로 계산했기 때문이다.

## 5. Verifier 회귀검사

두 문제 episode에서 33 candidates를 검사해 18개를 verified, 15개를 rejected로 판정했다. 잘못된 cutting 후보는 사라졌다. 실제 짧은 slice는 첫 candidate generation에서 누락되어 남았다.

따라서 verifier는 오탐 감소에는 도움이 되지만 후보 누락을 복구하지 않는다.

## 6. 결론

하나의 최적 방법보다 목적에 따른 tradeoff가 중요했다. 저장 효율이 우선이면 change evidence가 유리하고, 행동 전체 보존이 우선이면 complete episode가 유리하다. VLM 결과는 검증 단계를 거쳐야 하지만 검증만으로 candidate recall 문제를 해결할 수 없다.

## 7. 한계

- P04_01 한 영상 중심의 development result
- participant-held-out 평가 미완료
- 임상적 손 기능 또는 stroke assessment 검증 없음
- 개인 GoPro 영상 외부 검증 미완료
- 같은 VLM을 후보 생성과 검증에 사용할 때 correlated error 가능

이 결과는 연구용 구현과 실패 분석이며 진단 성능을 주장하지 않는다.
