# Experiment log

> 이 문서는 초기 기록 보존용이다. 이후 실제 진행 기록의 단일 기준본은
> `docs/actual_execution_log.md`다.

## Collaboration rule

- The user directly runs all environment setup, downloads, and experiment commands.
- The assistant writes required project code, then explains the implementation, inputs,
  outputs, execution command, and verification criteria.
- Large datasets and models are downloaded incrementally only after the user reviews the
  exact command and intended subset.

## Environment setup

- Date: 2026-11-15 (America/New_York)
- Workspace: `$PROJECT_ROOT`
- Miniconda: already installed at `$HOME/miniconda3`
- Conda version observed before setup: `25.3.1`
- System FFmpeg observed before setup: `6.1.1`
- Environment strategy: use the named environment `adl-event-logger`, activated with
  `conda activate adl-event-logger`.
- Base package verification: OpenCV 4.14.0; imports for pandas, NumPy, PyYAML, and
  Pydantic succeeded.
- GPU verified in the user's terminal: NVIDIA GeForce RTX 3090, 24576 MiB VRAM.
- NVIDIA driver: 580.159.04; driver-reported CUDA compatibility: 13.0.
- PyTorch CUDA GPU computation verified: a 4000×4000 matrix multiplication completed on
  `cuda:0`; peak allocated VRAM reported by the test was 132.1 MiB.
- Next step: install and verify the common Transformers/video inference dependencies.
- Full command history: see `docs/setup_log.md`.

# 실험 실행 기록

## SmolVLM2 비디오 smoke test

- 상태: 성공
- 모델: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- 입력: `data/smoke/test.mp4` (`nateraw/video-demo/archery.mp4`)
- 출력: `Two women are practicing archery on a grassy field.`
- 추론 시간: 0.9085초
- 최대 VRAM: 4585.855MiB
- 결론: RTX 3090에서 비디오 decoding과 2.2B VLM 추론이 정상 동작함

