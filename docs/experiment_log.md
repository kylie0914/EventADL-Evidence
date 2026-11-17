# Experiment Log

This compact public log records only reproducible, non-sensitive milestones. Detailed private command history, raw video, generated frames, and model caches are not published.

## Collaboration and reproducibility rules

- Environment setup, downloads, and experiment commands are run locally.
- Large datasets and models are downloaded incrementally after selecting the required subset.
- Every reported result should include the model ID, input scope, command or script, output path, and verification criterion.
- Tokens, credentials, private media, and machine-specific absolute paths must never be recorded here.

## Environment setup

- Date: 2026-11-15 (America/New_York)
- Main environment: `adl-event-logger`
- Conda: 25.3.1
- FFmpeg: 6.1.1
- OpenCV: 4.14.0
- GPU: NVIDIA GeForce RTX 3090, 24576 MiB VRAM
- NVIDIA driver: 580.159.04
- Driver-reported CUDA compatibility: 13.0
- PyTorch GPU test: 4000 x 4000 matrix multiplication on `cuda:0`
- Test peak allocation: 132.1 MiB

See `docs/setup_log.md` for installation commands and resolved errors.

## SmolVLM2 video smoke test

- Status: passed
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Input: `data/smoke/test.mp4` from the public `nateraw/video-demo/archery.mp4` sample
- Output: `Two women are practicing archery on a grassy field.`
- Inference time: 0.9085 seconds
- Peak allocated VRAM: 4585.855 MiB
- Conclusion: local video decoding and 2.2B VLM inference worked on the RTX 3090.

## EPIC-KITCHENS pilot

- Video: `P04_01`
- Duration: 1154.7726 seconds
- GT action intervals: 262
- Purpose: development pilot for timestamp validation, trigger comparison, VLM benchmarking, and failure analysis
- Limitation: one action-dense kitchen video is not a held-out or clinical benchmark

## Temporal selection

| Method | Recall at 80% GT coverage | Mean GT coverage | Storage reduction |
| --- | ---: | ---: | ---: |
| v3 change evidence | 35.50% | 43.41% | 62.76% |
| v4 complete interaction episode | 83.59% | 85.97% | 12.60% |

The complete-episode method preserved substantially more action context, while the short change-triggered clips saved more storage.

## VLM benchmark

On 43 shared v3 evidence clips, Qwen3-VL-4B achieved a 100% valid-JSON rate, averaged 1.12 seconds per clip, and used approximately 8.8 GiB peak allocated VRAM. This was the best practical balance in the pilot, not a universal model ranking.

## Second-pass action verification

- Candidates reviewed: 33
- Verified: 18
- Rejected: 15
- Known false cutting description: removed
- Known brief true slice: still missed because it was absent from first-pass candidates

The verifier reduced false positives but could not recover first-pass omissions.
