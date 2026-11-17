# Implementation Guide: Offline Pipeline

This guide explains what each stage does, why it is needed, how it was implemented, and what it produces.

## 1. Separate environments

The main VLM and evaluation code runs in the Python 3.11 `adl-event-logger` environment. Legacy 100DOH runs in a separate Python 3.8 `handobj-100doh` environment. Keeping them separate prevents old detector dependencies from breaking the modern Transformers stack.

```bash
conda env create -n adl-event-logger -f environment.yml
conda activate adl-event-logger
python -m pip install -e .
```

The external 100DOH repository and checkpoint must be installed separately under `tools/`; they are not redistributed in this repository.

## 2. Validate video and ground-truth time

- **What:** inspect FPS, frame count, duration, sampled frames, and annotation timestamps.
- **Why:** if the two time axes disagree, every recall and coverage metric becomes invalid.
- **How:** run `inspect_video.py` and `validate_gt_against_video.py`.
- **Output:** verified video metadata and a per-video GT CSV.

```bash
python scripts/inspect_video.py --video /absolute/path/to/video.mp4
python scripts/validate_gt_against_video.py \
  --video /absolute/path/to/video.mp4 \
  --split-csv data/manifests/epic/split_dev.csv \
  --output data/manifests/epic/VIDEO_ID_gt.csv
```

## 3. Detect hand-object interaction

- **What:** detect hand boxes, contact states, and contacted target objects at 3 FPS.
- **Why:** visible objects are not necessarily being manipulated. The interaction signal is more selective than object presence.
- **How:** run the 100DOH detector in its legacy environment, then convert its JSONL output into a timestamped score table.
- **Output:** frame-level JSONL and a detector-score CSV.

The pipeline treats passive objects as context. It triggers episodes from learned contact/interaction signals, not from every object visible in the kitchen.

## 4. Build variable-length episodes

- **What:** merge consecutive interaction detections into a single episode.
- **Why:** fixed clips can split one continuous activity into several fragments.
- **How:** start after two consecutive active samples, end after two seconds of inactivity, then add configurable context before and after the episode.
- **Output:** `interaction_episodes.csv` and one MP4 per complete episode.

```bash
bash scripts/run_adl_episode_v4.sh /absolute/path/to/video.mp4
```

The detector may briefly flicker while contact continues. Temporal smoothing and a short inactivity grace period prevent a single real action from being broken into many clips.

## 5. Sample frames for the VLM

- **What:** sample the inside of each episode at 3 FPS.
- **Why:** using only eight uniformly spaced frames can miss short evidence inside a long action.
- **How:** keep the saved MP4 at full episode length, but divide long VLM inputs into 10-second chunks with a 1-second overlap.
- **Output:** a frame manifest that maps every sampled image back to episode-relative and video-absolute time.

This distinction is important: chunking limits model input size; it does not destroy the full episode video.

## 6. Compare VLMs

Run SmolVLM2, Qwen3-VL, and InternVL on the same clips and prompt. Store each model's output separately and compare:

- completion and valid-JSON rates;
- latency and peak allocated VRAM;
- normalized verb, object, and ADL scores;
- qualitative failure cases.

The pilot selected Qwen3-VL-4B as the practical default, not as a universally best model.

## 7. Verify action candidates visually

The first pass proposes up to four visible action candidates and their relative times. `verify_vlm_actions_v5.py` reopens frames near each candidate and returns one of:

- `verified`: the visual evidence supports the action;
- `rejected`: the evidence contradicts it;
- `uncertain`: the available frames are insufficient.

For `cut`, `slice`, or `chop`, the verifier requires consecutive-frame evidence that a blade contacts and passes through material. Pressing, squeezing, or opening a bag must not pass as cutting.

```bash
python scripts/verify_vlm_actions_v5.py --help
```

Verification improves precision, but it cannot recover an action that the first pass never proposed. Candidate generation still controls recall.

## 8. Evaluate the system

- **Event recall at 0.8:** fraction of GT actions for which at least 80% of the interval is retained.
- **Mean GT coverage:** average fraction of each GT interval covered by saved episodes.
- **Storage reduction:** `1 - saved_video_duration / original_video_duration`.
- **Detector F1:** measured on reviewed audit frames, not assumed from detector scores.
- **VLM metrics:** reported separately from trigger metrics so localization errors and labeling errors are not confused.

## 9. Reproduction checklist

1. Confirm the video license and keep raw video outside Git.
2. Validate FPS, duration, and timestamp alignment.
3. Generate 100DOH outputs in the legacy environment.
4. Convert scores and build complete interaction episodes.
5. Run the adaptive 3 FPS VLM pipeline.
6. Run the second-pass verifier.
7. Calculate temporal, storage, detector, and VLM metrics.
8. Inspect false positives and false negatives visually before making claims.

## 10. Safety and interpretation

This is a research prototype. Automated labels are hypotheses supported by video evidence, not clinical judgments. Do not publish private video, extracted frames, model weights, tokens, absolute local paths, or authentication material.
