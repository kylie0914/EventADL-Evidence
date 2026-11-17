# Code Structure Guide

## Design principles

The project is divided into three layers:

1. `src/adl_event_logger/`: reusable core logic;
2. `scripts/`: small command-line entry points;
3. `configs/`, `data/`, and `outputs/`: configuration, input, and generated results.

Scripts should not duplicate shared functions. Video timestamp conversion belongs in `video.py`, while conversion from trigger scores to temporal events belongs in `events.py`.

## Data flow

```text
EPIC annotation CSV
  -> annotations.py
  -> split_dev.csv / split_test.csv / pilot_videos.csv

Untrimmed MP4
  -> video.py
  -> motion / hand / object / interaction scores
  -> events.py
  -> variable-length events.csv
  -> event MP4 files
  -> VLM action JSONL
  -> visual verification
  -> evaluation and plotting
```

## Core modules

### `video.py`

- reads MP4 metadata;
- converts frame indices to seconds;
- samples frames at a requested FPS;
- extracts frames for VLM input.

### `annotations.py`

- discovers EPIC train/validation CSV files;
- validates required columns and temporal intervals;
- summarizes participants and videos;
- creates participant-disjoint development and test splits.

### `triggers.py`

- calculates motion scores;
- calculates bounding-box IoU and center distance;
- provides an interpretable active-object heuristic;
- combines multiple normalized signals.

`active_object_score()` is a baseline, not a learned contact detector. The final pipeline replaces it with 100DOH interaction outputs using a compatible timestamped score format.

### `events.py`

- applies score thresholds;
- uses activation/deactivation hysteresis;
- adds pre/post context;
- merges nearby events.

### `evaluation.py`

- measures how many GT events retain at least 80% temporal coverage;
- uses interval unions so overlapping saved clips are not counted twice.

## Important scripts

- `smoke_test_smolvlm2_image.py`: verifies model download and single-image GPU inference.
- `smoke_test_smolvlm2_video.py`: verifies local MP4 decoding and video inference.
- `analyze_epic_annotations.py`: summarizes annotation scale without video downloads.
- `make_epic_split.py`: creates participant-held-out pilot splits.
- `inspect_video.py`: checks metadata and sampled frames.
- `run_motion_trigger.py`: runs the simplest event trigger.
- `run_detector_scores.py`: runs MediaPipe hand, TorchVision object, and heuristic interaction baselines.
- `convert_100doh_scores.py`: converts learned interaction output into the common score format.
- `build_interaction_episodes_v4.py`: creates complete variable-length interaction episodes.
- `build_adaptive_vlm_dataset_v4.py`: samples episode frames at adaptive 3 FPS.
- `run_qwen_episode_v4.py`: produces structured Qwen3-VL episode labels.
- `verify_vlm_actions_v5.py`: performs second-pass visual candidate verification.
- `evaluate_events.py`: calculates event recall and GT coverage.
- `evaluate_storage.py`: calculates storage reduction relative to the full recording.
- `evaluate_detector_audit.py`: calculates detector Precision, Recall, and F1.
- `evaluate_vlm_labels.py`: calculates verb/object Accuracy and Macro-F1.
- `plot_portfolio_results.py`: produces the public result figures.

## Standard file contracts

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
{"event_id":"episode_0001","verb":"take","object":"cup","uncertain":false}
```

Any detector or VLM can be replaced as long as it preserves these file contracts and timestamp semantics.
