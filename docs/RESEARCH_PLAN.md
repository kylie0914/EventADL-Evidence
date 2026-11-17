# Research Plan: Offline Event-Based ADL Video Labeling

## 1. Objective

Build an offline pipeline that finds hand-object interaction periods in long egocentric daily-living videos, measures how much useful activity is preserved versus how much video is stored, and uses vision-language models (VLMs) to describe actions and manipulated objects in structured JSON.

## 2. Research questions

1. Can event-based selection preserve meaningful activity while storing less video than Full or Periodic sampling?
2. Does a learned hand-object interaction detector separate manipulation from passive object visibility better than simple hand/object presence?
3. Which VLM that fits an RTX 3090 offers the best balance of JSON reliability, speed, VRAM use, and action-label quality?
4. Can a second visual verification pass reduce false-positive action labels from the first VLM pass?

## 3. Existing methods and limitations

- **Full video:** minimizes missed activity, but does not reduce storage or review time.
- **Periodic clips:** easy to implement, but ignores action boundaries and may miss short actions.
- **Motion trigger:** also reacts to camera shake and lighting changes.
- **Hand/object presence:** selects too much video when many passive objects are visible.
- **Distance heuristic:** easy to interpret, but it is not trained to recognize physical contact.

## 4. Proposed pipeline

```text
Untrimmed egocentric video
  -> motion and 100DOH hand-object interaction signals
  -> temporal smoothing
  -> variable-length episodes from interaction start to end
  -> pre/post context
  -> adaptive 3 FPS sampling inside each episode
  -> first-pass VLM action candidates
  -> second-pass visual verification
  -> offline temporal, storage, detector, and VLM evaluation
```

The central design choice is to save complete interaction episodes rather than treating every active frame as a separate event. Long episodes remain complete MP4 files; only the VLM input is divided into overlapping chunks when necessary.

## 5. Data and models

### Dataset

- **EPIC-KITCHENS-100** provides untrimmed first-person kitchen video and temporal verb/noun annotations.
- The development pilot uses `P04_01`, a 19.25-minute official video with 262 annotated action intervals.
- A participant-separated development/test split is generated to avoid identity leakage.
- This pilot video is action-dense, so storage reduction on it should not be interpreted as performance on hour-long home video with long inactive periods.

### Models

- **100DOH Hand-Object Detector:** hand boxes, contact states, and contacted target-object boxes.
- **SmolVLM2-2.2B:** compact video-language baseline.
- **Qwen3-VL-4B:** primary local VLM selected for its pilot balance of speed, JSON validity, and label quality.
- **InternVL3.5-4B:** same-size comparison model.

## 6. Evaluation

- **Temporal selection:** event recall at 80% ground-truth coverage and mean GT coverage.
- **Efficiency:** saved duration, storage ratio, storage reduction, and number of saved clips.
- **Detector audit:** hand and interaction Precision, Recall, and F1 on manually reviewed frames.
- **VLM:** JSON validity, completion rate, normalized verb/object/ADL scores, latency, and peak VRAM.
- **Verifier:** number of verified, rejected, and uncertain candidates, plus regression tests for known false positives.

## 7. Expected contribution

The project is not a clinical system. It is a reproducible engineering study showing how learned hand-object interaction, complete event episodes, local VLM labeling, and evidence-based verification can be combined for egocentric ADL video. The final result should clearly expose the trade-off between storage efficiency and preserving enough context for later review.
