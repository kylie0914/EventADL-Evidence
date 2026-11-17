# Command Runbook

Run every command from the repository root. Complete and verify one stage before moving to the next. Replace `PXX_XX` with the selected video ID.

```bash
cd /path/to/egocentric_CAM_VLM
conda activate adl-event-logger
```

## 1. Verify the environment

```bash
python scripts/check_system.py
python -c "import torch, transformers, accelerate, decord, av; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, accelerate.__version__, decord.__version__, av.__version__)"
```

## 2. Run SmolVLM2 smoke tests

Download a small public image:

```bash
mkdir -p data/smoke
curl -L https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg -o data/smoke/bee.jpg
python scripts/smoke_test_smolvlm2_image.py --image data/smoke/bee.jpg
```

Download a small public video for decoder testing. This is not an EPIC-KITCHENS download.

```bash
hf download nateraw/video-demo archery.mp4 --repo-type dataset --local-dir data/smoke
mv data/smoke/archery.mp4 data/smoke/test.mp4
python -m pip install torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cu130
python scripts/smoke_test_smolvlm2_video.py --video data/smoke/test.mp4
```

The first model run downloads approximately 5 GB to the Hugging Face cache.

## 3. Download and analyze EPIC-KITCHENS annotations

This downloads annotation files only, not the original videos.

```bash
mkdir -p data/annotations
git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git \
  data/annotations/epic-kitchens-100-annotations
python -m pip install -e .
python scripts/analyze_epic_annotations.py \
  --annotations-dir data/annotations/epic-kitchens-100-annotations
cat data/manifests/epic/annotation_summary.json
```

## 4. Create participant-separated splits

The splitter removes development participants from the test candidate pool to prevent participant leakage.

```bash
python scripts/make_epic_split.py \
  --annotations-dir data/annotations/epic-kitchens-100-annotations \
  --dev-participants 2 \
  --test-participants 1
column -s, -t data/manifests/epic/pilot_videos.csv | less -S
```

## 5. Download one selected EPIC video

Clone the official downloader and inspect its options before downloading. Do not download the full dataset by accident.

```bash
git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-download-scripts.git \
  tools/epic-kitchens-download-scripts
python tools/epic-kitchens-download-scripts/epic_downloader.py --help
```

The pilot used only `P04_01`. The explicit metadata paths avoid resolving `data/...` against the project directory.

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

Verify the downloaded file and create a convenient symbolic link without duplicating 4 GB of data:

```bash
find data/raw/EPIC-KITCHENS -type f -name 'P04_01.MP4' -ls
mkdir -p data/raw/epic
ln -s "$(pwd)/data/raw/EPIC-KITCHENS/P04/videos/P04_01.MP4" data/raw/epic/P04_01.MP4
```

## 6. Validate video metadata and GT timestamps

```bash
python scripts/inspect_video.py --video data/raw/epic/PXX_XX.MP4
python scripts/validate_gt_against_video.py \
  --video data/raw/epic/PXX_XX.MP4 \
  --split-csv data/manifests/epic/split_dev.csv \
  --output data/manifests/epic/PXX_XX_gt.csv
```

## 7. Run simple baselines

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 \
  --method full --output outputs/events/PXX_XX_full.csv
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 \
  --method periodic --period-seconds 10 --clip-seconds 8 \
  --output outputs/events/PXX_XX_periodic.csv
python scripts/run_motion_trigger.py --video data/raw/epic/PXX_XX.MP4 \
  --output-dir outputs/motion/PXX_XX
```

## 8. Run hand/object baselines

```bash
python -m pip install mediapipe matplotlib seaborn scikit-learn
python scripts/run_detector_scores.py --video data/raw/epic/PXX_XX.MP4 \
  --output outputs/detectors/PXX_XX_scores.csv
```

Generate `hand`, `object`, and `active_object` event files by changing the `--method` value:

```bash
python scripts/run_event_method.py --video data/raw/epic/PXX_XX.MP4 \
  --method hand --scores outputs/detectors/PXX_XX_scores.csv \
  --output outputs/events/PXX_XX_hand.csv
```

## 9. Audit detector predictions

```bash
python scripts/build_detector_audit.py --video data/raw/epic/PXX_XX.MP4 \
  --scores outputs/detectors/PXX_XX_scores.csv --count 250 \
  --output-dir data/audit/PXX_XX
```

Review the `gt_*` fields in `data/audit/PXX_XX/audit_labels.csv`, then calculate detector metrics:

```bash
python scripts/evaluate_detector_audit.py \
  --audit-csv data/audit/PXX_XX/audit_labels.csv \
  --output outputs/metrics/PXX_XX_detector_audit.json
```

## 10. Run 100DOH and build complete episodes

100DOH requires its separate Python 3.8 environment and external checkpoint. After its frame-level output is available, run the complete episode pipeline:

```bash
conda activate adl-event-logger
bash scripts/run_adl_episode_v4.sh /absolute/path/to/video.mp4
```

The script converts 100DOH scores, smooths interaction states, creates complete episodes, saves clips, samples VLM frames at 3 FPS, runs Qwen3-VL, and writes storage and comparison metrics.

## 11. Run VLM comparisons and verification

For the staged `P04_01` benchmark:

```bash
bash scripts/run_p04_01_vlm_benchmark.sh
python scripts/verify_vlm_actions_v5.py --help
```

Use the same clip set and prompt for every model. Keep trigger metrics separate from VLM labeling metrics.

## 12. Evaluate and plot results

```bash
python scripts/evaluate_events.py \
  --gt data/manifests/epic/PXX_XX_gt.csv \
  --predictions outputs/events/PXX_XX_fusion.csv \
  --output outputs/metrics/PXX_XX_fusion.json
python scripts/evaluate_storage.py \
  --video data/raw/epic/PXX_XX.MP4 \
  --events outputs/events/PXX_XX_fusion.csv \
  --output outputs/metrics/PXX_XX_fusion_storage.json
python scripts/plot_portfolio_results.py
```

## Completion checks

- Raw videos, frames, clips, checkpoints, and outputs are outside Git.
- Video metadata and GT timestamps agree.
- Detector metrics come from reviewed frames.
- Every VLM result records model ID, prompt, latency, VRAM, and valid-JSON status.
- Published conclusions identify `P04_01` as a single-video development pilot.
