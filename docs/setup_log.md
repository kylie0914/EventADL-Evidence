# Environment Setup and Troubleshooting Record

This document records the reproducible environment setup and the important problems encountered during installation. It intentionally contains no passwords, access tokens, private video, account details, or machine-specific absolute paths.

## 1. Validated environment

| Component | Validated value |
| --- | --- |
| Conda | 25.3.1 |
| Main environment | `adl-event-logger` |
| Python | 3.11 |
| GPU | NVIDIA RTX 3090, 24 GB VRAM |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.23.0 |
| Accelerate | 1.15.0 |
| Decord | 0.6.0 |
| PyAV | 18.1.0 |
| TorchCodec | 0.17.0, CUDA 13.0 wheel |
| SmolVLM2 video smoke test | passed |

100DOH is intentionally isolated in a separate Python 3.8 environment named `handobj-100doh`.

## 2. Create the named Conda environment

The project originally tried a path-prefix environment. A named environment is easier to activate and was adopted as the final setup.

```bash
cd /path/to/egocentric_CAM_VLM
conda env create -n adl-event-logger -f environment.yml
conda activate adl-event-logger
which python
python --version
```

Update an existing environment with:

```bash
conda env update -n adl-event-logger -f environment.yml --prune
```

If an incorrectly created prefix environment must be removed, first deactivate it and confirm the exact path with `conda env list`:

```bash
conda deactivate
conda env list
conda env remove -p "$PWD/.conda/envs/adl-event-logger"
```

## 3. Verify core Python packages

```bash
python -c "import cv2, pandas, numpy, yaml, pydantic; print('Environment ready'); print('OpenCV:', cv2.__version__)"
```

Validated result:

```text
Environment ready
OpenCV: 4.14.0
```

## 4. Verify the GPU and driver

```bash
nvidia-smi
```

Validated hardware:

```text
GPU: NVIDIA GeForce RTX 3090
VRAM: 24576 MiB
Driver: 580.159.04
Driver-reported CUDA compatibility: 13.0
```

The CUDA version shown by `nvidia-smi` is the driver's compatibility level. Check the CUDA runtime bundled with PyTorch separately using `torch.version.cuda`.

## 5. Install and verify PyTorch

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.14.1 \
  torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
```

The GPU matrix test passed:

```bash
python -c "import torch; x=torch.rand(4000,4000,device='cuda'); y=x@x; print('GPU computation passed:', y.shape, y.device); print('Allocated VRAM MB:', round(torch.cuda.max_memory_allocated()/1024**2,1))"
```

```text
GPU computation passed: torch.Size([4000, 4000]) cuda:0
Allocated VRAM MB: 132.1
```

## 6. Install common VLM packages

```bash
python -m pip install --upgrade \
  "transformers>=4.57" \
  "accelerate>=1.0" \
  "safetensors>=0.4" \
  "huggingface-hub>=0.30" \
  "num2words>=0.5.14" \
  "decord>=0.6.0" \
  "av>=14" \
  "sentencepiece>=0.2" \
  "protobuf>=5" \
  "einops>=0.8"
```

```bash
python -c "import torch, transformers, accelerate, huggingface_hub, decord, av; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('huggingface_hub:', huggingface_hub.__version__); print('decord:', decord.__version__); print('PyAV:', av.__version__)"
```

Validated output:

```text
torch: 2.14.1+cu130
torch CUDA: 13.0
transformers: 5.23.0
accelerate: 1.15.0
huggingface_hub: 2.4.0
decord: 0.6.0
PyAV: 18.1.0
```

Public model downloads do not require authentication. Never print or store a Hugging Face token in a log.

## 7. Video smoke-test input

Before downloading EPIC-KITCHENS, a 537 KB public sample was used to test only the video-decoding path:

```bash
mkdir -p data/smoke
hf download nateraw/video-demo archery.mp4 --repo-type dataset --local-dir data/smoke
mv data/smoke/archery.mp4 data/smoke/test.mp4
ls -lh data/smoke/test.mp4
```

`data/smoke/test.mp4` is a file path, not a shell command.

## 8. TorchCodec error and fix

The first video smoke test loaded all model weights but failed during preprocessing:

```text
ImportError: `torchcodec` is not installed or can't be loaded
```

Transformers 5.x uses TorchCodec because the old TorchVision video decoder was removed. Install the wheel that matches the PyTorch/CUDA build:

```bash
python -m pip install torchcodec==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -c "import torchcodec; print('torchcodec:', torchcodec.__version__)"
python scripts/smoke_test_smolvlm2_video.py --video data/smoke/test.mp4
```

Validated result:

```text
Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Answer: Two women are practicing archery on a grassy field.
Inference time: 0.9085 seconds
Peak VRAM: 4585.855 MiB
```

This confirms MP4 decoding, video preprocessing, GPU inference, and text generation.

## 9. Annotation-summary path correction

An early command looked for `outputs/annotation_analysis/annotation_summary.json`, but the script writes to the manifest directory. The correct command is:

```bash
cat data/manifests/epic/annotation_summary.json
```

This was a documentation-path error, not an analysis failure.

## 10. Participant leakage fix

The first split attempt raised:

```text
AssertionError: Participant leakage between dev and test
```

EPIC's official train and validation sets may contain the same participant. The split code now removes selected development participants from the test candidate pool before choosing test participants. The leakage assertion remains enabled as a safety check.

## 11. EPIC downloader metadata-path fix

The first video-download attempt stopped before transfer because it searched for `data/epic_55_splits.csv`. The metadata actually lives under the cloned downloader repository:

```text
tools/epic-kitchens-download-scripts/data/
```

The final download command therefore passes explicit paths for the EPIC-55 splits, EPIC-100 splits, MD5 file, and errata file. See `docs/runbook.md`.

## 12. P04_01 download and verification

Only the selected pilot video was downloaded. The downloader resumed a partial file and verified its MD5 before renaming it to the final MP4.

```text
Video: P04_01
Expected size: approximately 4.05 GiB
Verification: MD5 passed
```

The video, partial downloads, extracted frames, clips, and reports are excluded from Git.

## 13. Final validation checklist

```bash
conda activate adl-event-logger
python scripts/check_system.py
python -m compileall -q src scripts
```

- Main VLM environment imports successfully.
- CUDA tensor computation succeeds on the RTX 3090.
- SmolVLM2 image and video smoke tests succeed.
- TorchCodec matches the installed PyTorch/CUDA build.
- Raw data, checkpoints, cache files, outputs, tokens, and local paths remain outside version control.
