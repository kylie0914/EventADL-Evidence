# 설치 및 환경 구성 기록

> 이 문서는 초기 설치 기록 보존용이다. 이후의 실제 명령·오류·수정·해결 과정은
> `docs/actual_execution_log.md` 한 파일에 통합해서 기록한다.

이 문서는 프로젝트를 다른 컴퓨터에서 다시 만들 수 있도록 실제 실행 명령과 확인
결과를 순서대로 기록한다. 비밀번호, 토큰, 개인정보는 기록하지 않는다.

## 작업 진행 원칙

- Conda/pip 설치, 계정 로그인, 모델 다운로드, 데이터셋 다운로드, 실험 명령 실행은
  사용자가 자신의 터미널에서 직접 수행한다.
- 어시스턴트는 실행할 명령을 순서대로 제공하고, 각 명령의 목적·예상 출력·오류 확인
  방법을 설명한다. 사용자 확인 없이 설치나 대용량 다운로드를 대신 실행하지 않는다.
- 구현 코드가 필요하면 어시스턴트가 프로젝트 파일로 작성한 뒤, 변경한 파일과 핵심
  로직, 입력·출력, 실행 명령, 완료 기준을 사용자에게 설명한다.
- 사용자가 실행 결과를 전달하면 실제 버전·성공 여부·오류를 이 로그에 갱신한다.
- 데이터셋은 전체를 무조건 받지 않는다. 먼저 annotation과 파일 목록을 확인하고,
  연구에 필요한 participant/video subset을 정한 뒤 공식 도구로 일부만 다운로드한다.
- 모델도 한꺼번에 받지 않는다. SmolVLM2 smoke test를 먼저 통과한 뒤 Qwen3-VL,
  InternVL3.5 순서로 추가한다.

## 현재 상태

| 단계 | 상태 | 확인 결과 |
| --- | --- | --- |
| Miniconda 확인 | 완료 | Conda 25.3.1 |
| 이름 기반 Conda 환경 생성 | 완료 | `adl-event-logger` |
| Python 기본 패키지 | 완료 | OpenCV 4.14.0 및 주요 import 성공 |
| NVIDIA GPU 확인 | 완료 | RTX 3090, 24GB VRAM |
| PyTorch CUDA 설치 | 완료 | CUDA tensor 및 4000×4000 행렬곱 성공 |
| VLM 공통 패키지 | 완료 | Transformers 5.23.0, Accelerate 1.15.0 |
| SmolVLM2 모델 다운로드 | 완료 | 2.2B weight 657개 로드 성공 |
| 데이터셋 다운로드 | 진행 전 | 아직 다운로드하지 않음 |
| Smoke-test MP4 | 완료 | `nateraw/video-demo/archery.mp4`, 537KB |
| 비디오 VLM smoke test | 완료 | SmolVLM2 답변 생성, 0.91초, 최대 VRAM 4585.85MiB |

## 1. 시스템과 Miniconda 확인

작업 폴더:

```bash
cd /path/to/egocentric_CAM_VLM
```

확인 명령:

```bash
which conda
conda --version
```

확인된 결과:

```text
$HOME/miniconda3/bin/conda
conda 25.3.1
```

## 2. 초기 prefix 환경을 제거하고 이름 기반 환경으로 변경

처음에는 프로젝트 경로를 직접 지정하는 prefix 방식이 안내되었지만, 사용 편의를 위해
이름 기반 환경으로 변경했다. 최종적으로 사용하는 활성화 명령은 다음과 같다.

```bash
conda activate adl-event-logger
```

기존 환경이 활성화되어 있을 경우:

```bash
conda deactivate
```

환경 목록 확인:

```bash
conda env list
```

기존 환경이 prefix 방식으로 생성된 경우 사용한 제거 명령:

```bash
conda env remove -p "$PWD/.conda/envs/adl-event-logger"
```

기존 환경이 이름 기반으로 생성된 경우의 제거 명령:

```bash
conda env remove -n adl-event-logger
```

두 제거 명령은 환경 목록에 표시된 방식에 맞는 것 하나만 사용한다.

## 3. 이름 기반 Conda 환경 생성

환경 정의 파일: `environment.yml`

```bash
cd /path/to/egocentric_CAM_VLM
conda env create -n adl-event-logger -f environment.yml
conda activate adl-event-logger
```

Python 확인:

```bash
which python
python --version
```

환경 업데이트 시 사용할 명령:

```bash
conda env update -n adl-event-logger -f environment.yml --prune
```

## 4. 기본 Python 패키지 검증

실행 명령:

```bash
python -c "import cv2, pandas, numpy, yaml, pydantic; print('환경 설치 성공'); print('OpenCV:', cv2.__version__)"
```

실제 결과:

```text
환경 설치 성공
OpenCV: 4.14.0
```

따라서 OpenCV, pandas, NumPy, PyYAML, Pydantic import가 성공했다.

## 5. GPU와 드라이버 확인

실행 명령:

```bash
nvidia-smi
```

확인된 주요 정보:

```text
GPU: NVIDIA GeForce RTX 3090
VRAM: 24576 MiB (24GB)
Driver: 580.159.04
Driver-reported CUDA compatibility: 13.0
```

주의: `nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 CUDA 호환 수준이다. PyTorch가
사용하는 CUDA runtime은 설치한 PyTorch wheel에 포함되며, `torch.version.cuda`로 별도
확인한다.

## 6. PyTorch CUDA 설치

환경 활성화:

```bash
conda activate adl-event-logger
```

pip 업데이트:

```bash
python -m pip install --upgrade pip
```

Python 3.11, Linux, RTX 3090과 현재 드라이버에 사용할 고정 설치 명령:

```bash
python -m pip install \
  torch==2.14.1 \
  torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

설치 검증:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA 사용 가능:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))"
```

GPU 행렬 연산 검증:

```bash
python -c "import torch; x=torch.rand(4000,4000,device='cuda'); y=x@x; print('GPU 연산 성공:', y.shape, y.device); print('사용 VRAM MB:', round(torch.cuda.max_memory_allocated()/1024**2,1))"
```

GPU 행렬 연산의 실제 결과:

```text
GPU 연산 성공: torch.Size([4000, 4000]) cuda:0
사용 VRAM MB: 132.1
```

따라서 PyTorch가 CUDA GPU에 tensor를 만들고 행렬곱을 수행하는 것까지 확인했다.
정확한 `torch.__version__`과 `torch.version.cuda` 출력은 패키지 버전 점검 단계에서 함께
기록한다.

## 7. VLM 공통 패키지 설치

SmolVLM2 공식 모델 카드는 최신 Transformers와 `num2words`, 비디오 입력에는 `decord`를
요구한다. Qwen3-VL도 최신 Transformers 사용을 권장한다. FlashAttention은 첫 실행의
필수 조건이 아니며 CUDA/PyTorch 조합별 빌드 문제가 생길 수 있으므로 smoke test 이후
별도 최적화 단계에서 설치한다.

환경 활성화:

```bash
conda activate adl-event-logger
```

공통 패키지 설치:

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

설치 확인:

```bash
python -c "import torch, transformers, accelerate, safetensors, huggingface_hub, decord, av; print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('transformers:', transformers.__version__); print('accelerate:', accelerate.__version__); print('decord:', decord.__version__); print('PyAV:', av.__version__)"
```

Hugging Face CLI 확인:

```bash
hf version
hf auth whoami
```

공개 모델 다운로드에는 로그인이 필수는 아니다. `hf auth whoami`가 로그인되지 않았다고
나와도 다음 단계로 진행할 수 있다. 토큰 값 자체는 터미널 출력이나 로그에 남기지 않는다.

실제 확인 결과:

```text
torch: 2.14.1+cu130
torch CUDA: 13.0
CUDA: True
transformers: 5.23.0
accelerate: 1.15.0
huggingface_hub: 2.4.0
decord: 0.6.0
PyAV: 18.1.0
```

## 8. 다음 설치 단계

PyTorch GPU 검증 후 다음 순서로 진행한다.

1. Transformers, Accelerate, Safetensors 설치
2. 비디오 decoding 패키지 검증

## 9. 공개 smoke-test 영상 다운로드

직접 촬영 영상과 EPIC-KITCHENS 원본을 받기 전, decoder만 확인하기 위해 공개 영상
한 파일을 선택적으로 다운로드했다.

```bash
hf download nateraw/video-demo archery.mp4 --repo-type dataset --local-dir data/smoke
mv data/smoke/archery.mp4 data/smoke/test.mp4
ls -lh data/smoke/test.mp4
```

실제 결과:

```text
Downloaded: data/smoke/archery.mp4
Final path: data/smoke/test.mp4
Size: 537K
```

`data/smoke/test.mp4`는 실행 명령이 아니라 파일 경로다. 파일 자체를 Bash 명령처럼
입력하면 `그런 파일이나 디렉터리가 없습니다` 또는 실행 권한 오류가 발생할 수 있다.

## 10. 첫 비디오 smoke test와 TorchCodec 오류

실행 명령:

```bash
python scripts/smoke_test_smolvlm2_video.py --video data/smoke/test.mp4
```

모델 weight 657개는 정상적으로 로드됐지만 preprocessing 단계에서 다음 오류가 났다.

```text
ImportError: `torchcodec` is not installed or can't be loaded
```

원인은 Transformers 5.23.0에서 제거된 TorchVision video decoder 대신 TorchCodec을
기본 decoder로 사용하기 때문이다. PyTorch 2.14/CUDA 13.0과 호환되는 공식 wheel을
다음 명령으로 추가한다.

```bash
python -m pip install torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cu130
```

TorchCodec 설치 후 같은 명령을 다시 실행해 성공했다.

```text
Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Answer: Two women are practicing archery on a grassy field.
Inference time: 0.9085 seconds
Peak VRAM: 4585.855 MiB
```

따라서 로컬 MP4 decoding, video processor, GPU 추론, 텍스트 생성 전체 경로가
정상 작동한다. 다음 단계는 EPIC-KITCHENS 원본 영상이 아니라 annotation 저장소만
먼저 받는 것이다.

실제로 입력한 명령, 중간 오류, 수정 이유를 쉬운 설명과 함께 모은 기록은
`docs/actual_execution_log.md`에서 관리한다.

## 11. Annotation 결과 확인 경로 수정

처음 안내된 `outputs/annotation_analysis/annotation_summary.json`은 실제 스크립트의 기본
출력 경로와 달랐다. 올바른 확인 명령은 다음과 같다.

```bash
cat data/manifests/epic/annotation_summary.json
```

패키지 설치나 분석 알고리즘의 문제가 아니라 결과를 확인하는 경로만 수정했다.

최종 상태는 해결 완료다. 분석 결과 JSON이 실제로 출력되었고 마지막 열 목록에
`stop_sec`, `duration_sec`가 포함된 것도 확인됐다. 이후에는
`outputs/annotation_analysis/annotation_summary.json` 경로를 사용하지 않는다.

## 12. Participant split 누수 검사와 수정

첫 split 실행은 `Participant leakage between dev and test`로 중단됐다. EPIC 공식
train/validation에서 같은 참가자가 선택된 것이 원인이다. 누수 검사를 삭제하지 않고,
validation 후보에서 dev 참가자를 제외한 뒤 test 참가자를 선택하도록
`annotations.py`를 수정했다. 같은 명령으로 다시 실행해 검증한다.

## 13. EPIC 원본 영상 다운로드 단계 보완

`data/raw/epic`이 `합계 0`인 것은 downloader 저장소와 `--help`만 실행하고 실제 영상
다운로드 명령은 실행하지 않았기 때문이다. 첫 pilot은 `P04_01` 한 파일로 정했고,
진행률·재개·MD5 검사를 제공하는 `epic_video_downloader_verify.py` 명령을 runbook에
추가했다. 전체 데이터셋 다운로드는 여전히 하지 않는다.

첫 실행은 downloader가 `data/epic_55_splits.csv`를 찾지 못해 다운로드 전에 중단됐다.
프로젝트 폴더를 확인한 결과 metadata는
`tools/epic-kitchens-download-scripts/data/`에 있었다. 따라서 네 metadata 파일 경로를
명시하는 명령으로 수정했다. `P04_01` 한 파일만 받는 범위에는 변화가 없다.

