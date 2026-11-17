# Data directory

Dataset files are not committed to Git.

- `annotations/`: official EPIC-KITCHENS annotation clone
- `raw/`: downloaded EPIC-KITCHENS and private GoPro videos
- `manifests/`: locally generated participant splits and selected video IDs
- `audit/`: detector-audit frames and human labels
- `smoke/`: small local image/video smoke-test inputs

Follow `docs/runbook.md` to recreate these files. Respect each dataset's license and
do not publish original or private videos in this repository.
