"""Video input utilities shared by triggers, auditing, and evaluation.

Keep timestamp handling here so every experiment uses the same frame-to-time rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def read_metadata(path: str | Path) -> VideoMetadata:
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) for {video_path}")
    return VideoMetadata(
        path=str(video_path.resolve()),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
    )


def metadata_dict(path: str | Path) -> dict[str, object]:
    return asdict(read_metadata(path))


def iter_sampled_frames(path: str | Path, sample_fps: float, resize_width: int | None = None):
    """Yield `(time_sec, frame_index, BGR frame)` at an approximate fixed rate."""
    metadata = read_metadata(path)
    capture = cv2.VideoCapture(str(path))
    step = max(1, round(metadata.fps / sample_fps))
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                if resize_width and frame.shape[1] != resize_width:
                    scale = resize_width / frame.shape[1]
                    frame = cv2.resize(frame, (resize_width, round(frame.shape[0] * scale)))
                yield frame_index / metadata.fps, frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def sample_uniform_rgb(path: str | Path, count: int) -> list[np.ndarray]:
    """Read `count` RGB frames spread across the usable interior of a video.

    OpenCV/FFmpeg can report the correct total frame count while still failing to
    seek to the very last frame of a long-GOP H.264 file. Sampling at the centers
    of equal temporal bins avoids that fragile endpoint and is also a better visual
    summary than spending two samples on the exact first/last frame.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    metadata = read_metadata(path)
    if metadata.frame_count <= 0:
        raise ValueError(f"Video has no frames: {path}")

    # Example for count=5: sample near 1/6, 2/6, ..., 5/6 of the video rather
    # than requesting frame_count-1, which some H.264 decoders cannot seek to.
    indices = np.linspace(0, metadata.frame_count, count + 2, dtype=int)[1:-1]
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                # A keyframe seek can land just after the requested position. Retry
                # slightly earlier before treating a valid video as unreadable.
                for backoff in (1, 2, 4, 8, 16, 32):
                    retry_index = max(0, int(index) - backoff)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, retry_index)
                    ok, frame = capture.read()
                    if ok:
                        break
            if not ok:
                raise RuntimeError(f"Could not read a frame near {index} from {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return frames

