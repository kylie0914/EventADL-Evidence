from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adl_event_logger.triggers import active_object_score
from adl_event_logger.video import iter_sampled_frames


def main() -> None:
    """영상 전체를 저속 sampling하여 세 종류의 detector score를 한 CSV로 저장한다.

    MediaPipe는 손, Faster R-CNN은 일반 물체를 찾는다. active-object는 별도 학습
    모델이 아니라 손-물체 상자의 거리가 가까울수록 높아지는 해석 가능한 baseline이다.
    """
    parser = argparse.ArgumentParser(description="Hand, object and active-object score extraction")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config-fps", type=float, default=2.0)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--object-confidence", type=float, default=0.5)
    parser.add_argument("--max-distance", type=float, default=0.2)
    parser.add_argument("--output", default="outputs/detectors/scores.csv")
    args = parser.parse_args()

    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError("Install mediapipe first: python -m pip install mediapipe") from error
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

    # 모델은 한 번만 GPU에 올리고 모든 sampled frame에서 재사용한다.
    device = torch.device("cuda")
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    object_model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    categories = weights.meta["categories"]
    hand_model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35)
    rows = []
    try:
        for time_sec, frame_index, frame_bgr in iter_sampled_frames(args.video, args.config_fps, args.resize_width):
            height, width = frame_bgr.shape[:2]
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            hand_result = hand_model.process(frame_rgb)
            hand_boxes = []
            hand_confidences = []
            # landmark의 최소/최대 좌표로 정규화된 손 bounding box를 만든다.
            if hand_result.multi_hand_landmarks:
                for landmarks, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    xs = [item.x for item in landmarks.landmark]
                    ys = [item.y for item in landmarks.landmark]
                    hand_boxes.append((max(0, min(xs)), max(0, min(ys)), min(1, max(xs)), min(1, max(ys))))
                    hand_confidences.append(float(handedness.classification[0].score))
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255).to(device)
            with torch.inference_mode():
                prediction = object_model([tensor])[0]
            # COCO 물체 중 confidence threshold를 넘은 예측만 남긴다.
            object_boxes = []
            object_items = []
            for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
                confidence = float(score)
                if confidence < args.object_confidence:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                normalized = (x1 / width, y1 / height, x2 / width, y2 / height)
                object_boxes.append(normalized)
                object_items.append({"label": categories[int(label)], "confidence": confidence, "bbox": normalized})
            # '물체가 보임'과 '그 물체를 사용 중'을 구분하는 핵심 baseline이다.
            active = active_object_score(hand_boxes, object_boxes, args.max_distance)
            rows.append({
                "time_sec": time_sec, "frame_index": frame_index,
                "hand_score": max(hand_confidences, default=0.0),
                "object_score": max((item["confidence"] for item in object_items), default=0.0),
                "active_object_score": active,
                "hand_boxes": json.dumps(hand_boxes), "objects": json.dumps(object_items),
            })
    finally:
        hand_model.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"saved {len(rows)} samples to {output}")


if __name__ == "__main__":
    main()

