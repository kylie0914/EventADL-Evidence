from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLM2 local-video smoke test")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe the main action, manipulated object, and visible evidence. Do not guess.")
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke/smolvlm2_video.json"))
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "path": str(args.video.resolve())},
        {"type": "text", "text": args.prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    elapsed = time.perf_counter() - started
    new_tokens = generated[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    result = {
        "model": MODEL_ID, "input": str(args.video.resolve()), "prompt": args.prompt,
        "answer": answer, "inference_seconds": elapsed,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

