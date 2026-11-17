"""Run one Transformers VLM on a shared multi-frame clip benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


def clean_value(value: object, fallback: str = "unknown") -> str:
    text = str(value if value is not None else fallback).strip().lower()
    return text or fallback


def parse_prediction(text: str) -> dict[str, object]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned
    # Some models add a sentence before/after JSON. Use the outermost object.
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise TypeError("prediction is not an object")
        return {
            "verb": clean_value(payload.get("verb")),
            "object": clean_value(payload.get("object")),
            "description": str(payload.get("description", "")).strip(),
            "uncertain": bool(payload.get("uncertain", False)),
            "parse_status": "valid_json",
        }
    except (json.JSONDecodeError, TypeError):
        return {
            "verb": "unknown", "object": "unknown", "description": cleaned,
            "uncertain": True, "parse_status": "invalid_json",
        }


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--vocabulary", required=True, type=Path)
parser.add_argument("--model-id", required=True)
parser.add_argument("--model-key", required=True)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--max-new-tokens", type=int, default=128)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")
manifest = pd.read_csv(args.manifest)
vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
prompt = f"""You are labeling an egocentric daily-living clip shown as 8 frames in chronological order.
Choose the single MAIN visible hand action and manipulated object. Do not infer hidden actions.
Use exactly one verb from this closed vocabulary:
{', '.join(vocabulary['verbs'])}
Use exactly one object from this closed vocabulary:
{', '.join(vocabulary['objects'])}
Return exactly one JSON object and no markdown:
{{"verb":"...","object":"...","description":"one short visible description","uncertain":false}}
Use unknown when the frames do not provide enough evidence."""

args.output.parent.mkdir(parents=True, exist_ok=True)
completed = set()
if args.output.exists():
    for line in args.output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(json.loads(line)["event_id"])

load_started = time.perf_counter()
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
model = AutoModelForMultimodalLM.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
load_seconds = time.perf_counter() - load_started
revision = getattr(model.config, "_commit_hash", None)

mode = "a" if args.output.exists() else "w"
with args.output.open(mode, encoding="utf-8") as destination:
    for number, row in enumerate(manifest.itertuples(), start=1):
        if row.event_id in completed:
            print(f"[{number}/{len(manifest)}] {row.event_id}: already complete")
            continue
        frame_paths = json.loads(row.frame_paths_json)
        content = [{"type": "image", "path": path} for path in frame_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.05,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        prediction = parse_prediction(raw_text)
        record = {
            "event_id": row.event_id, "model_key": args.model_key,
            "model_id": args.model_id, "model_revision": revision,
            **prediction, "raw_text": raw_text,
            "inference_sec": elapsed,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        destination.flush()
        print(
            f"[{number}/{len(manifest)}] {row.event_id}: "
            f"{prediction['verb']} / {prediction['object']} | "
            f"{elapsed:.2f}s, {record['peak_vram_mib']:.0f} MiB, {prediction['parse_status']}"
        )
        del inputs, generated, new_tokens
        gc.collect()

metadata = {
    "model_key": args.model_key, "model_id": args.model_id,
    "model_revision": revision, "load_seconds": load_seconds,
    "clips_expected": len(manifest), "predictions_file": str(args.output),
}
args.output.with_suffix(".metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(metadata, indent=2, ensure_ascii=False))
