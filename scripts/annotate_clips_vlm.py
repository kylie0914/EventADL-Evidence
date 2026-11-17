from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


PROMPT = """Label this egocentric ADL clip. Return exactly ONE flat JSON object and no other text:
{"verb":"...","object":"...","description":"...","uncertain":false,"evidence":"one short visible clue"}
The evidence value MUST be a string, never an array or another JSON object. The uncertain value MUST be true or false. Use unknown instead of guessing."""


def first_text(value: object) -> str:
    """Convert occasionally nested model output into one readable string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list) and value:
        return first_text(value[0])
    if isinstance(value, dict):
        for key in ("text", "evidence", "description"):
            if key in value:
                return first_text(value[key])
    return str(value).strip()


def as_boolean(value: object) -> bool:
    """Normalize boolean-like model strings without treating 'uncertain' as false."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "no", "0", "certain"}


def recover_flat_fields(text: str) -> dict[str, object] | None:
    """Recover the first flat fields when a model repeats JSON until truncation.

    SmolVLM2 can recursively repeat an object inside `evidence`. Even if the final
    closing brace is missing, the first verb/object/description remain usable and
    are explicitly marked as recovered rather than silently called valid JSON.
    """
    fields: dict[str, str] = {}
    for key in ("verb", "object", "description"):
        match = re.search(rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
        if match:
            fields[key] = bytes(match.group(1), "utf-8").decode("unicode_escape")
    uncertain_match = re.search(r'"uncertain"\s*:\s*(true|false|"[^"]+")', text, re.IGNORECASE)
    if not {"verb", "object"}.issubset(fields):
        return None
    uncertain_raw = uncertain_match.group(1).strip('"') if uncertain_match else "true"
    return {
        "verb": fields["verb"],
        "object": fields["object"],
        "description": fields.get("description", ""),
        "uncertain": as_boolean(uncertain_raw),
        "evidence": fields.get("description", "recovered from truncated output"),
        "parse_status": "recovered_truncated_json",
    }


def parse_json(text: str) -> dict[str, object]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise TypeError("VLM output is not a JSON object")
        return {
            "verb": first_text(payload.get("verb", "unknown")) or "unknown",
            "object": first_text(payload.get("object", "unknown")) or "unknown",
            "description": first_text(payload.get("description", "")),
            "uncertain": as_boolean(payload.get("uncertain", True)),
            "evidence": first_text(payload.get("evidence", "")),
            "parse_status": "valid_json",
        }
    except (json.JSONDecodeError, TypeError):
        recovered = recover_flat_fields(cleaned)
        if recovered:
            return recovered
        return {"verb": "unknown", "object": "unknown", "description": text, "uncertain": True, "evidence": "invalid_json", "parse_status": "invalid_json"}


parser = argparse.ArgumentParser()
parser.add_argument("--clips-dir", required=True)
parser.add_argument("--model-id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--output", required=True)
args = parser.parse_args()
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8") as handle:
    for clip in sorted(Path(args.clips_dir).glob("*.mp4")):
        messages = [{"role": "user", "content": [{"type": "video", "path": str(clip.resolve())}, {"type": "text", "text": PROMPT}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=160, repetition_penalty=1.08)
        elapsed = time.perf_counter() - started
        text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        label = parse_json(text)
        row = {"event_id": clip.stem, "clip": str(clip), "model": args.model_id, **label, "inference_sec": elapsed, "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2}
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(clip.name, label.get("verb"), label.get("object"))

