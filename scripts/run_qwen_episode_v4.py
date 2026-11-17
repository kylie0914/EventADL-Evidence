"""Label adaptive episode segments with Qwen3-VL and structured JSON output."""
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


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
        actions = []
        for item in value.get("visible_actions", [])[:4]:
            if not isinstance(item, dict):
                continue
            try:
                start = min(1.0, max(0.0, float(item.get("start_fraction", 0.0))))
                end = min(1.0, max(start, float(item.get("end_fraction", 1.0))))
            except (TypeError, ValueError):
                start, end = 0.0, 1.0
            actions.append({
                "verb": str(item.get("verb", "unknown")).strip().lower(),
                "object": str(item.get("object", "unknown")).strip().lower(),
                "description": str(item.get("description", "")).strip(),
                "start_fraction": start, "end_fraction": end,
            })
        return {
            "verb": str(value.get("verb", "unknown")).strip().lower(),
            "object": str(value.get("object", "unknown")).strip().lower(),
            "description": str(value.get("description", "")).strip(),
            "uncertain": bool(value.get("uncertain", False)),
            "visible_actions": actions,
            "parse_status": "valid_json",
        }
    except Exception:
        return {"verb": "unknown", "object": "unknown", "description": text, "uncertain": True, "visible_actions": [], "parse_status": "invalid_json"}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
parser.add_argument("--max-new-tokens", type=int, default=384)
parser.add_argument("--episode-id", action="append", help="Only process selected episode ID(s); repeatable")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
manifest = pd.read_csv(args.manifest)
if args.episode_id:
    manifest = manifest[manifest.episode_id.isin(args.episode_id)].reset_index(drop=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
completed = set()
if args.output.exists():
    completed = {json.loads(line)["event_id"] for line in args.output.read_text().splitlines() if line.strip()}

processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
model = AutoModelForMultimodalLM.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
prompt = """These are chronological frames sampled at 3 FPS from one egocentric hand-object interaction segment.
Describe only actions directly visible across the chronological frames. Do not infer an action from the room,
an object lying nearby, a cutting board, or a single ambiguous frame. List up to four distinct visible actions,
including brief actions even when they are not the longest action. Estimate each action's start/end as fractions
from 0.0 to 1.0 of this segment.

Strict distinction: label cut/slice/chop only when a blade is visibly contacting and passing through material
across multiple frames. Opening, tearing, pressing, kneading, or squeezing a bag is not cutting. If that evidence
is absent, do not use a cutting verb. Use unknown and uncertain=true when evidence is insufficient.

Also choose the main repeated hand action and its manipulated object.
The verb and object fields must contain actual observed labels such as squeeze/bag, not placeholder words.
Return exactly one compact JSON object and no markdown, using this schema:
{"verb":"<actual verb>","object":"<actual object>","description":"<short visible summary>","visible_actions":[{"verb":"<actual verb>","object":"<actual object>","description":"<visible evidence>","start_fraction":0.0,"end_fraction":0.4}],"uncertain":false}"""

with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as output:
    for number, row in enumerate(manifest.itertuples(), start=1):
        if row.event_id in completed:
            print(f"[{number}/{len(manifest)}] {row.event_id}: already complete")
            continue
        paths = json.loads(row.frame_paths_json)
        messages = [{"role": "user", "content": [
            *[{"type": "image", "path": path} for path in paths],
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, repetition_penalty=1.05)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        prediction = parse_json(text)
        record = {
            "event_id": row.event_id, "episode_id": row.episode_id,
            "chunk_index": int(row.chunk_index), "frame_count": int(row.frame_count),
            **prediction, "raw_text": text, "inference_sec": elapsed,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        print(f"[{number}/{len(manifest)}] {row.event_id}: {prediction['verb']} / {prediction['object']} | {elapsed:.2f}s, {record['peak_vram_mib']:.0f} MiB")
        del inputs, generated
        gc.collect()
