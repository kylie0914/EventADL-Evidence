"""Second-pass visual verification of action candidates proposed by Qwen3-VL.

The first VLM proposes up to four actions and approximate positions.  This
script turns each proposal into a narrow yes/no visual question.  Rejected and
uncertain claims remain in the audit JSONL but must not enter the search index.
"""
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


def parse_json(text: str) -> tuple[dict, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        value = json.loads(match.group(0) if match else text)
        verdict = str(value.get("verdict", "uncertain")).lower()
        if verdict not in {"verified", "rejected", "uncertain"}:
            verdict = "uncertain"
        start = min(1.0, max(0.0, float(value.get("start_fraction", 0.0))))
        end = min(1.0, max(start, float(value.get("end_fraction", 1.0))))
        return {
            "verdict": verdict,
            "evidence": str(value.get("evidence", "")).strip(),
            "corrected_verb": str(value.get("corrected_verb", "unknown")).strip().lower(),
            "corrected_object": str(value.get("corrected_object", "unknown")).strip().lower(),
            "start_fraction": start,
            "end_fraction": end,
            "confidence": min(1.0, max(0.0, float(value.get("confidence", 0.0)))),
        }, "valid_json"
    except Exception:
        return {
            "verdict": "uncertain", "evidence": text,
            "corrected_verb": "unknown", "corrected_object": "unknown",
            "start_fraction": 0.0, "end_fraction": 1.0, "confidence": 0.0,
        }, "invalid_json"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", required=True, type=Path)
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
parser.add_argument("--padding-seconds", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=192)
parser.add_argument("--episode-id", action="append", help="Only verify selected episode ID(s); repeatable")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

manifest = pd.read_csv(args.manifest).set_index("event_id")
predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
if args.episode_id:
    predictions = [row for row in predictions if row.get("episode_id") in set(args.episode_id)]
args.output.parent.mkdir(parents=True, exist_ok=True)
completed: dict[str, tuple[str, str]] = {}
if args.output.exists():
    for line in args.output.read_text().splitlines():
        if not line.strip():
            continue
        previous = json.loads(line)
        completed[previous["candidate_id"]] = (
            str(previous.get("claimed_verb", "")), str(previous.get("claimed_object", ""))
        )

processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
model = AutoModelForMultimodalLM.from_pretrained(
    args.model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

with args.output.open("a" if args.output.exists() else "w", encoding="utf-8") as destination:
    candidate_number = 0
    for prediction in predictions:
        segment_id = prediction["event_id"]
        if segment_id not in manifest.index:
            continue
        row = manifest.loc[segment_id]
        paths = json.loads(row.frame_paths_json)
        duration = max(0.001, float(row.end_sec) - float(row.start_sec))
        actions = prediction.get("visible_actions") or [{
            "verb": prediction.get("verb", "unknown"),
            "object": prediction.get("object", "unknown"),
            "description": prediction.get("description", ""),
            "start_fraction": 0.0, "end_fraction": 1.0,
        }]
        for action_index, action in enumerate(actions):
            candidate_number += 1
            candidate_id = f"{segment_id}_candidate_{action_index:02d}"
            current_claim = (str(action.get("verb", "unknown")), str(action.get("object", "unknown")))
            if completed.get(candidate_id) == current_claim:
                print(f"{candidate_id}: already complete")
                continue
            start = min(1.0, max(0.0, float(action.get("start_fraction", 0.0))))
            end = min(1.0, max(start, float(action.get("end_fraction", 1.0))))
            padding = args.padding_seconds / duration
            view_start, view_end = max(0.0, start - padding), min(1.0, end + padding)
            first = max(0, int(view_start * len(paths)))
            last = min(len(paths), max(first + 2, int(view_end * len(paths)) + 1))
            evidence_paths = paths[first:last]
            claim = f"{action.get('verb', 'unknown')} {action.get('object', 'unknown')}"
            prompt = f"""You are the second-pass visual verifier. The first model claimed: {claim}.
Inspect these chronological egocentric frames and decide whether that exact action is directly visible.
Do not accept the claim merely because a related object is present. Require visible hand motion and object
state evidence across multiple frames. For cut/slice/chop specifically, require a blade contacting and
passing through material; opening, tearing, squeezing, pressing, mixing, or handling a bag is not cutting.
Use rejected when a different action is clearly visible. Use uncertain when visibility is insufficient.
If verified, refine its start/end within the displayed frames as fractions from 0.0 to 1.0.
In evidence, describe only what is positively visible. Do not repeat the rejected action name and do not use
phrases such as "rather than cutting", because this text will be audited separately from retrieval.
Return JSON only:
{{"verdict":"verified|rejected|uncertain","evidence":"short visible reason","corrected_verb":"verb","corrected_object":"object","start_fraction":0.0,"end_fraction":1.0,"confidence":0.0}}"""
            messages = [{"role": "user", "content": [
                *[{"type": "image", "path": path} for path in evidence_paths],
                {"type": "text", "text": prompt},
            ]}]
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
            raw = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()
            result, parse_status = parse_json(raw)
            # Convert the verifier's fractions in the narrowed view back to the full chunk.
            full_start = view_start + result["start_fraction"] * (view_end - view_start)
            full_end = view_start + result["end_fraction"] * (view_end - view_start)
            record = {
                "candidate_id": candidate_id, "event_id": segment_id,
                "episode_id": prediction["episode_id"], "action_index": action_index,
                "claimed_verb": str(action.get("verb", "unknown")),
                "claimed_object": str(action.get("object", "unknown")),
                "claimed_description": str(action.get("description", "")),
                **result, "start_fraction": full_start, "end_fraction": full_end,
                "parse_status": parse_status, "frames_reviewed": len(evidence_paths),
                "raw_text": raw, "inference_sec": elapsed,
                "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"{candidate_id}: {result['verdict']} ({result['confidence']:.2f}) | {elapsed:.2f}s")
            del inputs, generated
            gc.collect()
