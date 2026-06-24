from __future__ import annotations

import argparse
import json
import time
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

from .common import add_config_arg, ensure_dirs, load_config, project_path
from .ollama_client import ask_ollama
from .parse_outputs import _extract_json_object


LABELS = {"clean_removal", "minor_residual_shape", "major_residual_shape", "object_still_visible", "parse_fail"}


def artifact_prompt(object_name: str) -> str:
    return f"""You are auditing a counterfactual image where a target object was removed by masking and blur-fill.

Target object: "{object_name}"

Look only at the image. Decide whether the target object or an object-shaped residual artifact is still visible.

Labels:
- clean_removal: the target object is not visible and the masked region does not preserve a clear target-object shape.
- minor_residual_shape: a faint or ambiguous object-shaped blur remains, but the object itself is not visibly present.
- major_residual_shape: the masked region clearly preserves the target object's silhouette or strong visual evidence.
- object_still_visible: the target object itself is still visibly present.

Return strict JSON only:
{{
  "artifact_label": "clean_removal" | "minor_residual_shape" | "major_residual_shape" | "object_still_visible",
  "artifact_score": 0 | 1 | 2 | 3,
  "rationale": "short visible-evidence phrase"
}}"""


def parse_artifact_response(raw: str) -> dict[str, Any]:
    obj = _extract_json_object(raw or "")
    if isinstance(obj, dict):
        label = str(obj.get("artifact_label", "")).strip().lower()
        if label in LABELS:
            try:
                score = int(obj.get("artifact_score", 0))
            except (TypeError, ValueError):
                score = {"clean_removal": 0, "minor_residual_shape": 1, "major_residual_shape": 2, "object_still_visible": 3}.get(label, 0)
            return {
                "artifact_label": label,
                "artifact_score": max(0, min(3, score)),
                "artifact_parse_ok": True,
                "artifact_rationale": str(obj.get("rationale", "")).strip(),
                "artifact_raw_response": raw,
            }
    lower = (raw or "").lower()
    for label in ["object_still_visible", "major_residual_shape", "minor_residual_shape", "clean_removal"]:
        if label in lower:
            return {
                "artifact_label": label,
                "artifact_score": {"clean_removal": 0, "minor_residual_shape": 1, "major_residual_shape": 2, "object_still_visible": 3}[label],
                "artifact_parse_ok": True,
                "artifact_rationale": raw[:300],
                "artifact_raw_response": raw,
            }
    return {
        "artifact_label": "parse_fail",
        "artifact_score": 3,
        "artifact_parse_ok": False,
        "artifact_rationale": "",
        "artifact_raw_response": raw,
    }


def ask_with_retries(image_path: str, prompt: str, cfg: dict[str, Any]) -> tuple[str, float]:
    ollama = cfg["ollama"]
    retries = int(cfg["experiment"].get("retries", 2))
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            raw, _ = ask_ollama(
                image_path=image_path,
                prompt=prompt,
                model=ollama["model"],
                host=ollama["host"],
                temperature=ollama.get("temperature", 0),
                num_ctx=ollama.get("num_ctx", 2048),
                timeout=ollama.get("timeout_seconds", 180),
            )
            return raw, time.time() - start
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Artifact labeling request failed after {retries + 1} attempts: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Label counterfactual removal artifacts on masked images.")
    add_config_arg(parser)
    parser.add_argument("--out", default="outputs/artifact_labels.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)

    manifest_path = project_path(cfg.get("counterfactual", {}).get("output_dir", "outputs/counterfactuals"), "manifest.jsonl")
    if not manifest_path.exists():
        raise SystemExit(f"[ERROR] Missing counterfactual manifest: {manifest_path}")
    rows = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()]
    out_path = project_path(args.out)
    ensure_dirs(out_path.parent)

    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = {row["sample_id"]: row for row in existing.to_dict("records")}

    labeled = list(done.values())
    for sample in tqdm(rows, desc="artifact labels"):
        if sample["sample_id"] in done:
            continue
        prompt = artifact_prompt(sample["category_name"])
        raw, runtime = ask_with_retries(sample["masked_path"], prompt, cfg)
        parsed = parse_artifact_response(raw)
        labeled.append(
            {
                "sample_id": sample["sample_id"],
                "category_name": sample["category_name"],
                "masked_path": sample["masked_path"],
                "mask_fill": sample.get("mask_fill", cfg["counterfactual"].get("mask_fill", "")),
                "mask_dilation_pixels": sample.get("mask_dilation_pixels", cfg["counterfactual"].get("mask_dilation_pixels", 0)),
                "artifact_model": cfg["ollama"]["model"],
                "artifact_runtime_seconds": runtime,
                **parsed,
            }
        )
        pd.DataFrame(labeled).to_csv(out_path, index=False)
        time.sleep(float(cfg["experiment"].get("sleep_between_calls_seconds", 0.2)))

    pd.DataFrame(labeled).to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
