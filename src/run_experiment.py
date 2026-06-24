from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

from .common import add_config_arg, ensure_dirs, load_config, project_path
from .ollama_client import ask_ollama
from .parse_outputs import parse_presence_response
from .prompts import object_presence_prompt


VIEW_PATH_COLUMNS = {
    "original": "original_path",
    "crop": "crop_path",
    "masked": "masked_path",
}


def ask_with_retries(image_path: str, prompt: str, cfg: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    ollama = cfg["ollama"]
    retries = int(cfg["experiment"].get("retries", 2))
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        start = time.time()
        try:
            raw, meta = ask_ollama(
                image_path=image_path,
                prompt=prompt,
                model=ollama["model"],
                host=ollama["host"],
                temperature=ollama.get("temperature", 0),
                num_ctx=ollama.get("num_ctx", 2048),
                timeout=ollama.get("timeout_seconds", 180),
            )
            return raw, meta, time.time() - start
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Ollama request failed after {retries + 1} attempts: {last_error}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM object-presence prompts over COCO-Ghost views.")
    add_config_arg(parser)
    parser.add_argument("--self_localization", action="store_true", help="Reserved for future self-localized guard mode.")
    args = parser.parse_args()
    if args.self_localization:
        raise SystemExit("[ERROR] Self-localization mode is planned but not implemented in the first oracle-mask pilot.")

    cfg = load_config(args.config)
    manifest_path = project_path("outputs", "counterfactuals", "manifest.jsonl")
    out_dir = project_path("outputs", "results")
    ensure_dirs(out_dir)
    if not manifest_path.exists():
        raise SystemExit(f"[ERROR] Missing counterfactual manifest: {manifest_path}")

    rows = [json.loads(line) for line in open(manifest_path, "r", encoding="utf-8") if line.strip()]
    views = cfg["experiment"].get("views", ["original", "crop", "masked"])
    raw_rows: list[dict[str, Any]] = []
    parsed_rows: list[dict[str, Any]] = []

    for sample in tqdm(rows, desc="vlm calls"):
        prompt = object_presence_prompt(sample["category_name"])
        for view in views:
            path_col = VIEW_PATH_COLUMNS.get(view)
            if not path_col:
                raise SystemExit(f"[ERROR] Unknown view: {view}")
            image_path = sample[path_col]
            raw, meta, runtime = ask_with_retries(image_path, prompt, cfg)
            parsed = parse_presence_response(raw)
            base = {
                "sample_id": sample["sample_id"],
                "image_id": sample["image_id"],
                "annotation_id": sample["annotation_id"],
                "category_name": sample["category_name"],
                "view": view,
                "image_path": image_path,
                "prompt": prompt,
                "model": cfg["ollama"]["model"],
                "runtime_seconds": runtime,
            }
            raw_rows.append({**base, "raw_response": raw, "ollama_metadata": meta})
            parsed_rows.append(
                {
                    **base,
                    "answer": parsed["answer"],
                    "evidence": parsed["evidence"],
                    "parse_ok": parsed["parse_ok"],
                    "raw_response": parsed["raw"],
                }
            )
            time.sleep(float(cfg["experiment"].get("sleep_between_calls_seconds", 0.2)))

    raw_path = out_dir / "raw_vlm_outputs.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    answers_path = out_dir / "view_answers.csv"
    pd.DataFrame(parsed_rows).to_csv(answers_path, index=False)
    print(f"[OK] Wrote {raw_path}")
    print(f"[OK] Wrote {answers_path}")


if __name__ == "__main__":
    main()
