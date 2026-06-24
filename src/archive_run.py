from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import add_config_arg, ensure_dirs, load_config, project_path


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive the current outputs as a named experiment run.")
    add_config_arg(parser)
    parser.add_argument("--run-name", required=True, help="Run directory name under outputs/runs.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_dir = project_path("outputs", "runs", args.run_name)
    ensure_dirs(run_dir)
    for name in ["results", "figures", "reports"]:
        src = project_path("outputs", name)
        if src.exists():
            copy_dir(src, run_dir / name)

    metadata = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": args.config,
        "model": cfg["ollama"]["model"],
        "split": cfg["dataset"]["split"],
        "max_images_per_category": cfg["dataset"]["max_images_per_category"],
        "categories": cfg["dataset"]["categories"],
    }
    decisions_path = run_dir / "results" / "guard_decisions.csv"
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path)
        metadata["samples"] = int(len(decisions))
    answers_path = run_dir / "results" / "view_answers.csv"
    if answers_path.exists():
        answers = pd.read_csv(answers_path)
        metadata["view_answers"] = int(len(answers))

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Archived run to {run_dir}")


if __name__ == "__main__":
    main()
