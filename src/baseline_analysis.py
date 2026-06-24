from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import ensure_dirs, project_path


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def analyze_run(run_name: str) -> dict[str, object]:
    run_dir = project_path("outputs", "runs", run_name)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    decisions = pd.read_csv(run_dir / "results" / "guard_decisions.csv")
    n = len(decisions)
    original_yes = decisions.original_answer == "YES"
    crop_yes = decisions.crop_answer == "YES"
    masked_yes = decisions.masked_answer == "YES"
    ghost = original_yes & masked_yes
    crop_accept = original_yes & crop_yes
    guard_accept = decisions.final_answer == "YES"
    return {
        "run_name": run_name,
        "model": metadata["model"],
        "samples": n,
        "original_only_accept_rate": safe_div(original_yes.sum(), n),
        "original_only_ghost_accept_rate": safe_div(ghost.sum(), original_yes.sum()),
        "crop_gate_accept_rate": safe_div(crop_accept.sum(), n),
        "crop_gate_ghost_accept_rate": safe_div((crop_accept & masked_yes).sum(), crop_accept.sum()),
        "ghost_guard_accept_rate": safe_div(guard_accept.sum(), n),
        "ghost_guard_ghost_accept_rate": safe_div((guard_accept & masked_yes).sum(), guard_accept.sum()),
        "ghost_guard_abstention_rate": safe_div((decisions.final_answer == "UNSURE").sum(), n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare simple acceptance baselines against GHOST-Guard.")
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    out_dir = project_path("outputs", "comparison")
    ensure_dirs(out_dir)
    df = pd.DataFrame([analyze_run(run) for run in args.runs])
    df.to_csv(out_dir / "baseline_safety_comparison.csv", index=False)

    lines = [
        "# Baseline Safety Comparison",
        "",
        "| Model | Original-only accept | Original-only ghost accepts | Crop-gate accept | Crop-gate ghost accepts | GHOST-Guard accept | GHOST-Guard ghost accepts | GHOST-Guard abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.sort_values("model").iterrows():
        lines.append(
            f"| {row.model} | {pct(row.original_only_accept_rate)} | {pct(row.original_only_ghost_accept_rate)} | "
            f"{pct(row.crop_gate_accept_rate)} | {pct(row.crop_gate_ghost_accept_rate)} | "
            f"{pct(row.ghost_guard_accept_rate)} | {pct(row.ghost_guard_ghost_accept_rate)} | "
            f"{pct(row.ghost_guard_abstention_rate)} |"
        )
    lines.extend(
        [
            "",
            "Original-only accepts every original YES claim. Crop-gate requires crop support but does not test whether the claim disappears after removal. GHOST-Guard adds the masked counterfactual check and therefore abstains on claims that survive visual evidence removal.",
            "",
        ]
    )
    (out_dir / "baseline_safety_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote baseline comparison to {out_dir}")


if __name__ == "__main__":
    main()
