from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .evaluate import compute_metrics
from .common import ensure_dirs, project_path


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def run_metrics(run_name: str, labels: pd.DataFrame) -> list[dict[str, object]]:
    run_dir = project_path("outputs", "runs", run_name)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    decisions = pd.read_csv(run_dir / "results" / "guard_decisions.csv")
    merged = decisions.merge(labels, on=["sample_id", "category_name"], how="left")
    subsets = {
        "all_samples": merged,
        "clean_removal": merged[merged.artifact_label == "clean_removal"],
        "minor_or_clean": merged[merged.artifact_label.isin(["clean_removal", "minor_residual_shape"])],
        "artifact_or_visible": merged[merged.artifact_label.isin(["major_residual_shape", "object_still_visible", "parse_fail"])],
    }
    rows: list[dict[str, object]] = []
    for subset_name, subset in subsets.items():
        if subset.empty:
            continue
        for metric in compute_metrics(subset):
            rows.append(
                {
                    "run_name": run_name,
                    "model": metadata["model"],
                    "subset": subset_name,
                    "subset_n": len(subset),
                    **metric,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report metrics separately by artifact/removal label.")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", default="outputs/artifact_labels.csv")
    args = parser.parse_args()

    labels_path = project_path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f"[ERROR] Missing artifact labels: {labels_path}")
    labels = pd.read_csv(labels_path)
    out_dir = project_path("outputs", "comparison")
    ensure_dirs(out_dir)

    rows = []
    for run_name in args.runs:
        rows.extend(run_metrics(run_name, labels))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "clean_removal_metrics.csv", index=False)

    ghost = df[df.metric == "ghost_object_rate"].pivot_table(index="model", columns="subset", values="value", aggfunc="first")
    order = [c for c in ["all_samples", "clean_removal", "minor_or_clean", "artifact_or_visible"] if c in ghost.columns]
    ax = ghost[order].plot(kind="bar", figsize=(10, 5.4), width=0.72)
    ax.set_title("Ghost Object Rate by Removal Quality Subset")
    ax.set_ylabel("Ghost Object Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Subset", loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_dir / "clean_removal_ghost_rate.png", dpi=180)
    plt.close()

    counts = labels.artifact_label.value_counts().rename_axis("artifact_label").reset_index(name="count")
    counts.to_csv(out_dir / "artifact_label_counts.csv", index=False)

    lines = [
        "# Clean-Removal Metrics",
        "",
        "## Artifact Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for _, row in counts.iterrows():
        lines.append(f"| {row.artifact_label} | {int(row['count'])} |")

    lines.extend(["", "## Ghost Object Rate by Subset", "", "| Model | All samples | Clean removal | Minor or clean | Artifact or visible |", "|---|---:|---:|---:|---:|"])
    for model, group in df[df.metric == "ghost_object_rate"].groupby("model", sort=True):
        vals = group.set_index("subset")["value"].to_dict()
        lines.append(
            f"| {model} | {pct(vals.get('all_samples', 0))} | {pct(vals.get('clean_removal', 0))} | "
            f"{pct(vals.get('minor_or_clean', 0))} | {pct(vals.get('artifact_or_visible', 0))} |"
        )

    lines.extend(
        [
            "",
            "Clean-removal metrics should be treated as the most conservative estimate of context-induced persistence. Artifact-or-visible metrics are useful for diagnosing counterfactual construction quality but should not be overinterpreted as pure hallucination.",
            "",
        ]
    )
    (out_dir / "clean_removal_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote clean-removal report to {out_dir}")


if __name__ == "__main__":
    main()
