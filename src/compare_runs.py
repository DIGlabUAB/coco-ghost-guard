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

from .common import ensure_dirs, project_path


KEY_METRICS = [
    "original_yes_rate",
    "crop_yes_rate",
    "masked_yes_rate",
    "ghost_object_rate",
    "guard_acceptance_rate",
    "guard_abstention_rate",
    "prevented_ghost_claim_rate",
]

METRIC_LABELS = {
    "original_yes_rate": "Original YES",
    "crop_yes_rate": "Crop YES",
    "masked_yes_rate": "Masked YES",
    "ghost_object_rate": "Ghost Rate",
    "guard_acceptance_rate": "Guard Accept",
    "guard_abstention_rate": "Guard Abstain",
    "prevented_ghost_claim_rate": "Prevented Ghosts",
}


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def read_run(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "results" / "summary_metrics.csv")
    category = pd.read_csv(run_dir / "results" / "category_metrics.csv")
    return metadata, summary, category


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare archived COCO-Ghost runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run names under outputs/runs.")
    args = parser.parse_args()

    out_dir = project_path("outputs", "comparison")
    ensure_dirs(out_dir)
    rows = []
    cat_rows = []
    for run_name in args.runs:
        run_dir = project_path("outputs", "runs", run_name)
        metadata, summary, category = read_run(run_dir)
        model = metadata["model"]
        for _, row in summary.iterrows():
            rows.append(
                {
                    "run_name": run_name,
                    "model": model,
                    "samples": metadata.get("samples", 0),
                    "metric": row["metric"],
                    "label": row["label"],
                    "value": row["value"],
                }
            )
        category = category.copy()
        category["run_name"] = run_name
        category["model"] = model
        cat_rows.extend(category.to_dict("records"))

    summary_df = pd.DataFrame(rows)
    category_df = pd.DataFrame(cat_rows)
    summary_df.to_csv(out_dir / "model_summary_metrics.csv", index=False)
    category_df.to_csv(out_dir / "model_category_metrics.csv", index=False)

    pivot = summary_df[summary_df.metric.isin(KEY_METRICS)].pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    order = [m for m in KEY_METRICS if m in pivot.columns]
    plot_df = pivot[order].rename(columns=METRIC_LABELS).T
    ax = plot_df.plot(kind="bar", figsize=(12, 5.6), width=0.78)
    ax.set_title("COCO-Ghost Metrics by Local VLM")
    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_dir / "model_metric_comparison.png", dpi=180)
    plt.close()

    ghost = category_df[category_df.metric == "ghost_object_rate"].pivot_table(
        index="category_name", columns="model", values="value", aggfunc="first"
    )
    ax = ghost.plot(kind="bar", figsize=(11, 5.6), width=0.78)
    ax.set_title("Ghost Object Rate by Category and Model")
    ax.set_ylabel("Ghost Object Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Model", loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out_dir / "category_ghost_rate_comparison.png", dpi=180)
    plt.close()

    lines = [
        "# COCO-Ghost Multi-Model Experiment Summary",
        "",
        "## Model-Level Metrics",
        "",
        "| Model | Samples | Original YES | Crop YES | Masked YES | Ghost Object Rate | Guard Accept | Guard Abstain | Prevented Ghost Claims |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, group in summary_df.groupby("model", sort=True):
        vals = group.set_index("metric")["value"].to_dict()
        samples = int(group.samples.max())
        lines.append(
            f"| {model} | {samples} | {pct(vals.get('original_yes_rate', 0))} | "
            f"{pct(vals.get('crop_yes_rate', 0))} | {pct(vals.get('masked_yes_rate', 0))} | "
            f"{pct(vals.get('ghost_object_rate', 0))} | {pct(vals.get('guard_acceptance_rate', 0))} | "
            f"{pct(vals.get('guard_abstention_rate', 0))} | {pct(vals.get('prevented_ghost_claim_rate', 0))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A high Ghost Object Rate means the model continues to report a target object after the target object's COCO mask has been removed. GHOST-Guard should be interpreted as a safety wrapper: it converts unsupported or suspicious object-presence claims into abstentions, rather than improving the underlying model.",
            "",
            "## Artifacts",
            "",
            "- `model_summary_metrics.csv` contains all model-level metrics.",
            "- `model_category_metrics.csv` contains per-category metrics by model.",
            "- `model_metric_comparison.png` compares aggregate rates across models.",
            "- `category_ghost_rate_comparison.png` compares ghost rates by category and model.",
            "",
        ]
    )
    (out_dir / "multi_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
