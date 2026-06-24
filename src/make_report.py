from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import add_config_arg, ensure_dirs, load_config, project_path


def metric_value(summary: pd.DataFrame, metric: str) -> float:
    row = summary[summary.metric == metric]
    return float(row.iloc[0].value) if not row.empty else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def category_table(category_metrics: pd.DataFrame) -> str:
    if category_metrics.empty:
        return "_No category metrics available._"
    wanted = ["original_yes_rate", "crop_yes_rate", "ghost_object_rate", "guard_abstention_rate"]
    pivot = category_metrics[category_metrics.metric.isin(wanted)].pivot_table(
        index="category_name", columns="metric", values="value", aggfunc="first"
    )
    pivot = pivot.reset_index()
    labels = {
        "category_name": "Category",
        "original_yes_rate": "Original YES",
        "crop_yes_rate": "Crop YES",
        "ghost_object_rate": "Ghost Object Rate",
        "guard_abstention_rate": "Guard Abstention",
    }
    lines = ["| Category | Original YES | Crop YES | Ghost Object Rate | Guard Abstention |", "|---|---:|---:|---:|---:|"]
    for _, row in pivot.iterrows():
        lines.append(
            "| {category} | {original} | {crop} | {ghost} | {abstain} |".format(
                category=row["category_name"],
                original=pct(row.get("original_yes_rate", 0.0)),
                crop=pct(row.get("crop_yes_rate", 0.0)),
                ghost=pct(row.get("ghost_object_rate", 0.0)),
                abstain=pct(row.get("guard_abstention_rate", 0.0)),
            )
        )
    return "\n".join(lines)


def example_failures(decisions: pd.DataFrame) -> str:
    failures = decisions[(decisions.ghost_flag == True) & (decisions.final_answer == "UNSURE")].head(5)
    if failures.empty:
        return "_No ghost-risk examples were found in this run._"
    lines = []
    for _, row in failures.iterrows():
        lines.append(
            f"- `{row.sample_id}` ({row.category_name}): original={row.original_answer}, "
            f"crop={row.crop_answer}, masked={row.masked_answer}, final={row.final_answer}, risk={row.risk}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create markdown pilot report.")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = project_path("outputs", "reports")
    ensure_dirs(out_dir)

    summary_path = project_path("outputs", "results", "summary_metrics.csv")
    category_path = project_path("outputs", "results", "category_metrics.csv")
    decisions_path = project_path("outputs", "results", "guard_decisions.csv")
    for path in [summary_path, category_path, decisions_path]:
        if not path.exists():
            raise SystemExit(f"[ERROR] Missing required result file: {path}")

    summary = pd.read_csv(summary_path)
    category_metrics_df = pd.read_csv(category_path)
    decisions = pd.read_csv(decisions_path)
    ghost_rate = metric_value(summary, "ghost_object_rate")
    prevented_rate = metric_value(summary, "prevented_ghost_claim_rate")
    model = cfg["ollama"]["model"]
    categories = ", ".join(cfg["dataset"]["categories"])

    report = f"""# COCO-Ghost + GHOST-Guard Pilot Report

## Summary

This pilot evaluates whether a local vision-language model continues to report an object after the object has been removed from a natural image using COCO instance masks. We define this behavior as a ghost object claim. We then apply GHOST-Guard, a training-free evidence gate that accepts an object claim only when it is supported by the object crop and disappears after object removal.

## Main Result

The baseline Ghost Object Rate was {pct(ghost_rate)}. This means that, among images where the model originally reported the target object, it still reported the object after removal in {pct(ghost_rate)} of cases.

After applying GHOST-Guard, {pct(prevented_rate)} of these suspicious claims were downgraded to UNSURE rather than accepted as true object detections.

## Dataset

- COCO split: `{cfg["dataset"]["split"]}`
- Target categories: {categories}
- Samples evaluated: {len(decisions)}
- Images per category target: {cfg["dataset"]["max_images_per_category"]}

## Model

- Ollama model: `{model}`
- Temperature: {cfg["ollama"].get("temperature", 0)}
- Runtime mode: local black-box REST calls through Ollama

## Methods

For each selected COCO instance, COCO-Ghost creates three views: the original image, a crop around the target object, and a masked image where the target object's segmentation mask is replaced by blurred local background. The same strict object-presence prompt is sent for each view. GHOST-Guard accepts the object claim only if the model answers YES on the original image, YES on the crop, and NO on the masked image.

## Metrics

| Metric | Value |
|---|---:|
| Original YES Rate | {pct(metric_value(summary, "original_yes_rate"))} |
| Crop YES Rate | {pct(metric_value(summary, "crop_yes_rate"))} |
| Masked YES Rate | {pct(metric_value(summary, "masked_yes_rate"))} |
| Ghost Object Rate | {pct(ghost_rate)} |
| Context Persistence Index | {metric_value(summary, "context_persistence_index"):.3f} |
| GHOST-Guard Acceptance Rate | {pct(metric_value(summary, "guard_acceptance_rate"))} |
| GHOST-Guard Abstention Rate | {pct(metric_value(summary, "guard_abstention_rate"))} |
| Prevented Ghost Claim Rate | {pct(prevented_rate)} |

## Category Findings

{category_table(category_metrics_df)}

## Example Failures

{example_failures(decisions)}

## Interpretation

A high Ghost Object Rate suggests that the model is not relying purely on visible object evidence. Instead, it may be using scene priors, co-occurring objects, or activity context. For example, a model may continue to report a tennis racket when a person remains on a tennis court, even after the racket region has been removed.

Case patterns should be read as follows:

- High original YES and high masked YES indicates context-induced object persistence.
- High original YES and low crop YES suggests weak localized evidence.
- High crop YES and low masked YES is the desired visually grounded behavior.
- Low original YES means the model failed basic recognition for that category, so ghost behavior is less interpretable.
- Masked YES cases should be inspected visually for object-shaped masking artifacts.

## Significance

This result supports the need for visual evidence gates in applications where unsupported object claims are costly. The proposed guard is training-free, local, and compatible with black-box VLMs because it only requires repeated image-question calls.

## Novelty vs SOTA

COCO-Ghost is not claiming that object hallucination is new. CHAIR measures hallucinated objects in generated captions; POPE probes object existence in real images; AMBER covers broader hallucination categories; and methods such as Visual Contrastive Decoding, OPERA, Woodpecker, and object-aligned contrastive decoding target decoding or post-remedy behavior. COCO-Ghost instead creates controlled object-removed counterfactuals and asks whether an object claim survives removal of the visual evidence itself. GHOST-Guard is a decision-level, black-box-compatible wrapper rather than a replacement for model-internal decoding methods.

## Limitations

This pilot uses a small COCO subset and object removal based on ground-truth masks. The masked region may introduce artifacts. Results should be interpreted as evidence of a measurable failure mode, not as a full benchmark of model reliability.

## Next Steps

The next version should evaluate more images, more object categories, multiple Ollama models, and a self-localization mode where the model proposes the suspected object region without using COCO masks. A future medical imaging extension could test whether disease claims persist after removing or blurring suspected visual evidence, but that should remain future work rather than part of this first natural-image prototype.
"""

    out_path = out_dir / "pilot_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
