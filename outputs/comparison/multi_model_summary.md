# COCO-Ghost Multi-Model Experiment Summary

## Model-Level Metrics

| Model | Samples | Original YES | Crop YES | Masked YES | Ghost Object Rate | Guard Accept | Guard Abstain | Prevented Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava:7b | 200 | 43.0% | 60.5% | 14.5% | 33.7% | 24.5% | 18.5% | 100.0% |
| qwen2.5vl:3b | 200 | 88.5% | 80.0% | 43.0% | 47.5% | 30.0% | 58.5% | 100.0% |
| qwen2.5vl:7b | 200 | 96.0% | 94.5% | 49.5% | 51.6% | 44.5% | 51.5% | 100.0% |

## Interpretation

A high Ghost Object Rate means the model continues to report a target object after the target object's COCO mask has been removed. GHOST-Guard should be interpreted as a safety wrapper: it converts unsupported or suspicious object-presence claims into abstentions, rather than improving the underlying model.

## Artifacts

- `model_summary_metrics.csv` contains all model-level metrics.
- `model_category_metrics.csv` contains per-category metrics by model.
- `model_metric_comparison.png` compares aggregate rates across models.
- `category_ghost_rate_comparison.png` compares ghost rates by category and model.
