# COCO-Ghost Multi-Model Experiment Summary

## Model-Level Metrics

| Model | Samples | Original YES | Crop YES | Masked YES | Ghost Object Rate | Guard Accept | Guard Abstain | Prevented Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llava:7b | 50 | 52.0% | 74.0% | 32.0% | 61.5% | 18.0% | 34.0% | 100.0% |
| qwen2.5vl:3b | 50 | 92.0% | 86.0% | 74.0% | 78.3% | 14.0% | 78.0% | 100.0% |
| qwen2.5vl:7b | 50 | 100.0% | 96.0% | 84.0% | 84.0% | 16.0% | 84.0% | 100.0% |

## Interpretation

A high Ghost Object Rate means the model continues to report a target object after the target object's COCO mask has been removed. GHOST-Guard should be interpreted as a safety wrapper: it converts unsupported or suspicious object-presence claims into abstentions, rather than improving the underlying model.

## Artifacts

- `model_summary_metrics.csv` contains all model-level metrics.
- `model_category_metrics.csv` contains per-category metrics by model.
- `model_metric_comparison.png` compares aggregate rates across models.
- `category_ghost_rate_comparison.png` compares ghost rates by category and model.
