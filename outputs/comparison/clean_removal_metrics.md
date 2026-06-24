# Clean-Removal Metrics

## Artifact Label Counts

| Label | Count |
|---|---:|
| clean_removal | 200 |

## Ghost Object Rate by Subset

| Model | All samples | Clean removal | Minor or clean | Artifact or visible |
|---|---:|---:|---:|---:|
| llava:7b | 33.7% | 33.7% | 33.7% | 0.0% |
| qwen2.5vl:3b | 47.5% | 47.5% | 47.5% | 0.0% |
| qwen2.5vl:7b | 51.6% | 51.6% | 51.6% | 0.0% |

Clean-removal metrics should be treated as the most conservative estimate of context-induced persistence. Artifact-or-visible metrics are useful for diagnosing counterfactual construction quality but should not be overinterpreted as pure hallucination.
