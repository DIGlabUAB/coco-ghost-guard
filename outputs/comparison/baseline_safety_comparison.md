# Baseline Safety Comparison

| Model | Original-only accept | Original-only ghost accepts | Crop-gate accept | Crop-gate ghost accepts | GHOST-Guard accept | GHOST-Guard ghost accepts | GHOST-Guard abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava:7b | 52.0% | 61.5% | 44.0% | 59.1% | 18.0% | 0.0% | 34.0% |
| qwen2.5vl:3b | 92.0% | 78.3% | 84.0% | 78.6% | 14.0% | 0.0% | 78.0% |
| qwen2.5vl:7b | 100.0% | 84.0% | 96.0% | 83.3% | 16.0% | 0.0% | 84.0% |

Original-only accepts every original YES claim. Crop-gate requires crop support but does not test whether the claim disappears after removal. GHOST-Guard adds the masked counterfactual check and therefore abstains on claims that survive visual evidence removal.
