# Baseline Safety Comparison

| Model | Original-only accept | Original-only ghost accepts | Crop-gate accept | Crop-gate ghost accepts | GHOST-Guard accept | GHOST-Guard ghost accepts | GHOST-Guard abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| llava:7b | 43.0% | 33.7% | 35.0% | 30.0% | 24.5% | 0.0% | 18.5% |
| qwen2.5vl:3b | 88.5% | 47.5% | 76.0% | 44.7% | 30.0% | 0.0% | 58.5% |
| qwen2.5vl:7b | 96.0% | 51.6% | 91.5% | 51.4% | 44.5% | 0.0% | 51.5% |

Original-only accepts every original YES claim. Crop-gate requires crop support but does not test whether the claim disappears after removal. GHOST-Guard adds the masked counterfactual check and therefore abstains on claims that survive visual evidence removal.
