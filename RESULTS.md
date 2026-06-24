# Results

This repository has been run as an expanded local experiment on 50 COCO instances across 10 object categories and 3 local Ollama vision-language models.

Each sample has three views:

- `original`: the full original COCO image.
- `crop`: a crop around the target object.
- `masked`: the same image after the target object's COCO segmentation mask is filled with blurred local pixels.

GHOST-Guard accepts an object claim only when:

```text
original = YES
crop = YES
masked = NO
```

If the model continues to answer `YES` after object removal, the claim is treated as a ghost-risk case and downgraded to `UNSURE`.

## Expanded Multi-Model Summary

| Model | Samples | Original YES | Crop YES | Masked YES | Ghost Object Rate | Guard Accept | Guard Abstain | Prevented Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llava:7b` | 50 | 52.0% | 74.0% | 32.0% | 61.5% | 18.0% | 34.0% | 100.0% |
| `qwen2.5vl:3b` | 50 | 92.0% | 86.0% | 74.0% | 78.3% | 14.0% | 78.0% | 100.0% |
| `qwen2.5vl:7b` | 50 | 100.0% | 96.0% | 84.0% | 84.0% | 16.0% | 84.0% | 100.0% |

## Interpretation

The expanded experiment shows that context-induced object persistence is measurable across multiple local VLMs, but the effect varies by model.

`qwen2.5vl:7b` had the strongest object recognition on original images, but also the highest Ghost Object Rate. This is a useful finding: a model can be better at recognizing visible objects while still preserving object claims after the object is removed.

`qwen2.5vl:3b` showed slightly weaker original/crop recognition and a lower Ghost Object Rate than the 7B version, but still produced many masked-image YES claims.

`llava:7b` had lower original recognition and a lower masked YES rate. Its ghost rate is therefore less directly comparable as a pure persistence measure, because it failed more often on the original image.

GHOST-Guard prevented 100% of detected ghost claims in all three runs by downgrading them to `UNSURE`. This should be framed as abstention behavior, not as model improvement.

## Baseline Safety Comparison

| Model | Original-only accept | Original-only ghost accepts | Crop-gate accept | Crop-gate ghost accepts | GHOST-Guard accept | GHOST-Guard ghost accepts | GHOST-Guard abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llava:7b` | 52.0% | 61.5% | 44.0% | 59.1% | 18.0% | 0.0% | 34.0% |
| `qwen2.5vl:3b` | 92.0% | 78.3% | 84.0% | 78.6% | 14.0% | 0.0% | 78.0% |
| `qwen2.5vl:7b` | 100.0% | 84.0% | 96.0% | 83.3% | 16.0% | 0.0% | 84.0% |

This baseline comparison is important. A crop-only gate confirms that the model can recognize the object in a localized view, but it still accepts most ghost claims because it never asks whether the claim disappears after object removal. GHOST-Guard adds the masked counterfactual check, so detected ghost claims are not accepted.

## Generated Artifacts

- `outputs/runs/qwen2_5vl_7b/`
- `outputs/runs/qwen2_5vl_3b/`
- `outputs/runs/llava_7b/`
- `outputs/comparison/model_summary_metrics.csv`
- `outputs/comparison/model_category_metrics.csv`
- `outputs/comparison/baseline_safety_comparison.csv`
- `outputs/comparison/model_metric_comparison.png`
- `outputs/comparison/category_ghost_rate_comparison.png`
- `outputs/comparison/multi_model_summary.md`

## Publication Claim

The strongest defensible claim is:

> COCO-Ghost exposes context-induced object persistence in local VLMs using controlled object-removal counterfactuals, and GHOST-Guard provides a black-box abstention wrapper that refuses object-presence claims when they survive removal of the visual evidence.

Avoid claiming that GHOST-Guard solves hallucination. It catches and abstains on one interpretable class of unsupported object-presence claims.

## Remaining Work Before Submission

- Complete the visual artifact audit for masked regions. A first-pass note is in `ARTIFACT_AUDIT.md`.
- Run at least one larger setting, ideally 200-500 samples.
- Add a stronger baseline section comparing original-only, crop-only, masked-only rejection, and GHOST-Guard.
- Add references and related-work text around CHAIR, POPE, AMBER, VCD, OPERA, Woodpecker, visual prompt engineering, and counterfactual segmentation reasoning.
- Manually inspect representative contact sheets before making strong claims about removal quality.
