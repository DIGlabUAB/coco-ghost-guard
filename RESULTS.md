# Results

This repository has now been run as a scaled local experiment on 200 COCO validation instances across 20 object categories and 3 local Ollama vision-language models.

Each sample has three views:

- `original`: the full original COCO image.
- `crop`: a crop around the target object.
- `masked`: the same image after a padded target-object bounding box is replaced with solid local-mean color.

The scaled setting uses bbox occlusion rather than blur-fill segmentation. This choice is conservative: the masked image is less naturalistic, but it avoids object-shaped blur that could leak the target silhouette.

GHOST-Guard accepts an object claim only when:

```text
original = YES
crop = YES
masked = NO
```

If the model continues to answer `YES` after object removal, the claim is treated as a ghost-risk case and downgraded to `UNSURE`.

## Scaled Clean-Removal Summary

Artifact labels for the 200-sample bbox-solid counterfactual set:

| Artifact label | Count |
|---|---:|
| `clean_removal` | 200 |

Model results:

| Model | Samples | Original YES | Crop YES | Masked YES | Clean Ghost Rate | Guard Accept | Guard Abstain | Accepted Ghost Claims |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5vl:7b` | 200 | 96.0% | 94.5% | 49.5% | 51.6% | 44.5% | 51.5% | 0.0% |
| `qwen2.5vl:3b` | 200 | 88.5% | 80.0% | 43.0% | 47.5% | 30.0% | 58.5% | 0.0% |
| `llava:7b` | 200 | 43.0% | 60.5% | 14.5% | 33.7% | 24.5% | 18.5% | 0.0% |

## Interpretation

The scaled experiment shows that object-presence claims can persist after the target object has been removed by a conservative, non-silhouette occlusion. This makes the result more useful than the earlier blur-mask pilot, where visible residual shape was a serious confound.

`qwen2.5vl:7b` had the strongest original and crop recognition, but also the highest clean Ghost Object Rate. `qwen2.5vl:3b` showed the same pattern at a slightly lower rate. `llava:7b` had much weaker original recognition, so its lower ghost rate should be interpreted with that caveat.

GHOST-Guard prevented 100% of detected ghost claims in all three runs by downgrading them to `UNSURE`. This is abstention behavior, not model improvement.

## Baseline Safety Comparison

| Model | Original-only accept | Original-only ghost accepts | Crop-gate accept | Crop-gate ghost accepts | GHOST-Guard accept | GHOST-Guard ghost accepts | GHOST-Guard abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llava:7b` | 43.0% | 33.7% | 35.0% | 30.0% | 24.5% | 0.0% | 18.5% |
| `qwen2.5vl:3b` | 88.5% | 47.5% | 76.0% | 44.7% | 30.0% | 0.0% | 58.5% |
| `qwen2.5vl:7b` | 96.0% | 51.6% | 91.5% | 51.4% | 44.5% | 0.0% | 51.5% |

A crop-only gate confirms that the model can recognize the object in a localized view, but it still accepts many ghost claims because it never asks whether the claim disappears after object removal. GHOST-Guard adds that masked counterfactual check.

## Generated Artifacts

- `outputs/runs/large200_bboxsolid_qwen2_5vl_7b/`
- `outputs/runs/large200_bboxsolid_qwen2_5vl_3b/`
- `outputs/runs/large200_bboxsolid_llava_7b/`
- `outputs/artifact_labels_large200_bbox_solid.csv`
- `outputs/comparison/model_summary_metrics.csv`
- `outputs/comparison/model_category_metrics.csv`
- `outputs/comparison/clean_removal_metrics.csv`
- `outputs/comparison/baseline_safety_comparison.csv`
- `outputs/comparison/model_metric_comparison.png`
- `outputs/comparison/category_ghost_rate_comparison.png`
- `outputs/comparison/clean_removal_ghost_rate.png`
- `outputs/comparison/multi_model_summary.md`

## Earlier Blur-Mask Pilot

The earlier 50-sample, 3-model blur-fill pilot produced larger ghost rates. After visual inspection, those numbers should be treated as an ablation rather than the headline because blurred segmentation regions can preserve visible target shape.

That critique led to the scaled bbox-solid intervention and artifact-label reporting above.

## Publication Claim

The strongest defensible claim is:

> COCO-Ghost exposes context-induced object persistence in local VLMs using audited object-removal counterfactuals, and GHOST-Guard provides a black-box abstention wrapper that refuses object-presence claims when they survive removal of the visual evidence.

Avoid claiming that GHOST-Guard solves hallucination. It catches and abstains on one interpretable class of unsupported object-presence claims.

## Remaining Work Before Submission

- Add confidence intervals to the project page and paper tables.
- Add a larger 500-sample run or a second dataset if compute time allows.
- Add human spot-check labels for a subset of the artifact-labeling decisions.
- Add related-work text around CHAIR, POPE, AMBER, VCD, OPERA, Woodpecker, counterfactual editing, and object-presence calibration.
