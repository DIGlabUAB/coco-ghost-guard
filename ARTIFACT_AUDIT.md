# Artifact Audit Notes

This is a first-pass visual audit of the expanded run artifacts. It is not a complete human annotation pass.

## What Looks Good

- The counterfactual construction is simple and reproducible: COCO instance mask plus blurred local fill.
- The original/crop/masked contact sheets render correctly.
- Several examples clearly show the intended phenomenon: the target object is removed or heavily obscured, but the model still answers `YES`.

## Main Caveat

Some masked regions leave object-shaped blurred patches. This is expected from blur-fill masking, but it matters for interpretation:

- If the masked patch preserves the shape of a racket, bat, fork, or glass, the model may be responding to the residual silhouette rather than pure scene context.
- These cases should be treated as possible counterfactual artifacts.
- The paper should not claim every masked-image `YES` is definitely caused by scene priors.

## Recommended Paper Language

Use:

> Masked-image YES responses may reflect either context-induced persistence or residual evidence from object-shaped masking artifacts. We therefore report COCO-Ghost as a counterfactual stress test and include qualitative artifact inspection as part of the evaluation.

Avoid:

> Every masked-image YES is a hallucination.

## Next Audit Step

Before submission, add a manual artifact label for each sample:

- `clean_removal`
- `minor_residual_shape`
- `major_residual_shape`
- `object_still_visible`

Then report Ghost Object Rate both overall and on `clean_removal` examples only.
