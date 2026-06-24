# Artifact Audit Notes

The artifact audit changed the direction of the project.

The first expanded runs used COCO segmentation masks filled with blurred local pixels. Those counterfactuals were easy to reproduce, but visual inspection showed a serious problem: blur-fill regions often preserved the target object's silhouette. In those cases, a masked-image `YES` response may reflect residual visual evidence rather than pure context-induced persistence.

## Revised Scaled Intervention

For the scaled experiments, COCO-Ghost now uses:

- `mask_shape: bbox`
- `bbox_removal_margin_fraction: 0.12`
- `mask_fill: solid_local_mean`
- `mask_dilation_pixels: 0`

This removes a padded bounding-box region rather than the exact segmentation silhouette. It is less naturalistic, but it better answers the evidence question because it avoids leaving target-shaped blur.

The scaled paper framing should call this an **occlusion counterfactual**, not natural inpainting.

## Scaled Artifact Labels

The 200-sample bbox-solid counterfactual set was labeled with the artifact-labeling pipeline:

| Label | Count |
|---|---:|
| `clean_removal` | 200 |

Because all 200 scaled samples were labeled clean, the main results and clean-removal results are identical:

| Model | Clean Ghost Rate |
|---|---:|
| `qwen2.5vl:7b` | 51.6% |
| `qwen2.5vl:3b` | 47.5% |
| `llava:7b` | 33.7% |

## Recommended Paper Language

Use:

> We report scaled results on a conservative bbox-occlusion counterfactual set. This intervention is visually less natural than inpainting, but it removes object-shaped residuals and makes the measured effect harder to attribute to silhouette leakage.

Also use:

> Earlier blur-fill segmentation counterfactuals produced higher ghost rates, but because the blur can preserve target shape, we treat them as an ablation and use artifact-labeled bbox occlusion for the headline results.

Avoid:

> Every masked-image YES is a hallucination.

Avoid:

> The counterfactual images are photorealistic object removals.

## Why This Matters

The benchmark is stronger when it is honest about the intervention. The goal is not to make a perfect edited image; the goal is to ask whether the model's object claim depends on visible object evidence. A blunt but clean occlusion is a defensible stress test for that question.
