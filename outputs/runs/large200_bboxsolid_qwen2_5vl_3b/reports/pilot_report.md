# COCO-Ghost + GHOST-Guard Pilot Report

## Summary

This pilot evaluates whether a local vision-language model continues to report an object after the object has been removed from a natural image using COCO instance masks. We define this behavior as a ghost object claim. We then apply GHOST-Guard, a training-free evidence gate that accepts an object claim only when it is supported by the object crop and disappears after object removal.

## Main Result

The baseline Ghost Object Rate was 47.5%. This means that, among images where the model originally reported the target object, it still reported the object after removal in 47.5% of cases.

After applying GHOST-Guard, 100.0% of these suspicious claims were downgraded to UNSURE rather than accepted as true object detections.

## Dataset

- COCO split: `val2017`
- Target categories: tennis racket, baseball bat, skateboard, surfboard, fork, wine glass, laptop, keyboard, handbag, tie, backpack, umbrella, bottle, cup, bowl, chair, cell phone, book, clock, vase
- Samples evaluated: 200
- Images per category target: 10

## Model

- Ollama model: `qwen2.5vl:3b`
- Temperature: 0
- Runtime mode: local black-box REST calls through Ollama

## Methods

For each selected COCO instance, COCO-Ghost creates three views: the original image, a crop around the target object, and a masked image where the target object's segmentation mask is replaced by blurred local background. The same strict object-presence prompt is sent for each view. GHOST-Guard accepts the object claim only if the model answers YES on the original image, YES on the crop, and NO on the masked image.

## Metrics

| Metric | Value |
|---|---:|
| Original YES Rate | 88.5% |
| Crop YES Rate | 80.0% |
| Masked YES Rate | 43.0% |
| Ghost Object Rate | 47.5% |
| Context Persistence Index | 0.486 |
| GHOST-Guard Acceptance Rate | 30.0% |
| GHOST-Guard Abstention Rate | 58.5% |
| Prevented Ghost Claim Rate | 100.0% |

## Category Findings

| Category | Original YES | Crop YES | Ghost Object Rate | Guard Abstention |
|---|---:|---:|---:|---:|
| backpack | 60.0% | 80.0% | 33.3% | 20.0% |
| baseball bat | 90.0% | 90.0% | 33.3% | 60.0% |
| book | 100.0% | 80.0% | 70.0% | 70.0% |
| bottle | 90.0% | 90.0% | 77.8% | 70.0% |
| bowl | 90.0% | 50.0% | 66.7% | 60.0% |
| cell phone | 80.0% | 60.0% | 25.0% | 40.0% |
| chair | 70.0% | 40.0% | 85.7% | 70.0% |
| clock | 90.0% | 80.0% | 33.3% | 50.0% |
| cup | 60.0% | 40.0% | 66.7% | 50.0% |
| fork | 100.0% | 90.0% | 30.0% | 30.0% |
| handbag | 70.0% | 80.0% | 42.9% | 40.0% |
| keyboard | 100.0% | 90.0% | 80.0% | 80.0% |
| laptop | 100.0% | 90.0% | 10.0% | 20.0% |
| skateboard | 90.0% | 100.0% | 11.1% | 60.0% |
| surfboard | 90.0% | 80.0% | 22.2% | 80.0% |
| tennis racket | 100.0% | 100.0% | 40.0% | 80.0% |
| tie | 100.0% | 100.0% | 20.0% | 50.0% |
| umbrella | 100.0% | 90.0% | 50.0% | 70.0% |
| vase | 90.0% | 70.0% | 55.6% | 70.0% |
| wine glass | 100.0% | 100.0% | 100.0% | 100.0% |

## Example Failures

- `tennis_racket_203864_657072` (tennis racket): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `tennis_racket_96427_659911` (tennis racket): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `tennis_racket_323496_655580` (tennis racket): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `tennis_racket_64523_1482412` (tennis racket): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `baseball_bat_57232_1476315` (baseball bat): original=YES, crop=NO, masked=YES, final=UNSURE, risk=high_context_hallucination

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
