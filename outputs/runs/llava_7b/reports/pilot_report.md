# COCO-Ghost + GHOST-Guard Pilot Report

## Summary

This pilot evaluates whether a local vision-language model continues to report an object after the object has been removed from a natural image using COCO instance masks. We define this behavior as a ghost object claim. We then apply GHOST-Guard, a training-free evidence gate that accepts an object claim only when it is supported by the object crop and disappears after object removal.

## Main Result

The baseline Ghost Object Rate was 61.5%. This means that, among images where the model originally reported the target object, it still reported the object after removal in 61.5% of cases.

After applying GHOST-Guard, 100.0% of these suspicious claims were downgraded to UNSURE rather than accepted as true object detections.

## Dataset

- COCO split: `val2017`
- Target categories: tennis racket, baseball bat, skateboard, surfboard, fork, wine glass, laptop, keyboard, handbag, tie
- Samples evaluated: 50
- Images per category target: 5

## Model

- Ollama model: `llava:7b`
- Temperature: 0
- Runtime mode: local black-box REST calls through Ollama

## Methods

For each selected COCO instance, COCO-Ghost creates three views: the original image, a crop around the target object, and a masked image where the target object's segmentation mask is replaced by blurred local background. The same strict object-presence prompt is sent for each view. GHOST-Guard accepts the object claim only if the model answers YES on the original image, YES on the crop, and NO on the masked image.

## Metrics

| Metric | Value |
|---|---:|
| Original YES Rate | 52.0% |
| Crop YES Rate | 74.0% |
| Masked YES Rate | 32.0% |
| Ghost Object Rate | 61.5% |
| Context Persistence Index | 0.615 |
| GHOST-Guard Acceptance Rate | 18.0% |
| GHOST-Guard Abstention Rate | 34.0% |
| Prevented Ghost Claim Rate | 100.0% |

## Category Findings

| Category | Original YES | Crop YES | Ghost Object Rate | Guard Abstention |
|---|---:|---:|---:|---:|
| baseball bat | 40.0% | 40.0% | 50.0% | 40.0% |
| fork | 0.0% | 60.0% | 0.0% | 0.0% |
| handbag | 0.0% | 0.0% | 0.0% | 0.0% |
| keyboard | 80.0% | 100.0% | 50.0% | 40.0% |
| laptop | 60.0% | 100.0% | 66.7% | 40.0% |
| skateboard | 100.0% | 80.0% | 60.0% | 60.0% |
| surfboard | 80.0% | 60.0% | 100.0% | 80.0% |
| tennis racket | 80.0% | 100.0% | 25.0% | 20.0% |
| tie | 40.0% | 100.0% | 100.0% | 40.0% |
| wine glass | 40.0% | 100.0% | 50.0% | 20.0% |

## Example Failures

- `tennis_racket_369323_656095` (tennis racket): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `baseball_bat_57232_1476315` (baseball bat): original=YES, crop=NO, masked=YES, final=UNSURE, risk=high_context_hallucination
- `skateboard_125472_639877` (skateboard): original=YES, crop=NO, masked=YES, final=UNSURE, risk=high_context_hallucination
- `skateboard_562229_639293` (skateboard): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination
- `skateboard_489924_639066` (skateboard): original=YES, crop=YES, masked=YES, final=UNSURE, risk=high_context_hallucination

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
