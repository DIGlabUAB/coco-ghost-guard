# Method

## COCO-Ghost Counterfactual Construction

Given a COCO image containing a target object, we use the instance segmentation mask to construct three views: the original image, a crop around the target object, and a masked image in which the target object region is replaced by blurred local background. This preserves scene context while removing direct object evidence.

## Object Presence Querying

For each view, we ask a local vision-language model the same object-presence question: "Is there a {target object} visible in this image?" The model must answer YES, NO, or UNSURE and provide a short visible-evidence phrase.

## Ghost Object Rate

We define Ghost Object Rate as the proportion of cases where the model answers YES on the masked image after answering YES on the original image. This measures whether an object claim persists after object removal.

## GHOST-Guard

GHOST-Guard combines three answers: original image, object crop, and masked image. A claim is accepted only when the model answers YES on the original image, YES on the crop, and NO on the masked image. If the model continues to answer YES after object removal, the claim is downgraded to UNSURE.

## Oracle and Self-Localization Modes

In oracle mode, COCO masks define the removed object region. This isolates the hallucination phenomenon from localization error. In self-localization mode, the VLM proposes the suspected object bounding box, making the method closer to deployment settings.
