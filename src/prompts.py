from __future__ import annotations


def object_presence_prompt(object_name: str) -> str:
    return f"""You are a careful visual verifier. Look only at visible pixels in the image.

Question: Is there a "{object_name}" visible in this image?

Rules:
- Answer YES only if the object itself is visibly present.
- Answer NO if the object is not visible.
- Answer UNSURE if the image is ambiguous or the object is occluded.
- Do not infer from scene context.
- Do not infer from what would usually be present.
- Return strict JSON only.

JSON schema:
{{
  "answer": "YES" | "NO" | "UNSURE",
  "evidence": "short phrase describing visible evidence only"
}}"""


def localization_prompt(object_name: str) -> str:
    return f"""You are a careful visual grounding model.

Task: If a "{object_name}" is visible, return one bounding box around the visible object.
If not visible, return NONE.

Use image pixel coordinates.
Return strict JSON only.

JSON schema:
{{
  "visible": true | false,
  "bbox": [x1, y1, x2, y2] | null,
  "evidence": "short visible evidence phrase"
}}"""
