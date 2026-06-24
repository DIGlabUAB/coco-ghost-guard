from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from skimage.transform import resize
from tqdm import tqdm

from .common import add_config_arg, ensure_dirs, load_config, project_path


def resize_image_and_mask(image: Image.Image, mask: np.ndarray, max_side: int) -> tuple[Image.Image, np.ndarray, float]:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale == 1.0:
        return image, mask.astype(bool), scale
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    resized_mask = resize(
        mask.astype(float),
        (new_size[1], new_size[0]),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    return resized_image, resized_mask, scale


def crop_box_from_bbox(bbox: list[float], scale: float, image_size: tuple[int, int], margin_fraction: float) -> tuple[int, int, int, int]:
    x, y, w, h = [v * scale for v in bbox]
    margin = margin_fraction * max(w, h)
    width, height = image_size
    x1 = max(0, int(np.floor(x - margin)))
    y1 = max(0, int(np.floor(y - margin)))
    x2 = min(width, int(np.ceil(x + w + margin)))
    y2 = min(height, int(np.ceil(y + h + margin)))
    return x1, y1, x2, y2


def make_masked_image(image: Image.Image, mask: np.ndarray, fill_mode: str) -> Image.Image:
    rgb = image.convert("RGB")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if fill_mode == "mean":
        arr = np.asarray(rgb)
        mean_color = tuple(np.mean(arr[~mask], axis=0).astype(np.uint8).tolist()) if np.any(~mask) else (127, 127, 127)
        fill = Image.new("RGB", rgb.size, mean_color)
    else:
        radius = max(12, int(max(rgb.size) * 0.035))
        fill = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
    masked = rgb.copy()
    masked.paste(fill, mask=mask_img)
    return masked


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red = np.zeros((base.size[1], base.size[0], 4), dtype=np.uint8)
    red[mask] = [255, 40, 40, 115]
    overlay = Image.fromarray(red, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create COCO-Ghost counterfactual image views.")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)

    selected_path = project_path("data", "selected", "selected_instances.csv")
    annotation_path = project_path("data", "coco_annotations", "annotations", cfg["dataset"]["annotation_file"])
    out_root = project_path("outputs", "counterfactuals")
    ensure_dirs(out_root)
    if not selected_path.exists():
        raise SystemExit(f"[ERROR] Missing selected instances: {selected_path}")
    if not annotation_path.exists():
        raise SystemExit(f"[ERROR] Missing COCO annotations: {annotation_path}")

    df = pd.read_csv(selected_path)
    coco = COCO(str(annotation_path))
    manifest: list[dict] = []
    for row in tqdm(df.to_dict("records"), desc="counterfactuals"):
        out_dir = out_root / row["sample_id"]
        ensure_dirs(out_dir)
        image = Image.open(row["image_path"]).convert("RGB")
        ann = coco.loadAnns([int(row["annotation_id"])])[0]
        mask = coco.annToMask(ann).astype(bool)
        image, mask, scale = resize_image_and_mask(image, mask, cfg["dataset"]["image_max_side"])

        crop_box = crop_box_from_bbox(ann["bbox"], scale, image.size, cfg["counterfactual"]["crop_margin_fraction"])
        crop = image.crop(crop_box)
        masked = make_masked_image(image, mask, cfg["counterfactual"].get("mask_fill", "mean_blur"))
        overlay = make_overlay(image, mask)

        paths = {
            "original": out_dir / "original.jpg",
            "crop": out_dir / "crop.jpg",
            "masked": out_dir / "masked.jpg",
            "mask": out_dir / "mask.png",
            "overlay": out_dir / "overlay.jpg",
        }
        image.save(paths["original"], quality=92)
        crop.save(paths["crop"], quality=92)
        masked.save(paths["masked"], quality=92)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(paths["mask"])
        if cfg["counterfactual"].get("save_debug_overlay", True):
            overlay.save(paths["overlay"], quality=92)

        item = dict(row)
        item.update(
            {
                "original_path": str(paths["original"]),
                "crop_path": str(paths["crop"]),
                "masked_path": str(paths["masked"]),
                "mask_path": str(paths["mask"]),
                "overlay_path": str(paths["overlay"]),
                "resized_width": image.size[0],
                "resized_height": image.size[1],
                "resize_scale": scale,
                "crop_box": list(crop_box),
            }
        )
        manifest.append(item)

    manifest_path = out_root / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[OK] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
