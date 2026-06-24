from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from skimage.morphology import binary_dilation, disk
from skimage.restoration import inpaint
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


def mask_from_box(image_size: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    width, height = image_size
    x1, y1, x2, y2 = box
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.astype(bool)
    return binary_dilation(mask.astype(bool), footprint=disk(pixels))


def local_mean_fill(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb)
    fill_mask = ~mask
    mean_color = tuple(np.mean(arr[fill_mask], axis=0).astype(np.uint8).tolist()) if np.any(fill_mask) else (127, 127, 127)
    return Image.new("RGB", rgb.size, mean_color)


def blurred_fill(rgb: Image.Image) -> Image.Image:
    radius = max(12, int(max(rgb.size) * 0.035))
    return rgb.filter(ImageFilter.GaussianBlur(radius=radius))


def patch_shuffle_fill(rgb: Image.Image, mask: np.ndarray, seed: int = 42) -> Image.Image:
    arr = np.asarray(rgb).copy()
    h, w = mask.shape
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return rgb.copy()
    y1, y2 = max(0, yy.min() - 20), min(h, yy.max() + 21)
    x1, x2 = max(0, xx.min() - 20), min(w, xx.max() + 21)
    local = arr[y1:y2, x1:x2]
    local_unmasked = ~mask[y1:y2, x1:x2]
    candidates = local[local_unmasked]
    if len(candidates) < 8:
        return local_mean_fill(rgb, mask)
    rng = np.random.default_rng(seed)
    arr[mask] = candidates[rng.integers(0, len(candidates), size=int(mask.sum()))]
    return Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=2))


def biharmonic_fill(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    if not np.any(~mask):
        return local_mean_fill(rgb, mask)
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    try:
        filled = inpaint.inpaint_biharmonic(arr, mask.astype(bool), channel_axis=-1)
    except ValueError:
        return local_mean_fill(rgb, mask)
    filled = np.clip(filled * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(filled, mode="RGB")


def make_masked_image(image: Image.Image, mask: np.ndarray, fill_mode: str, seed: int = 42) -> Image.Image:
    rgb = image.convert("RGB")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if fill_mode in {"mean", "solid_mean", "solid_local_mean"}:
        fill = local_mean_fill(rgb, mask)
    elif fill_mode in {"patch_shuffle", "local_patch"}:
        return patch_shuffle_fill(rgb, mask, seed=seed)
    elif fill_mode in {"biharmonic", "inpaint_biharmonic"}:
        return biharmonic_fill(rgb, mask)
    else:
        fill = blurred_fill(rgb)
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
    out_root = project_path(cfg.get("counterfactual", {}).get("output_dir", "outputs/counterfactuals"))
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
        original_mask = coco.annToMask(ann).astype(bool)
        image, original_mask, scale = resize_image_and_mask(image, original_mask, cfg["dataset"]["image_max_side"])
        crop_box = crop_box_from_bbox(ann["bbox"], scale, image.size, cfg["counterfactual"]["crop_margin_fraction"])
        if cfg["counterfactual"].get("mask_shape", "segmentation") == "bbox":
            removal_margin = float(cfg["counterfactual"].get("bbox_removal_margin_fraction", 0.08))
            mask_box = crop_box_from_bbox(ann["bbox"], scale, image.size, removal_margin)
            mask = mask_from_box(image.size, mask_box)
        else:
            mask = dilate_mask(original_mask, int(cfg["counterfactual"].get("mask_dilation_pixels", 0)))
        crop = image.crop(crop_box)
        masked = make_masked_image(
            image,
            mask,
            cfg["counterfactual"].get("mask_fill", "mean_blur"),
            seed=int(cfg["dataset"].get("seed", 42)) + int(row["annotation_id"]),
        )
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
                "mask_fill": cfg["counterfactual"].get("mask_fill", "mean_blur"),
                "mask_dilation_pixels": int(cfg["counterfactual"].get("mask_dilation_pixels", 0)),
                "mask_shape": cfg["counterfactual"].get("mask_shape", "segmentation"),
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
