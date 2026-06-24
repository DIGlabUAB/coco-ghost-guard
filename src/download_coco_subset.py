from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import add_config_arg, ensure_dirs, load_config, project_path


def download_file(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK] Exists: {path}")
        return
    ensure_dirs(path.parent)
    with urllib.request.urlopen(url) as response, open(path, "wb") as f:
        total = int(response.headers.get("Content-Length", "0"))
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))


def load_annotations(annotation_path: Path) -> dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_instances(data: dict, cfg: dict) -> list[dict]:
    dataset = cfg["dataset"]
    rng = random.Random(dataset.get("seed", 42))
    categories_by_name = {cat["name"]: cat for cat in data["categories"]}
    images_by_id = {img["id"]: img for img in data["images"]}
    annotations_by_category: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        annotations_by_category.setdefault(ann["category_id"], []).append(ann)

    selected: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()
    for category_name in dataset["categories"]:
        category = categories_by_name.get(category_name)
        if not category:
            print(f"[WARN] Category not found in COCO: {category_name}")
            continue

        candidates = annotations_by_category.get(category["id"], [])[:]
        rng.shuffle(candidates)
        count = 0
        for ann in candidates:
            image = images_by_id.get(ann["image_id"])
            if not image:
                continue
            if (ann["image_id"], ann["category_id"]) in used_pairs:
                continue
            if not dataset.get("allow_crowd", False) and ann.get("iscrowd", 0):
                continue
            if not ann.get("segmentation"):
                continue
            area = float(ann.get("area", 0))
            image_area = float(image["width"] * image["height"])
            if area < dataset["min_object_area"]:
                continue
            if area > dataset["max_object_area_fraction"] * image_area:
                continue

            bbox_x, bbox_y, bbox_w, bbox_h = ann["bbox"]
            sample_id = f"{category_name.replace(' ', '_')}_{ann['image_id']}_{ann['id']}"
            selected.append(
                {
                    "sample_id": sample_id,
                    "image_id": ann["image_id"],
                    "annotation_id": ann["id"],
                    "category_id": ann["category_id"],
                    "category_name": category_name,
                    "file_name": image["file_name"],
                    "image_path": str(project_path("data", "selected", "images", image["file_name"])),
                    "width": image["width"],
                    "height": image["height"],
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "area": area,
                    "coco_url": image["coco_url"],
                }
            )
            used_pairs.add((ann["image_id"], ann["category_id"]))
            count += 1
            if count >= dataset["max_images_per_category"]:
                break
        print(f"[OK] Selected {count} for {category_name}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download annotations and selected COCO images.")
    add_config_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = cfg["dataset"]

    raw_dir = project_path("data", "raw")
    ann_dir = project_path("data", "coco_annotations")
    selected_dir = project_path("data", "selected")
    images_dir = selected_dir / "images"
    ensure_dirs(raw_dir, ann_dir, selected_dir, images_dir)

    zip_path = raw_dir / Path(dataset["annotation_url"]).name
    download_file(dataset["annotation_url"], zip_path)

    annotation_path = ann_dir / "annotations" / dataset["annotation_file"]
    if not annotation_path.exists():
        print(f"[INFO] Extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ann_dir)
    if not annotation_path.exists():
        raise SystemExit(f"[ERROR] Missing annotation file: {annotation_path}")

    data = load_annotations(annotation_path)
    selected = select_instances(data, cfg)
    if not selected:
        raise SystemExit("[ERROR] No samples selected. Try lowering area thresholds.")

    for row in tqdm(selected, desc="selected images"):
        target = Path(row["image_path"])
        download_file(row["coco_url"], target)

    df = pd.DataFrame(selected)
    csv_path = selected_dir / "selected_instances.csv"
    jsonl_path = selected_dir / "selected_instances.jsonl"
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {jsonl_path}")


if __name__ == "__main__":
    main()
