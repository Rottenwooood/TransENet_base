import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare RSSCN7 in the same SR layout as datasets/AID-dataset."
    )
    parser.add_argument("--source", default="RSSCN7", help="Path to the RSSCN7 source directory.")
    parser.add_argument("--output", default="datasets/RSSCN7-dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-npy", action="store_true", help="Also save .npy caches beside png files.")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def collect_images(source):
    rows = []
    for class_dir in sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                name = f"{class_dir.name}_{path.stem}".replace(" ", "_").replace("-", "_")
                rows.append({"path": path, "label": class_dir.name, "name": name})
    if not rows:
        raise FileNotFoundError(f"No images found under {source}")
    return rows


def split_by_label(rows, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for label in sorted(grouped):
        items = grouped[label]
        rng.shuffle(items)
        test_count = len(items) // 2
        trainval = items[test_count:]
        val_count = round(len(trainval) * 0.2)
        splits["test"].extend(items[:test_count])
        splits["val"].extend(trainval[:val_count])
        splits["train"].extend(trainval[val_count:])

    for split_items in splits.values():
        split_items.sort(key=lambda item: (item["label"], item["name"]))
    return splits


def save_image(image, path, save_npy=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    if save_npy:
        np.save(path.with_suffix(".npy"), np.asarray(image, dtype=np.uint8))


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to recreate it")
        shutil.rmtree(output)

    rows = collect_images(source)
    splits = split_by_label(rows, args.seed)
    total = len(rows)
    processed = 0

    for split, items in splits.items():
        for item in items:
            image = Image.open(item["path"]).convert("RGB")
            width, height = image.size
            width = width - width % 12
            height = height - height % 12
            if image.size != (width, height):
                image = image.crop((0, 0, width, height))

            filename = f"{item['name']}.png"
            save_image(image, output / split / "HR" / filename, args.save_npy)
            for scale in (2, 3, 4):
                lr = image.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
                save_image(lr, output / split / f"LR_x{scale}" / filename, args.save_npy)

            processed += 1
            if args.log_every and processed % args.log_every == 0:
                print(f"processed {processed}/{total}", flush=True)

    for split in ("train", "val", "test"):
        print(split, len(splits[split]))


if __name__ == "__main__":
    main()
