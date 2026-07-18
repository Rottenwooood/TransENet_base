import argparse
import random
import shutil
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


LABEL_NAMES = {
    0: "airport",
    1: "beach",
    2: "bridge",
    3: "commercial",
    4: "desert",
    5: "farmland",
    6: "football_field",
    7: "forest",
    8: "industrial",
    9: "meadow",
    10: "mountain",
    11: "park",
    12: "parking",
    13: "pond",
    14: "port",
    15: "railway_station",
    16: "residential",
    17: "river",
    18: "viaduct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare WHU-RS19 in the same SR layout as datasets/AID-dataset."
    )
    parser.add_argument(
        "--parquet",
        default="WHU-RS19/data/train-00000-of-00001-bc192ed023be1f72.parquet",
        help="Path to WHU-RS19 parquet file.",
    )
    parser.add_argument(
        "--output",
        default="datasets/WHU-RS19-dataset",
        help="Output dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save .npy caches beside png files.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def sanitize_stem(path, label):
    stem = Path(path).stem.replace(" ", "_").replace("-", "_")
    label_name = LABEL_NAMES.get(label, f"class_{label}")
    if stem.lower().startswith(label_name.lower()):
        return stem
    return f"{label_name}_{stem}"


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
        val_count = max(1, round(len(trainval) * 0.2))
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
    source = Path(args.parquet)
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to recreate it")
        shutil.rmtree(output)

    metadata = pq.read_table(source, columns=["label", "image.path"])
    rows = []
    used_names = set()
    labels = metadata.column("label").to_pylist()
    paths = metadata.column("path").to_pylist()
    for idx, (label, image_path) in enumerate(zip(labels, paths)):
        name = sanitize_stem(image_path, label)
        if name in used_names:
            name = f"{name}_{idx:04d}"
        used_names.add(name)
        rows.append(
            {
                "idx": idx,
                "name": name,
                "label": label,
            }
        )

    splits = split_by_label(rows, args.seed)
    split_by_idx = {}
    for split, items in splits.items():
        for item in items:
            split_by_idx[item["idx"]] = split, item["name"]

    parquet_file = pq.ParquetFile(source)
    current_idx = 0
    for batch in parquet_file.iter_batches(batch_size=1, columns=["image"]):
        for row_idx in range(batch.num_rows):
            image_info = batch.column(0)[row_idx].as_py()
            split, name = split_by_idx[current_idx]
            current_idx += 1
            image = Image.open(BytesIO(image_info["bytes"])).convert("RGB")
            width, height = image.size
            width = width - width % 12
            height = height - height % 12
            if image.size != (width, height):
                image = image.crop((0, 0, width, height))

            filename = f"{name}.png"
            save_image(image, output / split / "HR" / filename, args.save_npy)
            for scale in (2, 3, 4):
                lr = image.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
                save_image(lr, output / split / f"LR_x{scale}" / filename, args.save_npy)
            if args.log_every and current_idx % args.log_every == 0:
                print(f"processed {current_idx}/{len(rows)}", flush=True)

    for split in ("train", "val", "test"):
        print(split, len(splits[split]))


if __name__ == "__main__":
    main()
