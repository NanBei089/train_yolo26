from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

SOURCE_DIR = Path("data")
OUTPUT_DIR = Path("dataset")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_class_names(source_dir: Path) -> list[str]:
    classes_file = source_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"Missing classes file: {classes_file}")

    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_file}")
    return names


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError("Expected data/images and data/labels to exist")

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []

    for img in image_files:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing_labels.append(img.name)

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise FileNotFoundError(
            f"Missing label files for {len(missing_labels)} images. First items: {preview}"
        )

    return pairs


def split_pairs(pairs: list[tuple[Path, Path]]) -> dict[str, list[tuple[Path, Path]]]:
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled = pairs[:]
    random.Random(SEED).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
        "_counts": [train_count, val_count, test_count],
    }


def rebuild_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_split_files(split_data: dict[str, list[tuple[Path, Path]]], output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for image_path, label_path in split_data[split]:
            shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
            shutil.copy2(label_path, output_dir / "labels" / split / label_path.name)


def write_dataset_yaml(output_dir: Path, class_names: list[str]) -> Path:
    dataset_yaml = output_dir / "dataset.yaml"
    config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    dataset_yaml.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dataset_yaml


def main() -> None:
    class_names = read_class_names(SOURCE_DIR)
    pairs = collect_pairs(SOURCE_DIR)
    split_data = split_pairs(pairs)

    rebuild_output_dir(OUTPUT_DIR)
    copy_split_files(split_data, OUTPUT_DIR)
    dataset_yaml = write_dataset_yaml(OUTPUT_DIR, class_names)

    train_count, val_count, test_count = split_data["_counts"]
    print(f"Total pairs: {len(pairs)}")
    print(f"Train: {train_count}")
    print(f"Val:   {val_count}")
    print(f"Test:  {test_count}")
    print(f"Classes: {class_names}")
    print(f"Dataset YAML: {dataset_yaml}")


if __name__ == "__main__":
    main()
