from __future__ import annotations

import random
import shutil
from pathlib import Path


TRAIN_SAMPLES = 24
VALID_SAMPLES = 6
TEST_SAMPLES = 12
RANDOM_SEED = 20260522


def sample_stems(labels_dir: Path, sample_size: int, rng: random.Random) -> list[str]:
    stems = sorted(path.stem for path in labels_dir.glob("*.txt"))
    if len(stems) < sample_size:
        raise ValueError(
            f"{labels_dir} has {len(stems)} labels, but {sample_size} samples are required."
        )
    return sorted(rng.sample(stems, sample_size))


def copy_sample(split_src: Path, split_dst: Path, stems: list[str]) -> None:
    images_src = split_src / "images"
    labels_src = split_src / "labels"
    images_dst = split_dst / "images"
    labels_dst = split_dst / "labels"

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        label_path = labels_src / f"{stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for stem {stem}: {label_path}")
        shutil.copy2(label_path, labels_dst / label_path.name)

        image_matches = sorted(images_src.glob(f"{stem}.*"))
        if not image_matches:
            raise FileNotFoundError(f"Missing image files for stem {stem} in {images_src}")

        for image_path in image_matches:
            shutil.copy2(image_path, images_dst / image_path.name)


def write_fold_yaml(fold_dst: Path) -> None:
    yaml_content = "\n".join(
        [
            "names:",
            "- damage",
            "nc: 1",
            f"path: {fold_dst.resolve()}",
            "train: train/images",
            "val: valid/images",
            "",
        ]
    )
    (fold_dst / "data.yaml").write_text(yaml_content, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "data"
    smoke_root = project_root / "data_smoke"
    rng = random.Random(RANDOM_SEED)

    if not source_root.exists():
        raise FileNotFoundError(f"Source data directory does not exist: {source_root}")

    if smoke_root.exists():
        shutil.rmtree(smoke_root)

    for fold_dir in sorted(source_root.glob("fold_*")):
        fold_dst = smoke_root / fold_dir.name

        train_stems = sample_stems(fold_dir / "train" / "labels", TRAIN_SAMPLES, rng)
        valid_stems = sample_stems(fold_dir / "valid" / "labels", VALID_SAMPLES, rng)

        copy_sample(fold_dir / "train", fold_dst / "train", train_stems)
        copy_sample(fold_dir / "valid", fold_dst / "valid", valid_stems)
        write_fold_yaml(fold_dst)

    test_stems = sample_stems(source_root / "test" / "labels", TEST_SAMPLES, rng)
    copy_sample(source_root / "test", smoke_root / "test", test_stems)

    print(f"Created smoke dataset at: {smoke_root}")
    print(
        f"Per fold: train={TRAIN_SAMPLES}, valid={VALID_SAMPLES}; test={TEST_SAMPLES} common samples."
    )


if __name__ == "__main__":
    main()
