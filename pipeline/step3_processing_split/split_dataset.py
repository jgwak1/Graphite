from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _list_pickle_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix == ".pickle"])


def _split_list(items: Sequence[Path], train_ratio: float, seed: int) -> tuple[List[Path], List[Path]]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    train_count = int(len(items) * train_ratio)
    return items[:train_count], items[train_count:]


def create_train_test_split(
    processed_root: str | Path,
    output_root: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """
    Copy processed pickle samples into train/ and test/ directories.

    Expected inputs under processed_root:
    - Processed_Benign/
    - Processed_Malware/

    Outputs under output_root:
    - train/Processed_Benign/
    - train/Processed_Malware/
    - test/Processed_Benign/
    - test/Processed_Malware/
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    processed_root = Path(processed_root)
    output_root = Path(output_root)
    train_root = output_root / "train"
    test_root = output_root / "test"

    summary: Dict[str, Dict[str, int]] = {}

    for split_dir in [train_root, test_root]:
        split_dir.mkdir(parents=True, exist_ok=True)

    for class_name in ["Processed_Benign", "Processed_Malware"]:
        source_dir = processed_root / class_name
        files = _list_pickle_files(source_dir)
        if not files:
            continue

        train_files, test_files = _split_list(files, train_ratio=train_ratio, seed=seed)

        class_train_dir = train_root / class_name
        class_test_dir = test_root / class_name

        if class_train_dir.exists():
            shutil.rmtree(class_train_dir)
        if class_test_dir.exists():
            shutil.rmtree(class_test_dir)

        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        for src in train_files:
            shutil.copy2(src, class_train_dir / src.name)
        for src in test_files:
            shutil.copy2(src, class_test_dir / src.name)

        summary[class_name] = {
            "train": len(train_files),
            "test": len(test_files),
            "total": len(files),
        }

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/test splits for processed Graphite samples.")
    parser.add_argument("--processed-root", required=True, help="Directory containing Processed_Benign/ and/or Processed_Malware/.")
    parser.add_argument("--output-root", required=True, help="Directory where train/ and test/ folders will be written.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = create_train_test_split(
        processed_root=args.processed_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    for class_name, counts in summary.items():
        print(f"{class_name}: train={counts['train']} test={counts['test']} total={counts['total']}")


if __name__ == "__main__":
    main()
