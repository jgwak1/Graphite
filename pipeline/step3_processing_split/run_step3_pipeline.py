from __future__ import annotations

import argparse
from pathlib import Path

from .process_graph_data import GraphDataProcessor
from .split_dataset import create_train_test_split


def run_step3_pipeline(
    input_root: str,
    working_root: str,
    split_output_root: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    debug: bool = False,
) -> None:
    """
    Step 3 runner.

    1. Convert Step 2 graph artifacts into processed model-ready pickles.
    2. Create train/test splits from the processed samples.
    """
    processor = GraphDataProcessor(debug=debug)

    processor.process_dataset(
        input_root=input_root,
        output_root=working_root,
        data_type="Benign",
        concat_events=True,
        compute_order=True,
    )

    processor.process_dataset(
        input_root=input_root,
        output_root=working_root,
        data_type="Malware",
        concat_events=True,
        compute_order=True,
    )

    summary = create_train_test_split(
        processed_root=working_root,
        output_root=split_output_root,
        train_ratio=train_ratio,
        seed=seed,
    )

    for class_name, counts in summary.items():
        print(f"{class_name}: train={counts['train']} test={counts['test']} total={counts['total']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 3 processing and train/test split.")
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory containing Benign/ and Malware/ sample folders from Step 2.",
    )
    parser.add_argument(
        "--working-root",
        required=True,
        help="Directory where Processed_Benign/ and Processed_Malware/ will be written.",
    )
    parser.add_argument(
        "--split-output-root",
        required=True,
        help="Directory where train/ and test/ splits will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_step3_pipeline(
        input_root=args.input_root,
        working_root=args.working_root,
        split_output_root=args.split_output_root,
        train_ratio=args.train_ratio,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
