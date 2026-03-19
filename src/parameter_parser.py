from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "dataset"
DEFAULT_EVENTNAME_EDGEFEATS_PATH = DEFAULT_DATASET_PATH / "EventName_EdgeFeatures.json"
DEFAULT_NODETYPE_NODEFEATS_PATH = DEFAULT_DATASET_PATH / "NodeType_NodeFeatures.json"


def param_parser() -> argparse.Namespace:
    """Parse command-line arguments for the Graphite N-gram entrypoint."""

    parser = argparse.ArgumentParser(
        description="Run the Graphite N-gram modeling pipeline on the processed dataset."
    )

    parser.add_argument(
        "--N",
        type=int,
        default=4,
        help="N-gram size for Graphite N-gram modeling.",
    )

    parser.add_argument(
        "--pool",
        type=str,
        default="sum",
        choices=["sum", "mean", "max"],
        help="Pooling method used to aggregate thread-level embeddings.",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the processed dataset directory.",
    )

    parser.add_argument(
        "--eventname-edgefeats-path",
        type=str,
        default=str(DEFAULT_EVENTNAME_EDGEFEATS_PATH),
        help="Path to EventName_EdgeFeatures.json.",
    )

    parser.add_argument(
        "--nodetype-nodefeats-path",
        type=str,
        default=str(DEFAULT_NODETYPE_NODEFEATS_PATH),
        help="Path to NodeType_NodeFeatures.json.",
    )

    return parser.parse_args()