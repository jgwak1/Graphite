from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

from sklearn.metrics import accuracy_score, f1_score

from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram
from parameter_parser import param_parser


def load_json_file(path: str | Path):
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_label_from_sample_name(sample_name: str) -> int:
    """Infer the binary label from the sample name."""
    return 1 if "malware" in sample_name.lower() else 0


def build_dataset_paths(dataset_root: str | Path) -> dict[str, Path]:
    """Build standard train/test dataset paths."""
    root = Path(dataset_root)

    return {
        "train_benign": root / "train" / "benign",
        "train_malware": root / "train" / "malware",
        "test_benign": root / "test" / "benign",
        "test_malware": root / "test" / "malware",
    }


def evaluate_model(
    model: Graphite_Ngram,
    test_dataset: Iterable,
) -> Tuple[list[int], list[int]]:
    """Run inference on the test set and collect predictions and labels."""
    predictions: list[int] = []
    truths: list[int] = []

    for sample in test_dataset:
        prediction = int(model.predict(sample))
        truth = infer_label_from_sample_name(sample.name)

        print(
            f"Predicted: {prediction} | Truth: {truth} --- {sample.name}",
            flush=True,
        )

        predictions.append(prediction)
        truths.append(truth)

    return predictions, truths


def main() -> None:
    args = param_parser()

    eventname_edgefeats = load_json_file(args.eventname_edgefeats_path)
    nodetype_nodefeats = load_json_file(args.nodetype_nodefeats_path)

    dataset_paths = build_dataset_paths(args.dataset_path)
    dim_node = len(nodetype_nodefeats)
    dim_edge = len(eventname_edgefeats) + 1  # +1 for event timestamp

    train_dataset = load_dataset(
        benign_data_path=str(dataset_paths["train_benign"]),
        malware_data_path=str(dataset_paths["train_malware"]),
        dim_node=dim_node,
        dim_edge=dim_edge,
    )
    test_dataset = load_dataset(
        benign_data_path=str(dataset_paths["test_benign"]),
        malware_data_path=str(dataset_paths["test_malware"]),
        dim_node=dim_node,
        dim_edge=dim_edge,
    )

    model = Graphite_Ngram(N=args.N, pool=args.pool)
    model.fit(
        train_dataset=train_dataset,
        nodetype_nodefeats=nodetype_nodefeats,
        eventname_edgefeats=eventname_edgefeats,
    )

    predictions, truths = evaluate_model(model=model, test_dataset=test_dataset)

    test_accuracy = accuracy_score(y_true=truths, y_pred=predictions)
    test_f1 = f1_score(y_true=truths, y_pred=predictions, zero_division=0)

    print("-" * 50, flush=True)
    print(f"Test-Acc: {test_accuracy:.4f} | Test-F1: {test_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()