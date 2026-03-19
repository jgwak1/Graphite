from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, List

import torch
from torch_geometric.data import Data


MAX_NODES = 400_000
MIN_NODES = 10


def load_dataset(
    benign_data_path: str,
    malware_data_path: str,
    dim_node: int,
    dim_edge: int,
) -> List[Data]:
    """Load benign and malware graph samples into a single dataset."""

    loader = GraphDatasetLoader()
    benign_dataset = loader.parse_all_data(
        load_path=benign_data_path,
        num_node_attr=dim_node,
        num_edge_attr=dim_edge,
    )
    malware_dataset = loader.parse_all_data(
        load_path=malware_data_path,
        num_node_attr=dim_node,
        num_edge_attr=dim_edge,
    )

    loaded_dataset = benign_dataset + malware_dataset
    print(
        (
            f"+ dataset loaded #Benign = {len(benign_dataset)} | "
            f"#Malware = {len(malware_dataset)} from\n"
            f"\t'{benign_data_path}' and\n"
            f"\t'{malware_data_path}', respectively."
        ),
        flush=True,
    )
    return loaded_dataset


class GraphDatasetLoader:
    """Load processed graph samples into PyTorch Geometric Data objects."""

    def load_pickle(self, file_path: str | Path):
        """Load a pickled graph sample."""
        path = Path(file_path)
        with path.open("rb") as handle:
            try:
                return pickle.load(handle)
            except pickle.UnpicklingError:
                return None

    def parse_single_graph(
        self,
        file_path: str | Path,
        num_node_attr: int = 5,
        num_edge_attr: int = 80,
    ) -> Data | None:
        """
        Parse one processed graph sample into a PyTorch Geometric Data object.
        """
        path = Path(file_path)
        sample_name = path.name
        data = self.load_pickle(path)

        if data is None:
            print(f">>> pickle.UnpicklingError for sample {sample_name}", flush=True)
            return None

        x = data["x"]
        y = data["y"]
        edge_attr = data["edge_attr"]
        edge_list = data["edge_list"]

        num_nodes = len(x)
        num_edges = len(edge_list[0]) if edge_list and len(edge_list) > 0 else 0

        if num_nodes > MAX_NODES:
            print(
                f"{sample_name} >>> #nodes: {num_nodes} #edges: {len(edge_attr)} | sample skipped!",
                flush=True,
            )
            return None

        if num_nodes < MIN_NODES:
            print(
                f"{sample_name} >>> #nodes: {num_nodes} #edges: {len(edge_attr)} | sample skipped!",
                flush=True,
            )
            return None

        if not self._has_expected_feature_size(x, num_node_attr):
            observed = len(x[0]) if x else 0
            print(
                f"{sample_name} >>> #node attributes are mismatched, {observed} | sample skipped!",
                flush=True,
            )
            return None

        if not self._has_expected_feature_size(edge_attr, num_edge_attr):
            observed = len(edge_attr[0]) if edge_attr else 0
            print(
                f"{sample_name} >>> #edge attributes are mismatched, {observed} | sample skipped!",
                flush=True,
            )
            return None

        graph = Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_list, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long),
            name=sample_name,
        )

        print(f"> {sample_name} | #node: {num_nodes} #edge: {num_edges}", flush=True)
        return graph

    def parse_all_data(
        self,
        load_path: str | Path,
        num_node_attr: int,
        num_edge_attr: int,
    ) -> List[Data]:
        """Parse all valid graph samples in a directory."""
        path = Path(load_path)
        dataset: List[Data] = []

        for file_path in sorted(path.iterdir()):
            if not file_path.is_file():
                continue

            if not self._is_supported_sample(file_path.name):
                continue

            graph = self.parse_single_graph(
                file_path=file_path,
                num_node_attr=num_node_attr,
                num_edge_attr=num_edge_attr,
            )
            if graph is not None:
                dataset.append(graph)

        return dataset

    @staticmethod
    def _is_supported_sample(filename: str) -> bool:
        """Return True if the filename matches the expected processed-sample pattern."""
        return "_Sample_" in filename or "_SUBGRAPH_" in filename

    @staticmethod
    def _has_expected_feature_size(
        feature_rows: Iterable[Iterable[float]],
        expected_size: int,
    ) -> bool:
        """Return True if every feature row has the expected width."""
        for row in feature_rows:
            if len(row) != expected_size:
                return False
        return True