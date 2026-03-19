from __future__ import annotations

from typing import List

import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import Data


class Graphite_Ngram:
    """
    Graphite N-gram classifier.

    Each thread node is represented by:
    1. an N-gram count vector from its timestamp-sorted event sequence
    2. a count vector of neighboring node types

    Thread-level embeddings are pooled into a graph-level embedding, which is
    then classified by a Random Forest model.
    """

    def __init__(self, N: int = 4, pool: str = "sum") -> None:
        if N < 1:
            raise ValueError("N must be at least 1.")
        if pool not in {"sum", "mean", "max"}:
            raise ValueError("pool must be one of: 'sum', 'mean', or 'max'.")

        self.N = N
        self.pool = pool

        self.count_vectorizer = CountVectorizer(
            ngram_range=(N, N),
            max_df=1.0,
            min_df=1,
            max_features=None,
        )

        # Tuned hyperparameters from the original Graphite experiments.
        self.base_model = RandomForestClassifier(
            n_estimators=500,
            criterion="gini",
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=False,
            random_state=42,
        )

        self.nodetype_nodefeats: List[str] = []
        self.eventname_edgefeats: List[str] = []

    def _ensure_metadata_is_ready(self) -> None:
        if not self.nodetype_nodefeats or not self.eventname_edgefeats:
            raise RuntimeError(
                "Model metadata is not initialized. Call fit(...) first."
            )

    def _thread_nodetype_vector(self, reference_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                1 if node_type.lower() == "thread" else 0
                for node_type in self.nodetype_nodefeats
            ],
            dtype=reference_tensor.dtype,
            device=reference_tensor.device,
        )

    def _thread_node_indices(self, data: Data) -> torch.Tensor:
        thread_nodetype = self._thread_nodetype_vector(data.x)
        return torch.nonzero(
            torch.all(torch.eq(data.x, thread_nodetype), dim=1),
            as_tuple=False,
        ).flatten()

    def _get_thread_sorted_event_sequence(
        self,
        data: Data,
        thread_node_idx: int,
    ) -> List[str]:
        """
        Extract the timestamp-sorted event sequence associated with a thread node.
        """
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]

        outgoing_edge_indices = torch.nonzero(
            edge_src == thread_node_idx, as_tuple=False
        ).flatten()
        incoming_edge_indices = torch.nonzero(
            edge_dst == thread_node_idx, as_tuple=False
        ).flatten()

        if outgoing_edge_indices.numel() == 0 and incoming_edge_indices.numel() == 0:
            return []

        edge_features = torch.cat(
            [
                data.edge_attr[incoming_edge_indices],
                data.edge_attr[outgoing_edge_indices],
            ],
            dim=0,
        )

        if edge_features.numel() == 0:
            return []

        sort_indices = torch.argsort(edge_features[:, -1], descending=False)
        sorted_edge_features = edge_features[sort_indices]

        eventname_indices = torch.nonzero(
            sorted_edge_features[:, :-1],
            as_tuple=False,
        )[:, -1].tolist()

        return [self.eventname_edgefeats[i] for i in eventname_indices]

    def _get_thread_neighboring_nodetypes(
        self,
        data: Data,
        thread_node_idx: int,
    ) -> torch.Tensor:
        """
        Extract the neighboring node-type count vector for a thread node.
        """
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]

        outgoing_edge_indices = torch.nonzero(
            edge_src == thread_node_idx, as_tuple=False
        ).flatten()
        incoming_edge_indices = torch.nonzero(
            edge_dst == thread_node_idx, as_tuple=False
        ).flatten()

        neighboring_node_indices = []

        if outgoing_edge_indices.numel() > 0:
            unique_outgoing_edges = torch.unique(
                data.edge_index[:, outgoing_edge_indices],
                dim=1,
            )
            neighboring_node_indices.append(unique_outgoing_edges[1])

        if incoming_edge_indices.numel() > 0:
            unique_incoming_edges = torch.unique(
                data.edge_index[:, incoming_edge_indices],
                dim=1,
            )
            neighboring_node_indices.append(unique_incoming_edges[0])

        if not neighboring_node_indices:
            return torch.zeros(
                (1, len(self.nodetype_nodefeats)),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        thread_neighbors = torch.unique(torch.cat(neighboring_node_indices))
        thread_neighbors = thread_neighbors[thread_neighbors != thread_node_idx]

        if thread_neighbors.numel() == 0:
            return torch.zeros(
                (1, len(self.nodetype_nodefeats)),
                dtype=data.x.dtype,
                device=data.x.device,
            )

        neighboring_nodetype_counts = torch.sum(
            data.x[thread_neighbors],
            dim=0,
        ).view(1, -1)

        return neighboring_nodetype_counts

    def fit_count_vectorizer(self, train_dataset: List[Data]) -> None:
        """
        Fit the N-gram count vectorizer on all thread-level event sequences
        extracted from the training set.
        """
        self._ensure_metadata_is_ready()

        all_thread_sequences: List[str] = []

        for index, train_data in enumerate(train_dataset, start=1):
            thread_indices = self._thread_node_indices(train_data)

            for thread_node_idx in thread_indices.tolist():
                thread_sequence = self._get_thread_sorted_event_sequence(
                    data=train_data,
                    thread_node_idx=thread_node_idx,
                )
                if len(thread_sequence) >= self.N:
                    all_thread_sequences.append(" ".join(thread_sequence))

            print(
                f"{index} / {len(train_dataset)}: {train_data.name} -- extracted thread-level event-sequences",
                flush=True,
            )

        if not all_thread_sequences:
            raise ValueError(
                f"No thread-level event sequences with length >= {self.N} were found in the training dataset."
            )

        self.count_vectorizer.fit(all_thread_sequences)
        print(
            f"fitted {self.N}-gram count-vectorizer on all thread-level event-sequences",
            flush=True,
        )

    def _pool_embeddings(self, thread_embeddings: torch.Tensor) -> torch.Tensor:
        if self.pool == "sum":
            return torch.sum(thread_embeddings, dim=0)
        if self.pool == "mean":
            return torch.mean(thread_embeddings, dim=0)

        max_values, _ = torch.max(thread_embeddings, dim=0)
        return max_values

    def generate_graph_embedding(self, data: Data) -> torch.Tensor:
        """
        Generate a graph-level embedding for a single graph sample.
        """
        self._ensure_metadata_is_ready()

        thread_node_indices = self._thread_node_indices(data)
        ngram_feature_dim = len(self.count_vectorizer.get_feature_names_out())
        embedding_dim = len(self.nodetype_nodefeats) + ngram_feature_dim

        if thread_node_indices.numel() == 0:
            return torch.zeros(embedding_dim, dtype=torch.float32)

        all_thread_node_embeddings = []

        for thread_node_idx in thread_node_indices.tolist():
            thread_sequence = self._get_thread_sorted_event_sequence(
                data=data,
                thread_node_idx=thread_node_idx,
            )
            thread_sequence_text = " ".join(thread_sequence)

            thread_ngram_count_vector = self.count_vectorizer.transform(
                [thread_sequence_text]
            ).toarray()
            thread_ngram_count_tensor = torch.tensor(
                thread_ngram_count_vector,
                dtype=torch.float32,
            ).view(1, -1)

            thread_neighboring_nodetypes_tensor = self._get_thread_neighboring_nodetypes(
                data=data,
                thread_node_idx=thread_node_idx,
            ).to(dtype=torch.float32)

            thread_node_embedding = torch.cat(
                [thread_neighboring_nodetypes_tensor, thread_ngram_count_tensor],
                dim=1,
            )
            all_thread_node_embeddings.append(thread_node_embedding)

        stacked_thread_embeddings = torch.cat(all_thread_node_embeddings, dim=0)
        graph_embedding = self._pool_embeddings(stacked_thread_embeddings)

        return graph_embedding.to(dtype=torch.float32)

    def fit(
        self,
        train_dataset: List[Data],
        nodetype_nodefeats: List[str],
        eventname_edgefeats: List[str],
    ) -> None:
        """
        Fit the vectorizer, generate graph embeddings for the training set,
        and train the downstream Random Forest classifier.
        """
        self.nodetype_nodefeats = nodetype_nodefeats
        self.eventname_edgefeats = eventname_edgefeats

        self.fit_count_vectorizer(train_dataset)

        train_embeddings = []
        train_labels = []

        for index, train_data in enumerate(train_dataset, start=1):
            print(
                f"{index} / {len(train_dataset)}: {train_data.name} -- generate graph-embedding",
                flush=True,
            )
            graph_embedding = self.generate_graph_embedding(train_data)
            train_embeddings.append(graph_embedding.tolist())
            train_labels.append(1 if "malware" in train_data.name.lower() else 0)

        self.base_model.fit(X=train_embeddings, y=train_labels)
        print("fitted base-model on train dataset", flush=True)

    def predict(self, test_data: Data) -> int:
        """
        Predict the label of a single graph sample.

        Returns:
            int: malware = 1, benign = 0
        """
        test_graph_embedding = self.generate_graph_embedding(test_data)
        prediction = self.base_model.predict([test_graph_embedding.tolist()]).item()
        return int(prediction)
