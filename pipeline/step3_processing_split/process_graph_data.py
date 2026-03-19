from __future__ import annotations

import argparse
import ast
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx
import numpy as np
from sklearn import preprocessing


class GraphDataProcessor:
    """
    Convert projected graph artifacts into model-ready samples.

    Expected per-sample inputs:
    - new_graph.graphml
    - node_attribute.pickle
    - edge_attribute.pickle

    Output sample format:
    {
        "x": node feature matrix,
        "edge_list": [src_ids, dst_ids],
        "y": class label,
        "edge_attr": edge feature matrix,
    }
    """

    def __init__(
        self,
        node_attribute_list: Optional[Sequence[str]] = None,
        edge_attribute_list: Optional[Sequence[str]] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.node_attribute_list = list(node_attribute_list) if node_attribute_list else None
        self.edge_attribute_list = list(edge_attribute_list) if edge_attribute_list else None

    def process_single_graph(
        self,
        sample_dir: str | Path,
        class_label: int,
        concat_events: bool = True,
        compute_order: bool = True,
    ) -> Dict[str, Any]:
        sample_dir = Path(sample_dir)
        graph = nx.read_graphml(sample_dir / "new_graph.graphml")
        edge_dict = self._load_pickle(sample_dir / "edge_attribute.pickle")
        node_dict = self._load_pickle(sample_dir / "node_attribute.pickle")

        events_order = self._get_events_order(edge_dict) if compute_order else None

        if concat_events:
            edge_list, edge_names, freq_vector, normalized_count_dict = self._extract_edge_list_v3(
                graph, edge_dict
            )
        else:
            edge_list, edge_names = self._extract_edge_list(graph)
            freq_vector, normalized_count_dict = None, None

        x = self._extract_node_attributes(graph, node_dict)

        if concat_events and compute_order:
            edge_attr = self._extract_edge_attr_v3(
                edge_names=edge_names,
                freq_vector=freq_vector,
                events_order=events_order,
                normalized_count_dict=normalized_count_dict,
            )
        else:
            edge_attr = self._extract_edge_attributes(edge_names, edge_dict)

        if self.debug:
            print(f"Processed: {sample_dir}")
            print(f"  nodes={len(x)} edges={len(edge_attr)}")
            if x:
                print(f"  node_dim={len(x[0])}")
            if edge_attr:
                print(f"  edge_dim={len(edge_attr[0])}")

        return {
            "x": x,
            "edge_list": edge_list,
            "y": int(class_label),
            "edge_attr": edge_attr,
        }

    def process_dataset(
        self,
        input_root: str | Path,
        output_root: str | Path,
        data_type: str,
        concat_events: bool = True,
        compute_order: bool = True,
    ) -> None:
        """
        Process all samples under:
            input_root/<data_type>/<data_type>_Sample_*/
        and save pickled outputs under:
            output_root/Processed_<data_type>/
        """
        input_root = Path(input_root)
        output_root = Path(output_root)

        if data_type not in {"Benign", "Malware"}:
            raise ValueError("data_type must be either 'Benign' or 'Malware'")

        class_label = 1 if data_type == "Malware" else 0
        source_dir = input_root / data_type
        save_dir = output_root / f"Processed_{data_type}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source directory: {source_dir}")

        for sample_dir in sorted(source_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            if f"{data_type}_Sample_" not in sample_dir.name:
                continue

            sample = self.process_single_graph(
                sample_dir=sample_dir,
                class_label=class_label,
                concat_events=concat_events,
                compute_order=compute_order,
            )
            output_path = save_dir / f"Processed_{sample_dir.name}.pickle"
            self._save_pickle(output_path, sample)

            if self.debug:
                print(f"Saved: {output_path}")

    def _extract_node_attributes(
        self,
        graph: nx.Graph,
        node_dict: Dict[str, Dict[str, Any]],
    ) -> List[List[Any]]:
        x: List[List[Any]] = []
        for node_id, node_data in graph.nodes(data=True):
            node_name = node_data["name"]
            node_attr: List[Any] = []
            if node_name not in node_dict:
                raise KeyError(f"Missing node attributes for node name: {node_name}")

            attribute_dict = node_dict[node_name]
            for key, value in attribute_dict.items():
                if self.node_attribute_list and key not in self.node_attribute_list:
                    continue
                if isinstance(value, list):
                    node_attr.extend(value)
                else:
                    node_attr.append(value)
            x.append(node_attr)
        return x

    def _get_events_order(self, edge_dict: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        ordered_events = []
        total = len(edge_dict.keys())
        for name, attrs in edge_dict.items():
            ordered_events.append([name, attrs["TimeStamp"]])
        ordered_events.sort(key=lambda x: x[1])

        event_order: Dict[str, float] = {}
        for i, values in enumerate(ordered_events):
            event_order[values[0]] = (i + 1) / total
        return event_order

    def _extract_edge_list_v3(
        self,
        graph: nx.Graph,
        edge_dict: Dict[str, Dict[str, Any]],
    ):
        u: List[int] = []
        v: List[int] = []
        edge_names: List[str] = []

        task_freqs = []
        status_code_freqs = []
        read_op_count = []
        write_op_count = []
        read_transfer_kb = []
        write_transfer_kb = []
        size = []
        handle_count = []
        image_size = []
        subprocess_tag_freqs = []
        token_elevation_type_freqs = []
        token_is_elevated_freqs = []
        mandatory_label_freqs = []
        appid_freqs = []
        status_freqs = []
        disposition_freqs = []

        for src, dst, edge_data in graph.edges(data=True):
            src_id = int(str(src).strip("n"))
            dst_id = int(str(dst).strip("n"))
            u.append(src_id)
            v.append(dst_id)

            names = ast.literal_eval(edge_data["name"])
            first_name = names[0]
            ordered = [[first_name, edge_dict[first_name]["TimeStamp"]]]

            task_vector = edge_dict[first_name]["Task Name"].copy()
            status_code_vector = edge_dict[first_name]["StatusCode"].copy()
            subprocess_vector = edge_dict[first_name]["SubProcessTag"].copy()
            token_elevation_type_vector = edge_dict[first_name]["ProcessTokenElevationType"].copy()
            token_is_elevated_vector = edge_dict[first_name]["ProcessTokenIsElevated"].copy()
            mandatory_label_vector = edge_dict[first_name]["MandatoryLabel"].copy()
            appid_vector = edge_dict[first_name]["PackageRelativeAppId"].copy()
            status_vector = edge_dict[first_name]["Status"].copy()
            disposition_vector = edge_dict[first_name]["Disposition"].copy()

            read_count = int(edge_dict[first_name]["ReadOperationCount"])
            write_count = int(edge_dict[first_name]["WriteOperationCount"])
            read_kb = int(edge_dict[first_name]["ReadTransferKiloBytes"])
            write_kb = int(edge_dict[first_name]["WriteTransferKiloBytes"])
            size_sum = int(edge_dict[first_name]["size"])
            image_size_sum = int(edge_dict[first_name]["ImageSize"])
            handle_count_sum = int(edge_dict[first_name]["HandleCount"])

            for i in range(len(names) - 1):
                name = names[i + 1]
                ordered.append([name, edge_dict[name]["TimeStamp"]])

                for j, value in enumerate(edge_dict[name]["Task Name"]):
                    task_vector[j] += value
                for j, value in enumerate(edge_dict[name]["StatusCode"]):
                    status_code_vector[j] += value
                for j, value in enumerate(edge_dict[name]["SubProcessTag"]):
                    subprocess_vector[j] += value
                for j, value in enumerate(edge_dict[name]["ProcessTokenElevationType"]):
                    token_elevation_type_vector[j] += value
                for j, value in enumerate(edge_dict[name]["ProcessTokenIsElevated"]):
                    token_is_elevated_vector[j] += value
                for j, value in enumerate(edge_dict[name]["MandatoryLabel"]):
                    mandatory_label_vector[j] += value
                for j, value in enumerate(edge_dict[name]["PackageRelativeAppId"]):
                    appid_vector[j] += value
                for j, value in enumerate(edge_dict[name]["Status"]):
                    status_vector[j] += value
                for j, value in enumerate(edge_dict[name]["Disposition"]):
                    disposition_vector[j] += value

                read_count += int(edge_dict[name]["ReadOperationCount"])
                write_count += int(edge_dict[name]["WriteOperationCount"])
                read_kb += int(edge_dict[name]["ReadTransferKiloBytes"])
                write_kb += int(edge_dict[name]["WriteTransferKiloBytes"])
                size_sum += int(edge_dict[name]["size"])
                image_size_sum += int(edge_dict[name]["ImageSize"])
                handle_count_sum += int(edge_dict[name]["HandleCount"])

            ordered.sort(key=lambda x: x[1])
            edge_names.append(ordered[0][0])

            task_freqs.append(task_vector)
            status_code_freqs.append(status_code_vector)
            subprocess_tag_freqs.append(subprocess_vector)
            token_elevation_type_freqs.append(token_elevation_type_vector)
            token_is_elevated_freqs.append(token_is_elevated_vector)
            mandatory_label_freqs.append(mandatory_label_vector)
            appid_freqs.append(appid_vector)
            status_freqs.append(status_vector)
            disposition_freqs.append(disposition_vector)

            read_op_count.append(read_count)
            write_op_count.append(write_count)
            read_transfer_kb.append(read_kb)
            write_transfer_kb.append(write_kb)
            size.append(size_sum)
            image_size.append(image_size_sum)
            handle_count.append(handle_count_sum)

        normalized_count_dict = {
            "ReadOperationCount": preprocessing.normalize([np.array(read_op_count)]).tolist()[0],
            "WriteOperationCount": preprocessing.normalize([np.array(write_op_count)]).tolist()[0],
            "ReadTransferKiloBytes": preprocessing.normalize([np.array(read_transfer_kb)]).tolist()[0],
            "WriteTransferKiloBytes": preprocessing.normalize([np.array(write_transfer_kb)]).tolist()[0],
            "size": preprocessing.normalize([np.array(size)]).tolist()[0],
            "HandleCount": preprocessing.normalize([np.array(handle_count)]).tolist()[0],
            "ImageSize": preprocessing.normalize([np.array(image_size)]).tolist()[0],
        }

        freq_vector = {
            "Task Name": task_freqs,
            "StatusCode": status_code_freqs,
            "SubProcessTag": subprocess_tag_freqs,
            "ProcessTokenElevationType": token_elevation_type_freqs,
            "ProcessTokenIsElevated": token_is_elevated_freqs,
            "MandatoryLabel": mandatory_label_freqs,
            "PackageRelativeAppId": appid_freqs,
            "Status": status_freqs,
            "Disposition": disposition_freqs,
        }

        return [u, v], edge_names, freq_vector, normalized_count_dict

    def _extract_edge_attr_v3(
        self,
        edge_names: Sequence[str],
        freq_vector: Dict[str, List[List[Any]]],
        events_order: Dict[str, float],
        normalized_count_dict: Dict[str, List[float]],
    ) -> List[List[Any]]:
        edge_attr = []
        for i in range(len(edge_names)):
            scalar_values = [
                normalized_count_dict["ReadOperationCount"][i],
                normalized_count_dict["WriteOperationCount"][i],
                normalized_count_dict["ReadTransferKiloBytes"][i],
                normalized_count_dict["WriteTransferKiloBytes"][i],
                normalized_count_dict["size"][i],
                normalized_count_dict["HandleCount"][i],
                normalized_count_dict["ImageSize"][i],
            ]
            vector_values = (
                freq_vector["Task Name"][i]
                + freq_vector["StatusCode"][i]
                + freq_vector["SubProcessTag"][i]
                + freq_vector["ProcessTokenElevationType"][i]
                + freq_vector["ProcessTokenIsElevated"][i]
                + freq_vector["MandatoryLabel"][i]
                + freq_vector["PackageRelativeAppId"][i]
                + freq_vector["Status"][i]
                + freq_vector["Disposition"][i]
            )
            edge_attr.append(vector_values + scalar_values + [events_order[edge_names[i]]])
        return edge_attr

    def _extract_edge_attributes(
        self,
        edge_names: Sequence[str],
        edge_dict: Dict[str, Dict[str, Any]],
    ) -> List[List[Any]]:
        edge_attr: List[List[Any]] = []
        for name in edge_names:
            if name not in edge_dict:
                raise KeyError(f"Missing edge attributes for edge name: {name}")
            values: List[Any] = []
            for key, value in edge_dict[name].items():
                if self.edge_attribute_list and key not in self.edge_attribute_list:
                    continue
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
            edge_attr.append(values)
        return edge_attr

    def _extract_edge_list(self, graph: nx.Graph):
        u: List[int] = []
        v: List[int] = []
        edge_names: List[str] = []

        for src, dst, edge_data in graph.edges(data=True):
            src_id = int(str(src).strip("n"))
            dst_id = int(str(dst).strip("n"))
            names = ast.literal_eval(edge_data["name"])
            for name in names:
                u.append(src_id)
                v.append(dst_id)
                edge_names.append(name)

        return [u, v], edge_names

    @staticmethod
    def _load_pickle(path: str | Path):
        with Path(path).open("rb") as fp:
            return pickle.load(fp)

    @staticmethod
    def _save_pickle(path: str | Path, data: Any) -> None:
        with Path(path).open("wb") as fp:
            pickle.dump(data, fp)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process projected graph artifacts into model-ready samples.")
    parser.add_argument("--input-root", required=True, help="Root directory containing Benign/ and/or Malware/ sample folders.")
    parser.add_argument("--output-root", required=True, help="Directory where Processed_Benign/ and Processed_Malware/ will be written.")
    parser.add_argument(
        "--data-type",
        choices=["Benign", "Malware", "both"],
        default="both",
        help="Which sample set to process.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    processor = GraphDataProcessor(debug=args.debug)

    targets = [args.data_type] if args.data_type in {"Benign", "Malware"} else ["Benign", "Malware"]
    for data_type in targets:
        processor.process_dataset(
            input_root=args.input_root,
            output_root=args.output_root,
            data_type=data_type,
            concat_events=True,
            compute_order=True,
        )


if __name__ == "__main__":
    main()
