"""Public entry point for Step 2 graph generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from build_computation_graph import first_step
from encode_graph_attributes import second_step
from normalize_edge_directions import set_edge_dir
from project_graphlets import projection


def _load_edge_data(output_path: Path) -> tuple[dict, dict, dict, dict, dict]:
    with open(output_path / "file_edge.json", "r", encoding="utf-8") as handle:
        file_data = json.load(handle)
    with open(output_path / "net_edge.json", "r", encoding="utf-8") as handle:
        net_data = json.load(handle)
    with open(output_path / "reg_edge.json", "r", encoding="utf-8") as handle:
        reg_data = json.load(handle)
    with open(output_path / "proc_edge.json", "r", encoding="utf-8") as handle:
        proc_data = json.load(handle)
    edge_data = {**file_data, **net_data, **reg_data, **proc_data}
    return file_data, net_data, reg_data, proc_data, edge_data


def run_step2_pipeline(
    index_name: str,
    output_dir: str,
    event_types_to_exclude: Optional[Sequence[str]] = None,
    malware_pid: Optional[int] = None,
    firststep_hash_debugging_mode: bool = False,
    elasticsearch_url: str = "http://localhost:9200",
    hostname: str = "localhost",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    first_step(
        idx=index_name,
        root_path=str(output_path),
        EventTypes_to_Exclude_set=set(event_types_to_exclude or []),
        firststep_hash_debugging_mode=firststep_hash_debugging_mode,
        elasticsearch_url=elasticsearch_url,
        hostname=hostname,
    )

    file_data, net_data, reg_data, proc_data, edge_data = _load_edge_data(output_path)

    set_edge_dir(
        graph_root=str(output_path),
        file_data=file_data,
        net_data=net_data,
        reg_data=reg_data,
        proc_data=proc_data,
    )
    second_step(str(output_path))
    projection(root_path=str(output_path), idx=index_name, edge_data=edge_data, mal_PID=malware_pid)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 2 graph generation.")
    parser.add_argument("--index-name", required=True, help="Elasticsearch index containing formatted ETW events.")
    parser.add_argument("--output-dir", required=True, help="Directory where Step 2 outputs will be written.")
    parser.add_argument(
        "--exclude-event",
        action="append",
        default=[],
        help="Event type to exclude. Repeat for multiple values, e.g. --exclude-event queryea0.",
    )
    parser.add_argument("--malware-pid", type=int, default=None, help="Target PID for malware-focused projection.")
    parser.add_argument(
        "--firststep-hash-debugging-mode",
        action="store_true",
        help="Keep raw hash strings instead of UUID-based IDs for debugging.",
    )
    parser.add_argument("--elasticsearch-url", default="http://localhost:9200", help="Elasticsearch base URL.")
    parser.add_argument("--hostname", default="localhost", help="Host name mixed into stable UID generation.")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_step2_pipeline(
        index_name=args.index_name,
        output_dir=args.output_dir,
        event_types_to_exclude=args.exclude_event,
        malware_pid=args.malware_pid,
        firststep_hash_debugging_mode=args.firststep_hash_debugging_mode,
        elasticsearch_url=args.elasticsearch_url,
        hostname=args.hostname,
    )
