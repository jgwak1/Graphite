"""Normalize edge directions for the union computation graph.

The raw union graph groups multiple event UIDs onto a single edge. This step
splits and re-orients those grouped events according to ETW semantics before
projection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import networkx as nx

PROCESS_TO_PROC = {
    "PROCESSSTART",
    "PROCESSSTOP",
    "JOBSTART",
    "JOBSTOP",
    "CPUBASEPRIORITYCHANGE",
    "CPUPRIORITYCHANGE",
    "IOPRIORITYCHANGE",
    "PAGEPRIORITYCHANGE",
    "IMAGELOAD",
    "IMAGEUNLOAD",
}
FILE_TO_THREAD = {
    "READ",
    "QUERYINFORMATION",
    "QUERYSECURITY",
    "QUERYEA",
    "DIRENUM",
    "DIRNOTIFY",
}
NETWORK_TO_THREAD = {11, 13, 15, 43}
REGISTRY_TO_THREAD = {35, 38, 39, 40}


Direction = Tuple[str, str]


def _parse_events(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip().strip("[]'") for part in value.split(",") if part.strip()]


def _classify_event(
    event_uid: str,
    file_data: Dict[str, dict],
    net_data: Dict[str, dict],
    reg_data: Dict[str, dict],
    proc_data: Dict[str, dict],
) -> Tuple[Optional[Direction], Optional[int]]:
    proc = proc_data.get(event_uid)
    if proc is not None:
        task_name = proc.get("Task Name")
        if task_name in PROCESS_TO_PROC:
            return ("src", "tar"), proc.get("TimeStamp")
        return ("tar", "src"), proc.get("TimeStamp")

    file_event = file_data.get(event_uid)
    if file_event is not None:
        task_name = file_event.get("Task Name")
        if task_name in FILE_TO_THREAD:
            return ("tar", "src"), file_event.get("TimeStamp")
        return ("src", "tar"), file_event.get("TimeStamp")

    net_event = net_data.get(event_uid)
    if net_event is not None:
        opcode = net_event.get("Opcode")
        if isinstance(opcode, str) and opcode.isdigit():
            opcode = int(opcode)
        if opcode in NETWORK_TO_THREAD:
            return ("tar", "src"), net_event.get("TimeStamp")
        return ("src", "tar"), net_event.get("TimeStamp")

    reg_event = reg_data.get(event_uid)
    if reg_event is not None:
        opcode = reg_event.get("Opcode")
        if isinstance(opcode, str) and opcode.isdigit():
            opcode = int(opcode)
        if opcode in REGISTRY_TO_THREAD:
            return ("tar", "src"), reg_event.get("TimeStamp")
        return ("src", "tar"), reg_event.get("TimeStamp")

    return None, None


def set_edge_dir(
    graph_root: str,
    file_data: Dict[str, dict],
    net_data: Dict[str, dict],
    reg_data: Dict[str, dict],
    proc_data: Dict[str, dict],
) -> None:
    graph_path = Path(graph_root) / "Union.GraphML"
    graph = nx.read_graphml(graph_path)
    original_edges = list(graph.edges(data=True))
    graph.remove_edges_from(list(graph.edges()))

    for src, tar, edge_attrs in original_edges:
        events = _parse_events(edge_attrs.get("name", ""))
        grouped: dict[Direction, list[tuple[int, str]]] = {}

        for event_uid in events:
            direction, timestamp = _classify_event(
                event_uid=event_uid,
                file_data=file_data,
                net_data=net_data,
                reg_data=reg_data,
                proc_data=proc_data,
            )
            if direction is None:
                continue
            ts_value = 0 if timestamp is None else int(timestamp)
            grouped.setdefault(direction, []).append((ts_value, event_uid))

        for direction, timestamped_events in grouped.items():
            ordered = [event_uid for _, event_uid in sorted(timestamped_events, key=lambda item: item[0])]
            edge_src = src if direction == ("src", "tar") else tar
            edge_tar = tar if direction == ("src", "tar") else src
            graph.add_edge(edge_src, edge_tar, name=str(ordered))

    nx.write_graphml(graph, Path(graph_root) / "graph.GraphML")
