"""Generate projected graphlets from the normalized Step 2 graph."""

from __future__ import annotations

import ast
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Set

from igraph import Graph


def _edge_event_ids(edge_name: str) -> list[str]:
    return [str(event_id) for event_id in ast.literal_eval(edge_name)]


def _load_proc_sets(root_path: str) -> tuple[set[str], set[str]]:
    with open(os.path.join(root_path, "proc_node.json"), "r", encoding="utf-8") as handle:
        proc_nodes = set(json.load(handle).keys())
    with open(os.path.join(root_path, "proc_thread.json"), "r", encoding="utf-8") as handle:
        proc_thread_nodes = set(json.load(handle).keys())
    return proc_nodes, proc_thread_nodes


def _sanitize_component(value: object) -> str:
    return str(value).replace(os.sep, "_").replace(" ", "_")


def _find_process_start_edges(graph: Graph, edge_data: Dict[str, dict]):
    process_start_edges = []
    for edge in graph.es:
        try:
            event_ids = _edge_event_ids(edge["name"])
        except Exception:
            continue
        for event_id in event_ids:
            task_name = str(edge_data.get(event_id, {}).get("Task Name", "")).lower()
            if task_name == "processstart":
                process_start_edges.append((edge, event_id, edge_data.get(event_id, {})))
                break
    process_start_edges.sort(key=lambda item: item[2].get("TimeStamp", 0))
    return process_start_edges


def earliest_time(graph: Graph, edge_data: Dict[str, dict], node_index: int, edge_index: int):
    event_ids = _edge_event_ids(graph.es[edge_index]["name"])
    if not event_ids:
        return math.inf, ""
    first_event = edge_data.get(event_ids[0], {})
    timestamp = first_event.get("TimeStamp", math.inf)
    task_name = first_event.get("Task Name", "")
    if graph.vs[node_index]["taint"] < timestamp and task_name in {"PROCESSSTART", "THREADSTART"}:
        return timestamp, task_name
    return math.inf, ""


def tainted_subgraph(
    subgraph_path,
    graph,
    current_list,
    next_list,
    tainted_nodes,
    edge_data,
    edge_min_ts,
    sub_edge_set,
    proc_node,
    proc_thread_node,
):
    while next_list:
        current_list = next_list
        next_list = []
        while current_list:
            node = current_list.pop(0)
            node_name = graph.vs[node]["name"]
            if node_name in proc_node or node_name in proc_thread_node:
                for edge_idx in graph.incident(node, "out"):
                    target = graph.es[edge_idx].target
                    ts, _ = earliest_time(graph, edge_data, node, edge_idx)
                    if graph.vs[target]["taint"] > ts:
                        graph.vs[target]["taint"] = ts
                        next_list.append(target)
                        tainted_nodes.add(target)
    subgraph(subgraph_path, graph, tainted_nodes, edge_data, edge_min_ts, sub_edge_set, proc_node, proc_thread_node)


def subgraph(subgraph_path, graph, tainted_nodes, edge_data, edge_min_ts, sub_edge_set, proc_node, proc_thread_node):
    while tainted_nodes:
        node = tainted_nodes.pop()
        node_name = graph.vs[node]["name"]
        if node_name in proc_node or node_name in proc_thread_node:
            for edge_idx in graph.incident(node, "all"):
                src = graph.es[edge_idx].source
                tar = graph.es[edge_idx].target
                edge_min_ts[edge_idx] = min(graph.vs[src]["taint"], graph.vs[tar]["taint"])
                sub_edge_set.add(edge_idx)
    raw_subgraph = graph.subgraph_edges(sub_edge_set)
    exclude_events_based_on_cutoff(subgraph_path, edge_min_ts, edge_data, graph, raw_subgraph)


def exclude_events_based_on_cutoff(subgraph_path, edge_min_ts, edge_data, full_graph, subgraph_obj):
    edge_ids_to_delete = []
    for edge in subgraph_obj.es:
        source_id = subgraph_obj.vs[edge.source]["id"]
        target_id = subgraph_obj.vs[edge.target]["id"]
        full_edge_id = full_graph.get_eid(int(source_id[1:]), int(target_id[1:]))
        cutoff_time = edge_min_ts[full_edge_id]
        kept_events = []
        for event_id in _edge_event_ids(edge["name"]):
            timestamp = edge_data.get(event_id, {}).get("TimeStamp")
            if timestamp is not None and timestamp >= cutoff_time:
                kept_events.append(event_id)
        if kept_events:
            edge["name"] = str(kept_events)
        else:
            edge_ids_to_delete.append(subgraph_obj.get_eid(edge.source, edge.target))
    subgraph_obj.delete_edges(edge_ids_to_delete)
    subgraph_obj.write_graphml(os.path.join(subgraph_path, "new_graph.graphml"))


def attributes(root_path: str, subgraph_path: str) -> Graph:
    with open(os.path.join(root_path, "global_node_attribute.pickle"), "rb") as handle:
        global_node_attributes = pickle.load(handle)
    with open(os.path.join(root_path, "global_edge_attribute.pickle"), "rb") as handle:
        global_edge_attributes = pickle.load(handle)

    graph = Graph.Read_GraphML(os.path.join(subgraph_path, "new_graph.graphml"))
    node_attributes = {vertex["name"]: global_node_attributes.get(vertex["name"], {}) for vertex in graph.vs}
    with open(os.path.join(subgraph_path, "node_attribute.pickle"), "wb") as handle:
        pickle.dump(node_attributes, handle)

    edge_attributes = {}
    for edge in graph.es:
        for event_id in _edge_event_ids(edge["name"]):
            edge_attributes[event_id] = global_edge_attributes.get(event_id, {})
    with open(os.path.join(subgraph_path, "edge_attribute.pickle"), "wb") as handle:
        pickle.dump(edge_attributes, handle)
    return graph


def _build_subgraph(
    root_path: str,
    output_dir: str,
    graph: Graph,
    root_vertex: int,
    root_timestamp: int,
    edge_data: Dict[str, dict],
    proc_node: set[str],
    proc_thread_node: set[str],
) -> None:
    graph.vs["taint"] = math.inf
    graph.vs[root_vertex]["taint"] = root_timestamp
    tainted_subgraph(
        output_dir,
        graph,
        current_list=[],
        next_list=[root_vertex],
        tainted_nodes={root_vertex},
        edge_data=edge_data,
        edge_min_ts={},
        sub_edge_set=set(),
        proc_node=proc_node,
        proc_thread_node=proc_thread_node,
    )
    attributes(root_path, output_dir)


def benign(root_path: str, idx: str, ids: set[int], edge_data: Dict[str, dict], proc_node: set[str], proc_thread_node: set[str]) -> None:
    graph = Graph.Read_GraphML(os.path.join(root_path, "graph.GraphML"))
    covered_proc_nodes: Set[str] = set()

    for edge, event_id, event in _find_process_start_edges(graph, edge_data):
        root_name = graph.vs[edge.target]["name"]
        if edge.target in ids or root_name in covered_proc_nodes:
            continue
        ids.add(edge.target)
        image_name = _sanitize_component(str(event.get("ImageName", "unknown")).split("\\")[-1])
        process_id = _sanitize_component(event.get("ProcessID", "NA"))
        parent_process_id = _sanitize_component(event.get("ProcessId", "NA"))
        thread_id = _sanitize_component(event.get("ThreadId", "NA"))
        timestamp = _sanitize_component(event.get("TimeStamp", "NA"))
        subgraph_dir = os.path.join(
            root_path,
            f"Benign_Sample_P3_{idx}_{image_name}_PID{process_id}_PId{parent_process_id}_TId{thread_id}_TS{timestamp}",
        )
        os.makedirs(subgraph_dir, exist_ok=True)
        _build_subgraph(
            root_path=root_path,
            output_dir=subgraph_dir,
            graph=graph,
            root_vertex=edge.target,
            root_timestamp=event.get("TimeStamp", math.inf),
            edge_data=edge_data,
            proc_node=proc_node,
            proc_thread_node=proc_thread_node,
        )
        subgraph = Graph.Read_GraphML(os.path.join(subgraph_dir, "new_graph.graphml"))
        covered_proc_nodes.update({vertex["name"] for vertex in subgraph.vs if "PROC-NODE" in vertex["name"]})


def malware(root_path: str, idx: str, mal_PID: int, ids: set[int], edge_data: Dict[str, dict], proc_node: set[str], proc_thread_node: set[str]) -> None:
    graph = Graph.Read_GraphML(os.path.join(root_path, "graph.GraphML"))
    for edge, event_id, event in _find_process_start_edges(graph, edge_data):
        if str(event.get("ProcessID")) != str(mal_PID):
            continue
        if edge.target in ids:
            continue
        ids.add(edge.target)
        subgraph_dir = os.path.join(root_path, f"SUBGRAPH_P3_{idx}")
        os.makedirs(subgraph_dir, exist_ok=True)
        _build_subgraph(
            root_path=root_path,
            output_dir=subgraph_dir,
            graph=graph,
            root_vertex=edge.target,
            root_timestamp=event.get("TimeStamp", math.inf),
            edge_data=edge_data,
            proc_node=proc_node,
            proc_thread_node=proc_thread_node,
        )
        break


def projection(root_path: str, idx: str, edge_data: Dict[str, dict], mal_PID: int | None = None) -> None:
    ids: set[int] = set()
    proc_node, proc_thread_node = _load_proc_sets(root_path)
    if mal_PID is None:
        benign(root_path, idx, ids, edge_data, proc_node, proc_thread_node)
    else:
        malware(root_path, idx, mal_PID, ids, edge_data, proc_node, proc_thread_node)
