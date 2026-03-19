from typing import Any, Dict, Iterable, Iterator, Sequence

from elasticsearch import Elasticsearch, helpers

try:
    from .field_selection import filter_selected_fields
    from .flatten_event_record import flatten_record
except ImportError:
    from field_selection import filter_selected_fields
    from flatten_event_record import flatten_record


def flatten_elasticsearch_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a single Elasticsearch hit into a single-level ETW event record.
    """
    source = hit.get("_source", {})
    return flatten_record(source)


def transform_elasticsearch_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten and field-filter a single Elasticsearch hit.
    """
    flattened = flatten_elasticsearch_hit(hit)
    return filter_selected_fields(flattened)


def _scan_index(es: Elasticsearch, index_name: str) -> Iterable[Dict[str, Any]]:
    """
    Stream all documents from an index without relying on a large result window.
    """
    return helpers.scan(
        client=es,
        index=index_name,
        query={"query": {"match_all": {}}},
        preserve_order=False,
        scroll="5m",
    )


def _iter_reformatted_actions(source_hits: Iterable[Dict[str, Any]], target_index: str) -> Iterator[Dict[str, Any]]:
    for hit in source_hits:
        cleaned = transform_elasticsearch_hit(hit)
        yield {
            "_index": target_index,
            "_source": cleaned,
        }


def reformat_elasticsearch_indices(
    unformatted_indices: Sequence[str],
    elasticsearch_url: str = "http://localhost:9200",
    target_suffix: str = "_formatted",
    request_timeout: int = 30,
) -> None:
    """
    Read unformatted Elasticsearch ETW indices, flatten and clean each log entry,
    and write the results into new formatted indices.
    """
    es = Elasticsearch(elasticsearch_url, request_timeout=request_timeout)

    for source_index in unformatted_indices:
        target_index = f"{source_index}{target_suffix}"

        if not es.indices.exists(index=target_index):
            es.indices.create(
                index=target_index,
                body={"settings": {"index": {"number_of_replicas": 0}}},
            )

        print(f"[start] formatting index: {source_index}", flush=True)
        source_hits = _scan_index(es, source_index)
        actions = _iter_reformatted_actions(source_hits, target_index)

        try:
            success_count, _ = helpers.bulk(client=es, actions=actions, refresh=True)
            print(f"[done] wrote {success_count} formatted documents to {target_index}", flush=True)
        except Exception as exc:
            print(f"[upload error] {exc}", flush=True)
            raise

        print(f"[note] original index retained: {source_index}", flush=True)


def main() -> None:
    raise SystemExit(
        "Import and call reformat_elasticsearch_indices([...], elasticsearch_url=...)."
    )


if __name__ == "__main__":
    main()
