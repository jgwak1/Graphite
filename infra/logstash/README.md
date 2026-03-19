# Logstash example for Graphite

This directory contains a minimal public-facing Logstash example for the ETW-to-graph pipeline used in Graphite.

## File

- `logstash_pipeline_example.conf`: sanitized example ingestion pipeline that reflects Graphite’s ETW-to-Elasticsearch data path. It accepts ETW-derived events over HTTP, applies lightweight pre-index filtering to drop events associated with common Windows system accounts, and forwards the remaining events to Elasticsearch for downstream computation-graph construction.

## Why this file is included

This config shows the ingestion shape of the pipeline without exposing internal deployment details. It is included as a supporting artifact rather than as the main algorithmic contribution of the repository.

The system-account filter is included as an example of an ingestion-stage noise-reduction heuristic. In Graphite-style ETW collection, routine background activity from common Windows system accounts can contribute substantial event volume. Filtering part of that activity before indexing can reduce Elasticsearch load, reduce graph size, and make downstream graph construction more tractable in high-volume settings.

This should be read as an environment-specific preprocessing choice, not as a claim that system-account activity is universally unimportant or that Graphite’s core method depends on this exact filter. The core Graphite artifact is the downstream computation-graph construction, graph projection, and classification pipeline.

## Environment variables

Set these before running Logstash:

- `ELASTICSEARCH_HOST`: Elasticsearch endpoint, for example `http://localhost:9200`
- `LOGSTASH_INDEX`: target Elasticsearch index name

## Notes

- This is a sanitized example configuration intended for documentation and reproducibility.
- It is intentionally smaller than the original internal deployment setup.
- Runtime-specific files such as `logstash.yml`, `pipelines.yml`, `jvm.options`, and `startup.options` are omitted because they are deployment boilerplate rather than project-specific artifacts.