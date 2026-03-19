# Step 1: ETL / Ingestion / Formatting

This step normalizes legacy ETW log entries before graph construction.

It includes:
- legacy text-log parsing into event records
- nested event flattening
- field filtering and cleanup
- Elasticsearch index reformatting for downstream graph generation