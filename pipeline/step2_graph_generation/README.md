# Step 2: Graph Generation

This directory contains the public-facing Step 2 pipeline for converting formatted ETW events into graph artifacts used by the downstream Graphite workflow.

## Files

- `run_step2_pipeline.py`: end-to-end runner for Step 2
- `build_computation_graph.py`: construct the union computation graph from formatted events
- `normalize_edge_directions.py`: normalize grouped edge directions based on event semantics
- `encode_graph_attributes.py`: convert raw JSON node/edge attributes into Graphite feature pickles
- `project_graphlets.py`: project process-rooted subgraphs for benign or malware-focused runs

## Inputs

Step 2 expects formatted event data from Step 1, typically stored in Elasticsearch and exported into the JSON side artifacts written by `build_computation_graph.py`.

## Usage

```bash
python run_step2_pipeline.py \
  --index-name <elasticsearch_index> \
  --output-dir <output_directory> \
  --elasticsearch-url http://localhost:9200
```

For malware-focused projection, add:

```bash
--malware-pid <pid>
```

## Outputs

The pipeline writes:
- `Union.GraphML`
- `graph.GraphML`
- per-type raw JSON node/edge dictionaries
- `global_node_attribute.pickle`
- `global_edge_attribute.pickle`
- projected subgraph directories

## Notes

- Keep `resources/VN.txt` and `resources/VoN.txt` in this directory when using `encode_graph_attributes.py`.
- This public version intentionally keeps one exposed runner and omits older internal script variants.
