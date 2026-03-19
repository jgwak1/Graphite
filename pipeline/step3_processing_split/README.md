# Step 3: Processing and Train/Test Split

This step converts projected graph artifacts from Step 2 into model-ready processed samples and then creates train/test splits.

## Files
- `process_graph_data.py`: reads `new_graph.graphml`, `node_attribute.pickle`, and `edge_attribute.pickle`, then writes model-ready pickles.
- `split_dataset.py`: copies processed pickle files into `train/` and `test/` directories.
- `run_step3_pipeline.py`: public-facing runner for the full Step 3 flow.

## Expected input from Step 2
A root directory containing sample folders under:
- `Benign/Benign_Sample_*/`
- `Malware/Malware_Sample_*/`

Each sample directory should contain:
- `new_graph.graphml`
- `node_attribute.pickle`
- `edge_attribute.pickle`

## Example
```bash
python -m pipeline.step3_processing_split.run_step3_pipeline \
  --input-root /path/to/step2_outputs \
  --working-root /path/to/processed_samples \
  --split-output-root /path/to/dataset_split \
  --train-ratio 0.8
```
