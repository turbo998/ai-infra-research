# FlashInfer Multi-Level Cascade Attention & Base-2 LSE State Merging Simulator

Deterministic CPU/offline simulator grounded in `flashinfer-ai/flashinfer` (`commit f64b4ba8762ded12a4f4b9a2092974c9a395e7ca`):
- `flashinfer/cascade.py` (`MultiLevelCascadeAttentionWrapper`, `merge_state_in_place`)
- `include/flashinfer/attention/cascade.cuh` (`MergeStateKernel`, `MergeStateInPlaceKernel`, `PersistentVariableLengthMergeStatesKernel`)

## Usage

```bash
python experiments/flashinfer_cascade_paged_kv/run.py
python experiments/flashinfer_cascade_paged_kv/run.py --check
```
