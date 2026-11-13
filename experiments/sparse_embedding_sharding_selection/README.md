# Sparse recommendation embedding ownership and pooled jagged reductions

Round 96. **Synthetic inputs; independently_reproduced=false.** The separately
sealed latency section measures original local-CPU Python functions only, not
TorchRec, FBGEMM, GPU communication or a recommendation model.

```powershell
python -m experiments.sparse_embedding_sharding_selection.run
python -m experiments.sparse_embedding_sharding_selection.run --check
python -m unittest tests.test_sparse_embedding_sharding_selection
```

Default rendering and `--check` preserve the measured samples in `latency.json`.
To deliberately replace those samples, run with `--measure`; that command also
rebuilds the report and views. It cannot be combined with `--check`.

The original model compares table/row/column ownership against dense SUM and
ideal unweighted MEAN, plus weighted SUM. It checks every table-gradient
coordinate by independent central differences, including repeated IDs and
empty bags. Wrong controls cover local means, dedup without multiplicity and
dimension reassembly.

The pinned TorchRec RW/TWRW mean callback uses a global count/weight divisor
**plus 1e-6**. That is explicitly modeled as a separate numerical policy, not
called an upstream bug or bitwise identical to the ideal mathematical mean.
Weighted-MEAN support is not assessed; PyTorch EmbeddingBag restrictions are
not generalized to TorchRec's callback.

Training RW input bucketizes then AlltoAll; output is pooled ReduceScatter.
TW/CW use AlltoAll distributions, with column ordering restored. Inference
paths differ. The November 11, 2026 source fix maps subgroup-local root zero
to its global rank for DATA_PARALLEL initialization.

The workbench counts one table's peak weight shard, logical ID delivery and
dense pooled partials. It does not call those quantities network wire bytes.
Optimizer state, offsets/length metadata, activations, access skew and real
collective topology are excluded.
