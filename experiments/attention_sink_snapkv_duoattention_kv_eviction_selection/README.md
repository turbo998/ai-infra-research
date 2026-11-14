# Long-Context KV Cache Budget Control, Attention Sinks & Retrieval-Head Separation (`StreamingLLM` vs. `H2O` vs. `SnapKV` vs. `DuoAttention`) Selection (`attention_sink_snapkv_duoattention_kv_eviction_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 184**: comparing `Naive Sliding Window Without Sinks` (Tier 1), `StreamingLLM (4 Attention Sinks + Sliding Window)` (Tier 2), `H2O Dynamic Heavy-Hitter Oracle` (Tier 3), and `DuoAttention (20% Retrieval Heads + 80% Streaming Heads) + SnapKV` (Tier 4).

## Reproduction

```bash
python -m experiments.attention_sink_snapkv_duoattention_kv_eviction_selection.run
python -m unittest tests/test_attention_sink_snapkv_duoattention_kv_eviction_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_kv_compression_papers`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
