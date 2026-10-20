# TorchTitan Cuspy CUPTI CUDA Graph Annotations, Muse Spark `muse_spark_attention_qk` Fused RoPE Region & TransformerEngine JAX Single-NVLink Fused EP Prepare+Dispatch

> **中文版**: [torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.md](torchtitan-cuspy-cupti-graph-muse-spark-rope-norm-te-nvl-fused-ep.md)  
> **Evidence Class**: `implementation` (verified upstream commits) + `own-offline` (`experiments/torchtitan_cuspy_graph_muse_rope_te_ep/run.py --check`)

## 1. First-Hand Source Verification & Architectural Mechanics

1. **TorchTitan Cuspy CUPTI CUDA Graph Annotations & Python Launch Stack Capture (`pytorch/torchtitan` commit `b93d5c0cf518b4404ceab998adc529fd60ebf301`, `torchtitan/distributed/cuda_graph.py` & `torchtitan/observability/profiler.py`)**:
   - Adds `get_cuspy_cuda_graph_annotation_config(enable_event_node_ids, record_py_stacks)` and plumbs `annotation_config` into `torch.cuda.graph` across `forward_backward`, `optimizer`, and `validation` graph wrappers;
   - Uses `"backend": "cupti"` to discover `mark_kernels` scopes via CUPTI node-creation callbacks rather than walking capture-graph edges (which misses stream nodes created prior to capture);
   - Selects `"key_by": "exec"` when `enable_event_node_ids=True` so NCCL event-record nodes under `NCCL_GRAPH_MIXING_SUPPORT` resolve by exec-graph ID, and dumps `rank{rank}_cuda_graph_py_stacks.json.gz` keyed by `(graph_id << 32) | graph_node_id`.
2. **TorchTitan Muse Spark `muse_spark_attention_qk` Unified Compiled Region (`pytorch/torchtitan` commit `8f8dbfb50caacdaa2e1af3ebb1d1ee67f57b9a87`, `torchtitan/models/muse_spark/model.py` & `torchtitan_recipes/models/muse_spark.py`)**:
   - Compiles per-head Q/K RMSNorms, query scale, position temperature (`clamp_min(log((pos+1)/floor_scale)*query_scale + 1.0, 1.0)`), and `int32`-pair real-arithmetic ComplexRoPE inside a single `muse_spark_attention_qk` region (`recompile_limit=16`, `batch_invariant=False`), removing the external Helion RoPE override and its 3 bounds-check kernels;
   - Achieves **`3.7500x`** GB300 decode speedup at `256 seqs` (`39.0 -> 10.4 us`) and **`2.6905x`** at `1 seq` (`22.6 -> 8.4 us`, `author-benchmark`) with bitwise-identical BF16 outputs and `8 <= 16` compiled graphs across training, prefill, and decode.
3. **TransformerEngine JAX Single-NVLink Fused EP Prepare and Dispatch (`NVIDIA/TransformerEngine` commit `b7db08fd73129f5af71ff2ee0eaddacd3ff1a751`, `transformer_engine/jax/cpp_extensions/ep.py` & `docs/envvars.rst`)**:
   - Introduces `ep_prepare_and_dispatch_fwd` (`EpPrepareAndDispatchHandler`) gated by `NVTE_EP_FUSED_PREPARE_DISPATCH=1` for single-NVLink-domain NCCL EP count mode, returning `(recv_tokens, recv_topk_weights, handle_mem)` with `handle_mem` aliased as a dispatch output and cutting XLA FFI launches per dispatch step by **`50.0%` (`2 -> 1`)**.

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/torchtitan_cuspy_graph_muse_rope_te_ep/run.py --check`:
- **Cuspy CUPTI Graph Annotations**: `24/24` graph nodes captured (vs `20/24` edge-walk), `4/4` NCCL event-record nodes resolved via `key_by="exec"`, bijective 64-bit Python stack keys verified.
- **Muse Spark `muse_spark_attention_qk`**: `3.75x` decode speedup (`256 seqs`), `0.0` numerical drift across position-temperature + RoPE sweeps, `8/16` graph budget verified.
- **TE JAX Fused NVL EP**: `50.0%` FFI launch reduction (`2 -> 1`), `0.0` combine reduction difference under intra-expert row permutation.
