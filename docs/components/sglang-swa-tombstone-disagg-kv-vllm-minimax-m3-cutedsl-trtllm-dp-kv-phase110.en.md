# SGLang SWA Tombstone Prefix Clamp & Non-Blocking Disagg KV Restore, vLLM Hopper MiniMax-M3 Sparse Prefill & CuteDSL BF16x3 Router GEMM, TensorRT-LLM Attention-DP KV Connector & Logprobs Memory Reduction (Phase 110)

> 中文版: [sglang-swa-tombstone-disagg-kv-vllm-minimax-m3-cutedsl-trtllm-dp-kv-phase110.md](sglang-swa-tombstone-disagg-kv-vllm-minimax-m3-cutedsl-trtllm-dp-kv-phase110.md) | Companion experiment: `experiments/sglang_swa_tombstone_disagg_kv_vllm_minimax_trtllm_phase110/run.py`

## 1. First-Hand Code Evidence (2026-10-27)

| Repo | Commit | Date | Core Mechanism Change | Evidence Class |
|---|---|---|---|---|
| `sglang` | `e7c55f6959` | 2026-10-27 | Clamps `UnifiedRadixCache` SWA tombstone recovery to `window_start = max(evicted_seqlen, prev_prefix_len)` unconditionally so non-independent components never free a request's protected prefix | `implementation` |
| `sglang` | `b28f4fc065` | 2026-10-27 | Restores host-received disaggregated decode KV asynchronously via `start_restore_kv_cache`, dedicated `index_stream`, and `restore_done_event` without blocking the scheduler | `implementation` |
| `sglang` | `b7ac4c0e72` | 2026-10-27 | Uses `pyo3::sync::MutexExt::lock_py_attached` in Rust `sglang-radix-tree` to keep the Python GIL during uncontended `O(1)` flag, size, and digest reads | `implementation` |
| `vllm` | `8e26dec5f5` | 2026-10-27 | Adds Hopper (`SM90`) Triton query-tiled sparse prefill (`_PREFILL_TILE_Q = 16`, `SPARSE_BLOCK_SIZE = 128`) for MiniMax-M3 so 16 queries share each selected KV page load | `implementation` |
| `vllm` | `0076619a94` | 2026-10-27 | Adds Hopper (`SM90`) CuteDSL `BF16x3` router GEMM decomposing FP32 router weights into 3 BF16 register terms with `wgmma` and rank-ordered DSMEM split-K reduction | `implementation` |
| `vllm` | `5c1eb11cb3` | 2026-10-27 | Enables pipeline parallelism (`PP > 1`) for Qwen4Exp PLE models and their MTP drafter (`PPMissingLayer` on non-first ranks, input-driven MTP branching, and per-stage KV group projection) | `implementation` |
| `vllm` | `8977b13a1c` | 2026-10-27 | Sets DeepSeek-V4.1 sparse MLA and Compressor metadata builders to `AttentionCGSupport.UNIFORM_BATCH` (`max_decode_query_len`) to keep non-graph-safe sparse prefill out of FULL CUDA Graphs | `implementation` |
| `vllm` | `cebcc5d8c7` | 2026-10-27 | Allocates `MRV2` `self.model_state` outside the weights memory pool so `sleep_mode` weight offloading preserves runtime device state tensors | `implementation` |
| `TensorRT-LLM` | `0a03c3ed05` | 2026-10-27 | Supports Attention DP in V2 KV cache connectors (`attention_dp_rank` per owner scheduler, non-consuming `pending_cancellation_ids()`, and `_failed_kv_connector_executors` DMA lifetime retention) | `implementation` |
| `TensorRT-LLM` | `8354c4c398` | 2026-10-27 | Allocates full-vocabulary `logprobs_cuda` only for `need_processed_logprobs` rows while computing raw logprobs in-place from logits via Triton `selected_raw_logprobs` (saving `74.19 MiB` at `B=128, V=151936`) | `implementation` |

## 2. Key Implementation & Architectural Takeaways

1. **SWA Protected-Prefix Safety & Non-Blocking Disagg KV Restore**: Unclamped SWA tombstone recovery when `evicted_seqlen = 256 < prev_prefix_len = 512` corrupts `128` protected prefix tokens (`[384, 512)`); clamping `window_start = max(evicted_seqlen, prev_prefix_len)` reduces corrupted tokens to `0`, while event-driven `start_restore_kv_cache` eliminates `1440 us` of scheduler stall per 8-request decode batch.
2. **Query-Tiled Hopper Sparse Prefill & Sampler Memory Scaling**: Tiling `16` queries per `128-token` sparse KV page cuts MiniMax-M3 KV page HBM traffic by `16x`, and direct Triton raw-logprob selection eliminates `74.19 MiB` of temporary FP32 vocabulary buffers per `128`-request step.
