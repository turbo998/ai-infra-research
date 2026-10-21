# Experiment: `sglang-disagg-deferred-kv-tp-sync-radix-walk-trtllm-render-gemma4-vllm-offload`

Deterministic Python 3.12 standard-library reproduction of **Phase 48 (`priority: 57`) Topic 2**:
1. **SGLang Disaggregated Decode Deferred-KV TP Synchronization & Composable Radix Eviction Walk (`sgl-project/sglang@b57b5e2286`, `@5e79333c95`)**: Models `DecodeTransferQueue.pop_transferred` cross-TP failure synchronization (`#45453`) and `BaseEvictionResult.cache_actions` ordering invariant (`_free_values` drained before `_apply_cache_actions`, `#45439`).
2. **TensorRT-LLM Standalone CPU Render Endpoint, Gemma-4 Non-SM100 Hybrid KV Spec & Gen-Only Perf Baseline Floor (`NVIDIA/TensorRT-LLM@ec3352fee2`, `@e8127c7c5c`, `@7f0caf443e`)**: Models `trtllm-render` (`render_fingerprint.digest` + Kimi-K3 3-token opener exclusion, `#19964`), `KVCacheLayerSpec` + `max_num_heads_per_kv=8` for Gemma-4 (`#18547`), and `GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS = 1790856543000` (`#20297`).
3. **vLLM Placeholder Offload Group Classification & Multimodal Chunked-Embedding Guard (`vllm-project/vllm@550c1cbb8b`, `@65a886c69f`)**: Models static `_group_specs` survival across `OffloadingEventsTracker.reset()` (`#59703`) and `VLLMValidationError` rejection on multimodal inputs in `maybe_pre_process_chunked` (`#62859`).

## Run
```bash
python experiments/sglang-disagg-deferred-kv-tp-sync-radix-walk-trtllm-render-gemma4-vllm-offload/run.py --check
```
