# SGLang DSV4.1 Physical KV Capacity, vLLM NVFP4 MLA Gather & GLM-5.3 KDA Fusion, and TRT-LLM Helix/GDN Analysis (Phase 109 · Priority 118)

> **中文版**: [sglang-dsv41-kv-cap-vllm-nvfp4-mla-kda-trtllm-helix-gdn-phase109.md](sglang-dsv41-kv-cap-vllm-nvfp4-mla-kda-trtllm-helix-gdn-phase109.md)  
> **Evidence Boundary**: `implementation` (first-hand source commits) + `own-offline` (deterministic stdlib modeling: `experiments/sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109/run.py --check`)

## 1. First-Hand Source Evidence & Implementation Paths

1. **`sgl-project/sglang@dea05ca16e` (`#46718`), `7f8ad67373` (`#46144`) & `20ff40cb84` (`#41275`)**:
   - **DeepSeek-V4 Capture Stub, Per-Role Disaggregated Topology & DSV4.1 Physical KV Capacity**: `#46718` adds `_write_tail_source_kv_capture_stub` in `deepseek_v4.py` so CUDA Graph capture records `tail_source_kv_layer_id` without graph breaks; `#46144` recomputes disaggregated diffusion topology (`tp_size`, `sp_degree`) per role (`Encoder = 2 GPUs -> TP 2`, `Denoiser = 4 GPUs -> TP 4`); `#41275` updates `_compressed_bytes_per_full_token` in `pool_configurator.py` to include `c128` fixed state bytes at `ratio = 2` (`352.0 B -> 480.0 B/token`), eliminating a **`36.36%` token capacity over-estimation** on a `16 GiB` pool.
2. **`vllm-project/vllm@741584f8ca` (`#58744`), `3bc521d9e6` (`#61543`) & `90309f6b74` (`#60490`)**:
   - **NVFP4 MLA Chunked-Context Gather, Bounded KV Group Skipping & Qwen4Exp QSA Workspace Reservation**: `#58744` extends `cp_gather_and_upconvert_nvfp4_kv_cache` with `seq_starts` to gather `nvfp4_ds_mla` chunked-context KV directly (`612 B/token` vs `1,152 B/token` BF16 staging, **`1.8824x` gather bandwidth reduction**, `18.0 MiB -> 9.5625 MiB` at `16,384` tokens); `#61543` skips non-transfer bounded KV groups (`CircularBufferSpec`, `HiSparseHotSpec`) in `DecodeBenchConnector`; `#60490` reserves **`128.0 MiB`** for the Qwen4Exp QSA indexer prefill workspace during memory profiling.
3. **`vllm-project/vllm@de096755ad` (`#60991`) & `115b59a2da` (`#63857`)**:
   - **GLM-5.3-Flash KDA Fused `conv1d` Parameter & Heterogeneous Per-Layer TP Auto-Detection**: `#60991` fuses GLM-5.3 KDA `q|k|v` short convolutions into a single `conv1d.weight` parameter (`3 -> 1` parameter), ensuring RL weight refits update all channels in-place without invalidating FULL CUDA Graph captured pointers; `#63857` inspects all per-layer attention configurations (`[32, 48, 64]` KV heads) in `tools/recipes/model_constraints.py` and computes `gcd(32, 48, 64) = 16` as the maximum valid TP degree instead of failing on `TP = 32` inferred from layer 0.
4. **`NVIDIA/TensorRT-LLM@4b027a381d` (`#20759`) & `ae91047ac3` (`#20008`)**:
   - **Aggregated Helix KV Cache Sizing & FlashInfer Ring-Buffer Replay for GDN MTP**: `#20759` defines `_HELIX_AGG_FREE_GPU_MEMORY_FRACTION = 0.6` for unprofiled Aggregated Helix runs (allocating `36.0 GiB` KV and preserving **`24.0 GiB` (`40%`) prefill activation headroom** on `60.0 GiB` free VRAM); `#20008` integrates FlashInfer ring-buffer cached replay for GDN Multi-Token Prediction (`MambaHybridCacheManagerV2`), cutting speculative state replay latency from `115.0 us` to **`46.0 us` (`2.50x` speedup)**.

## 2. Architect Design Exercises & Falsifiable Questions

- **Calculation Exercise**: Given heterogeneous KV head counts $H_{\text{kv}} \in \{32, 48, 64\}$ across layers, verify why maximum non-replicated TP is $\gcd(32, 48, 64) = 16$ whereas layer-0-only detection selects $32$ and fails on 48-head layers.
- **Falsifiable Hypothesis**: Pricing `ratio = 2` compressed KV tokens without `c128` fixed state bytes over-estimates token capacity by $36.36\%$ (`48,806,446` vs `35,791,394` tokens on `16 GiB`), falsified and fixed by `sglang #41275`.
