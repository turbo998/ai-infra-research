# SGLang MNNVL NVML Fabric Detection, SM107 KDA Spec-Verify BV=4, DCP Block-FP8 Q-Proj Replication, vLLM FP8 Indexer WK Persistence & TensorRT-LLM Gemma4 / INT8-WOQ MoE Padding

> Chinese version: [sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq.md](sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq.md)
> Evidence grade: `implementation` + `author-benchmark` (upstream source/PR benchmarks) + `own-offline` (deterministic Python 3.12 stdlib simulator, not direct GPU hardware re-execution)

## 1. Core Architectural Questions & Code Evolution

Phase 84 (`priority: 93`) audits 9 upstream commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. **SGLang MNNVL NVML Fabric Detection & SM107 KDA Spec-Verify `BV=4` Tile Narrowing ([#46463](https://github.com/sgl-project/sglang/commit/39b54ae0eeca90f7097b22fe016e9212523a5968), [#46465](https://github.com/sgl-project/sglang/commit/640e11ebc0dfc660ea766acf43b0586d1819f494))**:
   - Extends `is_mnnvl_fabric_device()` beyond `"GB200"`/`"GB300"` string matching to check NVML fabric registration state, enabling MNNVL paths on generic-named `SM107` multi-node NVLink parts.
   - Narrows the KDA speculative-verification recurrence tile on `SM107` (212 SMs) from `BV=32` (48 CTAs at Kimi-K3 TP8) to `BV=4` (**384 CTAs**, 8x parallelism), cutting isolated latency from `13.66 us` to `6.99 us` (**1.954x** speedup).
2. **SGLang DCP `128x128` Block-FP8 `q_b_proj` Replication & DeepSeek-V4 MoE Finalize Guard ([#46359](https://github.com/sgl-project/sglang/commit/f264183bfe1d6f7e8f9046d5234f22e0496ecac7), [#46365](https://github.com/sgl-project/sglang/commit/ad950cdcd9666a104321deeaf1cbf883b4e8c60b))**:
   - Extends `--dcp-replicate-q-proj` to support `128x128` block-FP8 query projections, saving one per-layer query all-gather across all 61 MLA layers.
   - Guards DeepSeek-V4 `prepare_output()` so `summed=True` is set when `self.rows.group is None` only if `post_experts_output_is_complete(is_tp_path=False)` holds.
3. **vLLM FP8 Indexer WK Persistence, cuBLAS Workspace Release & NIXL OBJ Telemetry ([#59223](https://github.com/vllm-project/vllm/commit/2c880f1e68200f79c4663dd0f1e7beab6f536cfa), [#63451](https://github.com/vllm-project/vllm/commit/b3466364b69db84ddeca30435f56c822551f8b14), [#55576](https://github.com/vllm-project/vllm/commit/4a81a5097631ad9851a7045ebb4f2f8c4873eec0))**:
   - Preserves pending DeepSeek-V3.2 FP8 `indexer.wk` / `weight_scale_inv` pairs across split `load_weights()` calls triggered by `lm_head.weight` prefix transitions in `AutoWeightsLoader`.
   - Releases cached cuBLAS workspaces after `WeightCacheDaemon` warmup and enables `capture_telemetry=True` on the NIXL OBJ offload tier.
4. **TensorRT-LLM Gemma4 Encoder Budget Scheduling & INT8-WOQ MoE 64-Row Padding ([#20663](https://github.com/NVIDIA/TensorRT-LLM/commit/abb94ab8fc1c93ee377a5137c9b36466e2ae3664), [#16198](https://github.com/NVIDIA/TensorRT-LLM/commit/64ef54ce1249f89ea23b03d4868425c3d5aec268))**:
   - Schedules Gemma4 multimodal encoder items under the encoder budget while keeping encoder-free `Gemma4Unified` projections unscheduled; pads `INT8-woq` MoE per-partition intermediate sizes (`1856 // TP`) up to multiples of `64` (`rows_per_tile = 128 * 8 // 16 = 64`) for `TP > 1`.

## 2. Reproducible Experiment

- Command: `python experiments/sglang_mnnvl_kda_sm107_dcp_qproj_vllm_wk_cublas_nixl_trtllm_gemma4_int8_woq/run.py`
- Evidence ID: `ev-sglang-mnnvl-kda-sm107-dcp-qproj-vllm-wk-cublas-nixl-trtllm-gemma4-int8-woq-2026`
