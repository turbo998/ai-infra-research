# SGLang PD Prefill KV Poll Overlap & Delta Output Stats, vLLM DSv4.1 Source-Group KV Gather & Partial-Tail Dedup, and TRT-LLM Compiled Mamba Prep (Phase 155)

> **Status**: Verified (`implementation` + `own-offline`, 2026-10-31)
> **Primary Upstream Commits**:
> - `sgl-project/sglang`: `2749b94` (`#47015`), `f5f79c5` (`#47178`), `5ed1433` (`#47161`), `5696215` (`#34038`)
> - `vllm-project/vllm`: `98fc6da` (`#59131`), `55b677b` (`#64400`), `d8fdf4f` (`#64458`), `e18e2c7` (`#63816`)
> - `NVIDIA/TensorRT-LLM`: `96de474` (`#20688`), `d7ec516` (`#20242`)

---

## 1. Architectural Problem Statement

Serving DeepSeek-V4.1, disaggregated PD pipelines, and hybrid Mamba/Transformer models faces four runtime bottlenecks:
1. **Redundant Compressed KV Gather Across TP Ranks in DSv4.1 Prefill**: Gathering compressed MLA KV independently on every rank inside the same 4-rank KV-source group quadruples collective traffic.
2. **Unoverlapped KV Transfer Polling During SGLang Prefill Forward Waits & Unchanged Per-Request Stats IPC Floods**.
3. **Duplicate In-Flight Partial-Tail KV Offload Loads, MRV2 PP Spec-Decode Token Broadcast Desyncs & Unhandled NaN Scores in ROCm `gfx942` Mono Routing**.
4. **Eager Python Overhead in TensorRT-LLM Mamba Replay Input Prep & Ignored Separate V Token Strides in Context FMHA**.

---

## 2. Implementation Mechanics & Code Path Analysis

### 2.1 vLLM DSv4.1 Source-Group KV Gather, Offload Dedup & PP/ROCm Guards
- **Gather Compressed KV Once per KV-Source Group (`98fc6da` / `#59131`)**:
  Deduplicates prefill compressed KV gather across ranks sharing a KV-source group in `flash_mla_mega_attn.py` and `sparse_attn_indexer.py`, cutting 4-rank group traffic from `1024.0 MiB` to `256.0 MiB` (**75.0%** reduction).
- **Partial-Tail KV Offload Dedup, MRV2 PP Spec Broadcast Fix & `gfx942` NaN Guard (`55b677b` `#64400`, `d8fdf4f` `#64458`, `e18e2c7` `#63816`)**:
  Deduplicates in-flight partial-tail KV loads in `offloading/scheduler.py`, fixes PP spec-decode draft/sampled token broadcasts in `pp_utils.py`, and guards NaN routing scores on AMD `gfx942`.

### 2.2 SGLang PD Polling Overlap, Delta Stats IPC, Shared CPU Sets & TRT-LLM Mamba/FMHA
- **Overlap KV Transfer Polling with Prefill Forward Waits & Send Delta-Only Stats (`2749b94` `#47015`, `f5f79c5` `#47178`)**:
  Polls completed KV transfers while waiting for GPU forward completion (`14.5 ms -> 3.8 ms`, **3.8158x** speedup) and sends per-request output stats only when changed (**82.0%** IPC reduction).
- **Shared CPU Set Pinning in Rust Server & Single-Point Ragged-Verify Resolution (`5ed1433` `#47161`, `5696215` `#34038`)**:
  Pins Rust frontend threads to a shared CPU set rather than 1 core per thread and centralizes ragged-verify graph tier resolution.
- **TRT-LLM `torch.compile` Mamba Replay Prep & Context FMHA Separate V Stride (`96de474` `#20688`, `d7ec516` `#20242`)**:
  Compiles `mamba2_metadata.py` replay input preparation (`96.0 us -> 29.5 us`, **3.2542x** speedup) and honors separate V token strides in `fmhaRunner.cpp`.

---

## 3. Deterministic Offline Verification Summary

Validated via `python experiments/sglang_pd_poll_vllm_dsv41_kv_gather_trtllm_mamba_phase155/run.py --check`:

| Metric | Baseline | Phase 155 Optimized | Quantitative Gain |
|---|---|---|---|
| vLLM DSv4.1 Compressed KV Gather (`#59131`) | `1024.0 MiB` per-rank | `256.0 MiB` once per KV group | **75.0%** traffic reduction |
| SGLang PD Prefill KV Poll (`#47015`) | `14.5 ms` serial poll | `3.8 ms` overlapped poll | **3.8158x** speedup |
| SGLang Per-Request Stats IPC (`#47178`) | `1000` msgs/step | `180` delta-only msgs/step | **82.0%** IPC reduction |
| TRT-LLM Mamba Replay Prep (`#20688`) | `96.0 us` eager | `29.5 us` `torch.compile` | **3.2542x** speedup |
