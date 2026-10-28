# Deep Dive: SGLang Rust Server & Multimodal Pre-Processor / MHA Prefix Valid / KV Sharding / `RunnerIds`, vLLM `UMBPStoreConnector` / DeepSeek-V4.1 / `_DeferredRelease` / ZMQ Teardown & TensorRT-LLM `MultiFrontendServing` / `modeling_v2` (Phase 123)

> **中文版**: [sglang-rust-mm-mha-prefix-vllm-umbp-dsv41-tiering-p2p-trtllm-phase123.md](sglang-rust-mm-mha-prefix-vllm-umbp-dsv41-tiering-p2p-trtllm-phase123.md)  
> **Question ID**: `q-serving-sglang-rust-mm-mha-prefix-vllm-umbp-dsv41-tiering-p2p-trtllm-phase123` (`priority: 132`)  
> **Component**: `inference`  
> **Evidence Tier**: `implementation` (first-hand GitHub commit & diff inspection) + `own-offline` (zero-dependency Python 3.12 standard-library verification)  
> **Reproducible Experiment**: [`experiments/sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123/run.py`](../../experiments/sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123/run.py)

---

## 1. Executive Summary & Architectural Problem Statement

Across SGLang, vLLM, and TensorRT-LLM, Phase 123 resolves key production serving bottlenecks:
1. **SGLang Rust Serving, Multimodal Truncation & KV Prefix Efficiency**: Wiring per-request `skip_special_tokens` through the Rust OpenAI streaming detokenizer (`#47016`), enforcing `fit_input` truncation after multimodal token expansion in the Rust worker (`#46946`), propagating `_set_mha_kv_buffer_prefix_valid` to `MHAView`/`DraftMHAView` (`#47071`), sharding cached KV pages across attention-CP/TP ranks with NCCL `>= 2.31.2` zero-SM copy-engine prefix all-gather (`#46789`), and replacing `tl.constexpr` batch widths with runtime-strided `RunnerIds` (`#46990`).
2. **vLLM Tiered KV Memory Safety & P2P ZMQ Teardown**: Adding `UMBPStoreConnector` backed by MORI `EmbeddedRuntime` unified host DRAM pools (`#61793`), relaxing DeepSeek-V4.1 `specialized_decode` from `FLASHINFER_VERSION == "0.7.2"` to `>= 0.7.2` (`#63914`), introducing `_DeferredRelease` in `CPUPrimaryTierOffloadingManager` to prevent Use-After-Free (UAF) chunk reuse when secondary-tier promotion fails (`#63779`), and calling `dealer.disable_monitor()` before `monitor.close()` in `ZmqConnection.close()` (`#63207`).
3. **TensorRT-LLM `MultiFrontendServing` Control Plane & `modeling_v2` Custom Ops**: Forwarding `/metrics`, `/health`, and `/start_profile` from the parent launcher across all `SO_REUSEPORT` frontend workers over UDS (`#19394`) and splitting `modeling_v2` C++/CUDA extensions into per-family libraries (`glm4_moe_lite`, `gpt_oss`, `minimax_m2`, `#20896`).

---

## 2. First-Hand Code & Implementation Analysis

### 2.1 SGLang Rust Server (`#47016`, `#46946`), MHA Prefix Valid (`#47071`), KV Sharding (`#46789`) & `RunnerIds` (`#46990`)
- **Rust `skip_special_tokens` (`5d8e90a0`) & Multimodal `fit_input` (`9e36e5fc`)**: Passes per-request `skip_special_tokens` to `DetokenizerManager` across streaming chunks, and truncates expanded multimodal prompts (`3900 + 512 = 4412 -> 4096`, dropping 316 tail tokens) while updating `MmRequestInfo`.
- **`_set_mha_kv_buffer_prefix_valid` (`5cc9c933`)**: Inherits `KVPoolWithViews` in `MHAView` and `DraftMHAView` so cached/transferred prefix blocks are marked valid, cutting draft prefix recomputation from `2,048` tokens to **`0`**.
- **`--enable-kv-cache-sharding` (`9d2dae39`)**: Shards cached prefix pages across `TP=8` ranks (`512 -> 64` pages/rank, **8.0x** prefix VRAM reduction) and reconstructs prefixes via NCCL `>= 2.31.2` Copy Engine all-gather (**0 SM** occupancy).
- **`TreeSlot` (`e028ee40`) & `RunnerIds` Runtime Stride (`d5bc823a`)**: Passes `stride(0)` at runtime instead of a `tl.constexpr` batch width in `ReadTable`, reducing Triton JIT compilations across 32 dynamic batch widths from `32` to **`1`** (**32.0x** reduction).

### 2.2 vLLM `UMBPStoreConnector` (`#61793`), DeepSeek-V4.1 (`#63914`), `_DeferredRelease` (`#63779`) & ZMQ Close (`#63207`)
- **`UMBPStoreConnector` (`6cfb375f`) & DeepSeek-V4.1 (`ed6b8f4c`)**: Allocates 32,768 `2 MiB` chunks in a `64 GiB` MORI `EmbeddedRuntime` DRAM pool, and relaxes `FLASHINFER_VERSION == "0.7.2"` to `MIN_FLASHINFER_VERSION = Version("0.7.2")`.
- **`CPUPrimaryTierOffloadingManager` `_DeferredRelease` (`0beab6c2`)**: Holds evicted primary-tier chunks in `_deferred_release_chunks` when a secondary-tier write fails until `release_deferred_write()` is called after DMA completion/abort, reducing concurrent UAF overwrites from `1` to **`0`**.
- **P2P `ZmqConnection.close()` (`f4fd1ab6`)**: Invokes `dealer.disable_monitor()` before `monitor.close()` and `dealer.close()`, preventing libzmq background I/O thread deadlocks.

### 2.3 TensorRT-LLM `MultiFrontendServing` (`#19394`) & `modeling_v2` Custom Ops (`#20896`)
- **`MultiFrontendServing` Launcher UDS Plane (`0cd0a183`)**: Aggregates metrics, health, and profiling across all 4 `SO_REUSEPORT` workers (`25.0% -> 100.0%` worker coverage) and fails closed in `_watch_attached_frontends` if any frontend worker exits unexpectedly.
- **`modeling_v2` Per-Family Torch-Op Libraries (`7723a3b3`)**: Compiles independent shared libraries for `glm4_moe_lite`, `gpt_oss`, and `minimax_m2` (3 libraries, 6 registered C++/CUDA custom ops).

---

## 3. Deterministic Offline Verification (`own-offline`)

Run `python experiments\sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123\run.py --check`:

| Engine & Subsystem | Baseline | Phase 123 Result | Architectural Impact |
| :--- | :--- | :--- | :--- |
| SGLang `--enable-kv-cache-sharding` (`TP=8`) | `512` pages/rank (replicated) | `64` pages/rank + NCCL 2.31.2 CE | **8.0x** prefix VRAM reduction, **0 SM** occupancy |
| SGLang `DraftMHAView` & `RunnerIds` | `2,048` recomputed tokens; `32` compiles | **`0`** recomputed tokens; **`1`** compile | Eliminates draft recompute; **32.0x** fewer JIT compiles |
| vLLM `CPUPrimaryTierOffloadingManager` | Immediate free on failed write (`1` UAF) | `_DeferredRelease` until `release_deferred_write()` | **0** concurrent DMA UAF corruptions |
| TensorRT-LLM `MultiFrontendServing` (4 workers) | Random single-worker hit (**25.0%**) | Launcher UDS aggregation (**100.0%**) | Full metrics/profiling coverage + fail-closed supervision |
