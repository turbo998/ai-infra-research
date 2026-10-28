# SGLang Rust Server / MM Pre-Processor / MHA Prefix Valid / KV Sharding / `TreeSlot` / `RunnerIds`, vLLM `UMBPStoreConnector` / DeepSeek-V4.1 / `_DeferredRelease` / ZMQ Close & TensorRT-LLM `MultiFrontendServing` / `modeling_v2` (Phase 123)

Deterministic Python 3.12 standard-library model covering:
- SGLang Rust OpenAI server `skip_special_tokens` per-request detokenization (`#47016`), Rust multimodal worker `fit_input` truncation (`#46946`), `_set_mha_kv_buffer_prefix_valid` for `MHAView`/`DraftMHAView` (`#47071`), `--enable-kv-cache-sharding` page-based KV cache sharing across attention-CP/TP prefill ranks with NCCL `>= 2.31.2` zero-SM copy-engine prefix all-gather (`#46789`), `TreeSlot` prefix-tree storage slots (`#46991`), and `RunnerIds` runtime-strided `ReadTable` views (`32x` fewer Triton JIT compilations, `#46990`).
- vLLM `UMBPStoreConnector` with MORI `EmbeddedRuntime` unified memory buffer pool (`64 GiB` DRAM / `32,768` 2 MiB chunks, `#61793`), DeepSeek-V4.1 `specialized_decode` relaxing `FLASHINFER_VERSION == "0.7.2"` pin to `>= 0.7.2` (`#63914`), `CPUPrimaryTierOffloadingManager` `_DeferredRelease` preventing premature primary-tier chunk reuse during failed secondary-tier promotions (`#63779`), and P2P `ZmqConnection.close()` `dealer.disable_monitor()` ordering (`#63207`).
- TensorRT-LLM `MultiFrontendServing` launcher-owned UDS metrics/profiling/liveness forwarding across `SO_REUSEPORT` workers (`25% -> 100%` coverage, `#19394`) and `modeling_v2` per-family C++/CUDA torch-op libraries (`glm4_moe_lite`, `gpt_oss`, `minimax_m2`, `#20896`).

## Run & Verify

```powershell
python experiments\sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123\run.py --check
```
