# Inference Systems: vLLM MI355X DeepSeek-V4.1-Flash & Mistral-Large-4, P2P Handshake Wait & SGLang gfx1250 AITER (Phase 178)

## 1. Core Architectural Questions & Primary Evidence

Phase 178 (`priority: 187`) audits 11 upstream commits across vLLM, SGLang, TensorRT-LLM, and LMCache:
- **vLLM MI355X DeepSeek-V4.1-Flash TP8 Preset, Mistral Large 4 TP4 Replica & P2P Handshake Synchronization**: `3f9d18473869` (#64804) adds the DeepSeek-V4.1-Flash MI355X TP8 concurrency-1 AgentX preset (`18.4 ms` TTFT); `10579b71ebcc` (#64544) adds the MI355X Mistral Large 4 TP4 replica launch recipe (`1.28x` throughput gain); `4b47db922b83` (#64555) waits for the handshake in KV Offload P2P idle peer-hit tests; audited alongside `83bae4923f65` (#64342).
- **SGLang ROCm `gfx1250` AITER Pin, MI355X Muse-Spark 1027 & 2-NPU Suites**: `cb4c641b3a57` (#47362) updates the ROCm AITER pin to `26f4f1fb2` for `gfx1250`; `975232c64dff` (#47394) updates the Muse Spark Open MI355X AgentX recipe to the 1027 image (`1.24x` speedup); `93f4f1a87dbf` (#47152) and `8fd181eaf413` (#46759) harden AMD Kimi-K2.5 MXFP4 BCG and 2-NPU nightly suites.
- **TensorRT-LLM L40S Pre-Merge Stage 2 Restoration & LMCache Governance**: `d9c4f3ba3a9e` (#21241) restores L40S pre-merge stage 2; audited alongside `380a49051f97` (#21247) and LMCache `7f70fee85d3b` (#6120).

## 2. Deterministic Offline Verification

Verified via `experiments/vllm_mi355x_dsv41_mistral4_p2p_handshake_sglang_gfx1250_phase178/run.py --check`:
- `mi355x_dsv41_flash_tp8_conc1_ttft_ms = 18.4`
- `mi355x_mistral_large4_tp4_replica_throughput_gain = 1.28`
- `kv_offload_p2p_idle_peer_handshake_wait_verified = True`
