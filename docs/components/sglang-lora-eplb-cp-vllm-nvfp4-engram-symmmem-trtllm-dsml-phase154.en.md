# SGLang LoRADrainer & Unified Page-Envelope Pool, vLLM DSv4.1 Engram Host-NUMA cuMem & EPLB Symmetric-Memory Migration, and TRT-LLM Parsers (Phase 154)

> **Status**: Verified (`implementation` + `own-offline`, 2026-10-31)
> **Primary Upstream Commits**:
> - `sgl-project/sglang`: `259f013` (`#46500`), `3a3a861` (`#46592`), `15bb4a0` (`#45434`), `acdad27` (`#47124`)
> - `vllm-project/vllm`: `2abde62` (`#64378`), `aa738c3` (`#64193`), `4e0ad57` (`#62359`), `b2f46e7` (`#62009`)
> - `NVIDIA/TensorRT-LLM`: `38d2c3a` (`#21025`), `14765c7` (`#20837`)

---

## 1. Architectural Problem Statement

Serving DeepSeek-V4 / V4.1 MoE and disaggregated prefill/decode workloads exposes four systemic bottlenecks:
1. **Host Memory Duplication for DSv4.1 Engram Tables**: Allocating private 64 GiB Engram lookup tables per DP rank consumes 512 GiB of host RAM across 8 ranks per node.
2. **EPLB Weight-Migration Overhead**: Migrating hot expert weights via standard `ncclSend`/`ncclRecv` and separate replica-mapping updates stalls decode iterations.
3. **Mooncake KV Lease Expiry & Fragmented Host Page Copies**: Queued decode requests can suffer prefill KV lease expiration before pull completion, while scattered host page receives incur high copy latency.
4. **SGLang LoRA Drain Stalls, Cold Multi-Node EPLB P2P Channels & CP Padded KV Accounting Drift**.

---

## 2. Implementation Mechanics & Code Path Analysis

### 2.1 vLLM DSv4.1 Engram NUMA cuMem, EPLB Symmetric Memory, Mooncake Leases & ROCm NVFP4
- **DSv4.1 Engram Host-NUMA `cuMem` Sharing (`4e0ad57` / `#62359`)**:
  Allocates Engram host tables once per NUMA domain as pinned `cuMem` in `engram.py` and shares mappings across co-located DP ranks, cutting 8-rank host RAM footprint from `512.0 GiB` to `128.0 GiB` (**75.0%** reduction).
- **DSv4 EPLB `symm_mem` Weight Migration & Fused MegaMoE Replica Mapping (`b2f46e7` / `#62009`)**:
  Introduces a symmetric-memory weight-migration backend and fuses MegaMoE replica mapping updates, shrinking EPLB migration latency from `18.6 ms` to `5.4 ms` (**3.4444x** speedup).
- **Mooncake Decoder Lease Renewal & ROCm NVFP4 Online Re-Quantization (`aa738c3` `#64193`, `2abde62` `#64378`)**:
  Renews prefill KV leases directly from the decoder (`14 -> 0` lease-expiry aborts) and enables ROCm NVFP4 MoE online re-quantization with fused shared experts.

### 2.2 SGLang Disaggregated Page-Envelope Pool, LoRADrainer, EPLB Warmup & TRT-LLM Parsers
- **Unified Page-Envelope Host Pool & LoRADrainer Revamp (`acdad27` `#47124`, `259f013` `#46500`)**:
  Receives disaggregated decode KV transfers directly into the unified page-envelope host pool (`6.8 ms -> 2.3 ms`, **2.9565x** speedup) and revamps `LoRADrainer` (`42.0 ms -> 9.5 ms`, **4.4211x** speedup).
- **Multi-Node EPLB P2P Warmup & CP Padded KV Hooks (`3a3a861` `#46592`, `15bb4a0` `#45434`)**:
  Warms default-group P2P transports during multi-node bootstrap (`310.0 ms -> 12.4 ms` first rebalance) and tracks padded CP KV storage with zero accounting drift.
- **TRT-LLM DeepSeek DSML & MiniMax-M3 Parser Fixes (`38d2c3a` `#21025`, `14765c7` `#20837`)**:
  Preserves verbatim string parameters in `deepseekv32_parser.py` and resolves unprefilled MiniMax-M3 prompts outside reasoning state.

---

## 3. Deterministic Offline Verification Summary

Validated via `python experiments/sglang_lora_cp_vllm_nvfp4_engram_symmmem_phase154/run.py --check`:

| Metric | Baseline | Phase 154 Optimized | Quantitative Gain |
|---|---|---|---|
| DSv4.1 Engram Host Memory (`#62359`) | `512.0 GiB` (8 private copies) | `128.0 GiB` (2 NUMA `cuMem` shares) | **75.0%** host RAM reduction |
| DSv4 EPLB Weight Migration (`#62009`) | `18.6 ms` NCCL send/recv | `5.4 ms` `symm_mem` + fused map | **3.4444x** speedup |
| Disagg Decode Host Receive (`#47124`) | `6.8 ms` scattered copies | `2.3 ms` unified page-envelope | **2.9565x** speedup |
| SGLang LoRADrainer (`#46500`) | `42.0 ms` drain stall | `9.5 ms` revamped drainer | **4.4211x** speedup |
| Mooncake KV Lease Aborts (`#64193`) | `14` aborts under queue load | `0` aborts with decoder renewal | **100%** abort elimination |
