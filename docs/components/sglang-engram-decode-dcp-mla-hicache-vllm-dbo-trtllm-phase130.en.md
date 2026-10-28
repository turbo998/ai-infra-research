# SGLang DSV4.1 Engram All-to-All Decode, ROCm Separate K/V & Sparse-MLA DCP, HiCache Page-Unified E2E, vLLM MRV2 DBO + SpecDecode/DCP & TensorRT-LLM Anthropic Keepalive (Phase 130)

> **Chinese Edition**: [sglang-engram-decode-dcp-mla-hicache-vllm-dbo-trtllm-phase130.md](sglang-engram-decode-dcp-mla-hicache-vllm-dbo-trtllm-phase130.md)  
> **Phase 130 (`priority: 139`) · 130-Phase & 400-Experiment-Suite Milestone · Track 2 (LLM Serving & Disaggregated Runtime)**  
> **Evidence Class**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic CPU simulation; not a GPU hardware rerun)

---

## 1. Architectural Context & Problem Statement

Across SGLang, vLLM Model Runner V2 (`MRV2`), and TensorRT-LLM serving stacks, 10 upstream commits landed on 2026-10-28 resolve five production bottlenecks:

1. **DeepSeek-V4.1 Engram Lookup Collective Amplification in Decode & CUDA Graphs**: When an Engram table is sharded across a TP group (`replication: none`), running `all_reduce` during decode scales per-rank traffic with global batch size. Switching decode and Piecewise CUDA Graph paths to `all_to_all`—and deferring `lookup: auto` resolution until model construction once parallel topology and GPU capabilities are known—cuts per-rank collective volume by **16.0x** at `TP=8`.
2. **Rust Control-Plane Fanout & Flat Token-ID Serde Overhead**: `sglang-api-types` previously buffered flat token-ID/string lists as intermediate `Value` structs before decoding; meanwhile, multi-DP deployments required a collective (`RustServer.gather_server_endpoints`) so every Rust HTTP listener learns all peer DP listener endpoints for control-request fanout.
3. **ROCm Fused K/V Pool Corruption under Mooncake PD & Sparse-MLA DCP Enablement**: While CUDA's `trtllm_mha` `fmha_v2` prefill reaches `V` from `k_cache` via int32 block offsets, same-node Mooncake PD disaggregation on ROCm decodes garbage when `V` sits inside `K`'s allocation. Additionally, AMD's Triton sparse-MLA backend lacked Decode Context Parallelism (DCP) support, and AITER DSA decode passed unsliced `kv_indptr` buffers.
4. **HiCache `page_unified` Write-Back JIT Probing & Unified Memory + INT8 Mamba Validation**: Probing `page_first` JIT availability cannot validate `page_unified` write-back because each layout instantiates a distinct template under its own `cache_once` key; separately, `--enable-unified-memory` must reject `--enable-int8-mamba-checkpoint` because the unified pool allocates no int8 Mamba checkpoint buffer.
5. **vLLM MRV2 DBO Microbatch Thresholds under Speculative Decoding & TensorRT-LLM Silent-Stream Timeouts**: Under Dual-Batch Overlap (DBO) with $K_{\text{spec}}$ speculative draft tokens, each request carries $1 + K_{\text{spec}}$ tokens, so 2 microbatches require at least $2(1 + K_{\text{spec}})$ tokens; meanwhile, TensorRT-LLM's Anthropic Messages SSE endpoint needed periodic `ping` events while a stream is silent during long prefill or reasoning phases.

---

## 2. First-Hand Upstream Evidence Matrix (10 Commits)

| # | Repository & Commit | Date | Mechanism & Quantitative Architectural Impact |
|---|---|---|---|
| 1 | `sgl-project/sglang@1825a09` (`#46955`) | 2026-10-28 | Enables `all_to_all` Engram lookup for DeepSeek-V4.1 (`dsv41`) decode and CUDA graphs, resolving `lookup: auto` at model construction (`resolve_engram_config`) (**16.0x** lower per-rank collective bytes vs `all_reduce` at `TP=8`). |
| 2 | `sgl-project/sglang@59af99a` (`#47008`) | 2026-10-28 | `sglang-api-codegen` / `sglang-api-types` decodes flat token-ID/string lists directly instead of buffering intermediate `Value` enums (**8.0x** lower per-token temporary heap footprint). |
| 3 | `sgl-project/sglang@69233a4` (`#46181`) | 2026-10-28 | Gathers `rust_server_endpoints` across all ranks in `maybe_init_rust_server` and fans control requests out to every DP rank (**100%** DP control coverage). |
| 4 | `sgl-project/sglang@8044ed4` (`#46972`) | 2026-10-28 | `handle_int8_mamba_checkpoint` raises a `ValueError` when `--enable-int8-mamba-checkpoint` is combined with `--enable-unified-memory`. |
| 5 | `sgl-project/sglang@86f72e5` (`#46861`) | 2026-10-28 | Restricts fused K/V pool allocation to `k_shape == v_shape and not _is_hip`, keeping `K` and `V` in separate allocations on ROCm to prevent same-node Mooncake PD corruption. |
| 6 | `sgl-project/sglang@edd270f` (`#44322`) | 2026-10-28 | Enables DCP on the AMD Triton sparse-MLA backend via `_dcp_pack_q_kernel` (`[q_pe \| q_nope]`) and runtime-stride `_sparse_mla_fwd_split_dim_kernel`. |
| 7 | `sgl-project/sglang@f65af1c` (`#39728`) | 2026-10-28 | Completes HiCache `page_unified` end-to-end integration with dedicated `can_use_page_unified_write_back_jit_kernel` compilation probing. |
| 8 | `sgl-project/sglang@aa8e0a9` (`#41696`) | 2026-10-28 | Slices `kv_indptr = kv_indptr[: bs + 1]` in `_prepare_aiter_dsa_decode_metadata` (**15.12x** metadata slice reduction at `bs=16` vs `max_bs=256`). |
| 9 | `vllm-project/vllm@cdd2750` (`#61464`) | 2026-10-28 | Supports DBO with speculative decoding and DCP in vLLM `MRV2`, clamping `dbo_decode_token_threshold` to cover a full request per microbatch (`(7, 8, 16) -> (8, 8, 16)` for 3 drafts) and rebuilding prefill metadata from per-microbatch target slots. |
| 10 | `NVIDIA/TensorRT-LLM@461580b` (`#21030`) | 2026-10-28 | Adds `AnthropicPingEvent(type="ping")` keepalive emissions while an Anthropic Messages stream is silent, preventing proxy idle timeouts during long prefills. |

---

## 3. Analytical Formulations & Deterministic Verification

All 5 invariants in `experiments/sglang_engram_decode_dcp_mla_hicache_vllm_dbo_trtllm_phase130/run.py --check` pass (`engram_all_to_all_reduces_decode_comm_by_16x`, `flat_id_list_serde_saves_8x_heap`, `rocm_separates_kv_and_slices_aiter_indptr`, `vllm_dbo_threshold_7_clamped_to_8_for_3_drafts`, `trtllm_anthropic_keepalive_prevents_30s_proxy_timeout`).

---

## 4. Architect Guidance & Falsifiable Follow-Up

- **vLLM MRV2 DBO + Speculative Decoding Sizing**: Always size `dbo_decode_token_threshold` and `dbo_prefill_token_threshold` to at least $2 \times (1 + K_{\text{spec}})$ so dual microbatches never starve one microbatch of a complete draft-verification bundle.
- **Falsifiable Follow-Up Question**: Under `MRV2` with `DCP + DBO + ngram speculative decoding`, at what KV sequence length does DCP's cross-rank KV all-to-all communication exceed the compute time of the alternate microbatch?
