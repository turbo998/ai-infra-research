# SGLang DSV4.1 Engram All-to-All Decode, ROCm Separate K/V & Sparse-MLA DCP, HiCache Page-Unified E2E, vLLM MRV2 DBO + SpecDecode/DCP & TensorRT-LLM Anthropic Keepalive (Phase 130)

Deterministic CPU/offline analytical simulator covering 10 first-hand upstream commits in `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. `sgl-project/sglang@1825a09` (`#46955`): Uses `all_to_all` lookup for DeepSeek-V4.1 Engram tables during decode and CUDA graph capture/replay (`~16.0x` lower per-rank collective traffic vs `all_reduce` at `TP=8`).
2. `sgl-project/sglang@59af99a` (`#47008`): `sglang-api-types` decodes flat token-ID/string lists directly instead of buffering intermediate `Value` enums (`8x` lower per-token temporary heap footprint).
3. `sgl-project/sglang@69233a4` (`#46181`): Rust server gathers all DP listener endpoints (`rust_server_endpoints`) and fans control requests out to every DP rank (`100%` DP rank coverage).
4. `sgl-project/sglang@8044ed4` (`#46972`): Rejects `--enable-int8-mamba-checkpoint` when `--enable-unified-memory` is enabled because the unified pool has no int8 Mamba checkpoint pool.
5. `sgl-project/sglang@86f72e5` (`#46861`): Keeps `K` and `V` in separate memory pool allocations on ROCm (`not _is_hip`) so same-node Mooncake PD disaggregation on ROCm does not decode corrupted V pointers.
6. `sgl-project/sglang@edd270f` (`#44322`): Enables Decode Context Parallelism (DCP) on the AMD Triton sparse-MLA backend with `_dcp_pack_q_kernel` (`[q_pe | q_nope]`) and runtime-stride sparse-MLA kernels.
7. `sgl-project/sglang@f65af1c` (`#39728`): Completes HiCache `page_unified` end-to-end integration with dedicated `can_use_page_unified_write_back_jit_kernel` compilation probing.
8. `sgl-project/sglang@aa8e0a9` (`#41696`): Slices `kv_indptr[: bs + 1]` in `_prepare_aiter_dsa_decode_metadata` to the active batch size (`15.12x` reduction at `bs=16` vs `max_bs=256`).
9. `vllm-project/vllm@cdd2750` (`#61464`): Supports Dual-Batch Overlap (DBO) with speculative decoding and DCP in Model Runner V2, clamping `dbo_decode_token_threshold` (`7 -> 8` for 3 drafts = 4 tokens/request across 2 microbatches).
10. `NVIDIA/TensorRT-LLM@461580b` (`#21030`): Emits `AnthropicPingEvent` (`type: "ping"`) keepalive frames while an Anthropic Messages SSE stream is silent during long prefill or reasoning steps.

## Usage

```bash
python experiments/sglang_engram_decode_dcp_mla_hicache_vllm_dbo_trtllm_phase130/run.py --check
```
