# FlashInfer CAKE `tinygemm2` `PACKED_M64_N8` / KCP Pool Sharing / Rubin AOT / cuDNN 4-Slot Prefill Cache / Blackwell `bf16_fp4` / Mamba SSU / MiniMax-H3 SM120 / DSV4, FlashMLA SM100 WAR Barrier & TileLang Guards (Phase 123)

Deterministic Python 3.12 standard-library model covering:
- FlashInfer CAKE `tinygemm2` `tcgen05` `PACKED_M64_N8` (`(16, 8)` tile, `101,376 B` dynamic SMEM, Rubin `sm_107a` 212 SMs, `8x4096x4096` `6.24 -> 5.98 us`, `1.0435x`) & `cake_kda_kcp` pool sharing (`SHARED_POOLS`, `LAZY_POOLS`, saving 4 device buffers on `all_to_all` and 2 on `ring`).
- FlashInfer Rubin (`SM107`) AOT registration of `gen_cutlass_fused_moe_sm100_module` & `gen_gemm_sm100_module_cutlass_fp8` (eliminating `789 s` runtime JIT vs `1200 s` GSM8K deadline) & `_reuse_cudnn_prefill_preparation` (`_MAX_CUDNN_PREPARED_CACHE_ENTRIES = 4`, cutting B200 first-layer switch time `121.69 -> 74.89 us` on B1 and `117.31 -> 69.64 us` on B3).
- FlashInfer CAKE Blackwell `bf16_fp4` `M=128` 2-CTA pair-tile pre-cluster-barrier load (`1.0126x` GB300, `1.0109x` B200), Mamba2 `selective_state_update` `sm_100a` batch-64 `64h/8g` z-gated dispatch (`40.440 -> 38.368 us`, `1.0540x`), MiniMax-H3 SM120 NVFP4 warp-specialized `ATTENTION_WS_NODELTA_LEAD/TAIL` (`384` threads, `80,896 B` SMEM, `13 -> 15` programs, `1.041x`), and `cake_dsv4` (`89/197` faster on GB300, `0` slower).
- DeepSeek FlashMLA SM100 sparse decode `head64` `NamedBarrier::arrive_and_wait(128, wg0_sync)` eliminating cross-lane (`idx ^ 64`) WAR race, and TileLang `scalar_read_coeff` separation from 16-bit `DataType::lanes()`, `CUtensorMap` `desc_name_map` verification, `AtomicAddx4` `float*` constraint, and `gemm_sp` `K % micro_size_k == 0` guard.

## Run & Verify

```powershell
python experiments\flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123\run.py --check
```
