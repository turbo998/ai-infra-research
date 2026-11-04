# Inference Serving Systems: TensorRT-LLM Kimi-K3 CuTe DSL PDL Decode Residual Mix & GPU Vision Pixel Normalization, vLLM Hybrid Spec Decode SW Divisor Block Sizing, SGLang AITER Zero-Copy Strided-Q Extend & R3 Non-Output Rank Skip, and Dynamo PoolRelay (Phase 182)

> 🌐 **Language / 语言切换**: **[中文](inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.md)** | **[English](inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 182 (`priority: 191`) audits **11 upstream mainline commits** across `NVIDIA/TensorRT-LLM`, `vllm-project/vllm`, `sgl-project/sglang`, and `ai-dynamo/dynamo`:

1. **TensorRT-LLM Kimi-K3 CuTe DSL PDL Decode Attention-Residual Mix & GPU `uint8` Vision Pixel LUT Normalization**:
   - `c8d72931196a` (#21399) replaces Kimi-K3's decode attention-residual mix (+ RMSNorm) for `M=1..64` tokens (`KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS=64`) on `SM100/103/107` with a CuTe DSL PDL kernel (`trtllm::kimi_k3_attn_res_*`), folding the preceding MoE layer's residual update into the next layer's mix (`_DeferredMoEResidual`, saving two BF16 adds per MoE layer) and handing verify-only batches a bitwise-matching `R128c4` MXFP8 side output directly to KDA (`fp8_block_scale_low_m_gemm_prequantized`, alongside gated/sigmoid low-M GEMMs). On GB300 (in a CUDA graph behind a PDL producer), per-call latency at `M=7..64` drops from `3.3–11.0 us` (C++) to **`1.7–2.6 us` (up to `4.231x` speedup)**; on Rubin (`sm_107a`) at `M=32, N=9` with folded MoE add it drops from `6.01 us` to **`2.83 us` (`2.124x`)**; Rubin `helix16 b1 mtp7` decode step latency drops from `11.45 ms` to **`9.93 ms` (`-13.28%`)**.
   - `0160bf7a89f4` (#20973) adds `multimodal_config.vision_gpu_normalize: true` (opted in by Qwen3-VL and Gemma 4) to keep resized pixels in `uint8` across H2D transfer (**`2.0x` H2D bandwidth compression** vs BF16) and normalize on GPU via a per-channel 256-entry LUT that is bitwise identical to CPU preprocessing. On 4x GB200 serving `Qwen3-VL-235B-A22B NVFP4` (`c=8`), HTTP CPU preprocessing time drops **`11.2%–19.27%`** (`110.0 -> 88.8 ms` on cold chat) and mean steady-state TTFT drops from `183.2 ms` to **`151.8 ms` (`-17.14%`)** on screenshot sessions and `93.5 -> 80.4 ms` (`-14.01%`) on multimodal chat.
2. **vLLM Hybrid Spec Decode Sliding-Window Divisor Block Sizing, ModelOpt CUTLASS FP8 IPC Restarts & Lazy KV Offload Eviction-End Placement**:
   - `59519ec0ee0b` (#60252) adds a `divisor_of` constraint to `_largest_kernel_block_within` in `Attention.get_kv_cache_spec` so that when a GDN hybrid target (`1,648`-token primary block) pairs with a sliding-window DFlash2 drafter on a `[16, 32, 64]`-block backend, the SW block remains a divisor of the primary block and scales cleanly via `unify_kv_cache_spec_page_size` instead of inflating the scheduler LCM block **`4.0x`** (`1,648 -> 6,592`) and disabling fine-grained prefix-cache hits. On RTX PRO 5000 (`sm_120`, `Qwen3.8-27B NVFP4 + DFlash2 k=7`), prefix-cache hit rate jumps from `43.8%` to **`71.2%` (`+27.4` pts)**; on DGX Spark (`GB10`), a 168-turn agentic replay drops from `26.1 min` to **`20.0 min` (`1.305x` speedup, hit rate `56.3% -> 77.4%`)**.
   - `69d27bf8002a` (#64119) enables CUDA IPC weight-cache restarts for serialized ModelOpt `FP8_PB_WO` linear weights on CUTLASS (`116/116` layer regressions pass on B300); `a92efadb30f5` (#59383) adds `evict_first=True` to `BlockPool.free_blocks` so lazily offloaded GPU blocks return to the eviction front of the GPU free queue rather than the MRU end, preventing CPU offload cache churn when the CPU tier is smaller than the GPU KV cache.
3. **SGLang AITER Zero-Copy Strided-Q Unified Extend, R3 Non-Output Rank Capture Skip, DSpark Mamba Tracking Deduplication & RequestRetirer Consolidation**:
   - `27aedf7f93cc` (#47822) introduces `_unified_extend_q` in AITER `_forward_extend_unified` to return a zero-copy `view(-1, H, D)` whenever a QKV column slice after fused RoPE + KV write is already dense within each token (`stride(2)==1, stride(1)==head_dim`), eliminating `58.8–121.8 us/layer` (`2.0–4.3 ms/step` across 36 layers on `gpt-oss-120b`) and cutting `gfx1250` (`7,548` tokens) CUDA-graph unified-extend time from `72.7 us` to **`44.2 us` (`1.645x` speedup, `-39.20%`)** on `TP2 window` and `252.5 -> 196.9 us` (`1.282x`, `-22.02%`) on `TP1 window`.
   - `744af7ce724f` (#47553) skips R3 routed-expert capture and device buffer allocation on non-output ranks (`pp0·attn-tp0·attn-cp0`), shrinking `--tp 8` device buffers **`8.0x`** (`8 x 12 MB -> 1 x 12 MB`), saving `32 MiB/rank` across 6 ranks under `DP attention + DeepEP`, and reducing `c=1` ITL on 8x H200 by `1.4%` (`3.621 -> 3.571 ms`).
   - `435418253e5e` (#47494) removes duplicate `prepare_mamba_track_for_verify` execution in the DSpark worker (`1` refresh per step, **`96.0%`** `192/200` GSM8K on B200/GB300); `fd19cfb660f4` (#48327) consolidates **7** in-step finish sequences into `RequestRetirer.release_finished` (enforcing HiSparse row cleanup before KV/Mamba release).
4. **Dynamo `PoolRelay` Live State Sharing & Disaggregated Remote Prefill Cancellation Propagation**:
   - `8b4ad6a1b041` (#17716) shares live catalog, readiness, serving-load, and KV-load state across **2** router connection modes (relay-initiated and router-initiated) via `PoolRelay` and stops routing when the last ready Relay view is lost; `e6b5a99a7a45` (#14775) propagates client cancellation contexts to remote disaggregated SGLang prefill workers when both legs advertise `disagg_prefill_cancel_anytime_v1`.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 Inference Serving Stack: TensorRT-LLM (Blackwell/Rubin), vLLM v1, SGLang & NVIDIA Dynamo     |
+--------------------------------------------------------------------------------------------------------+
  [TensorRT-LLM Kimi-K3 CuTe DSL PDL Mix & GPU Vision LUT (#21399, #20973)]
    |-- Kimi-K3 Decode Attn-Res (#21399)--> CuTe DSL PDL M=1..64 + Deferred MoE Add + MXFP8 KDA (4.231x GB300)
    +-- GPU Vision Pixel Norm (#20973)  --> H2D uint8 (2.0x BW) + GPU Per-Channel LUT (-17.14% TTFT, -19.27% CPU)
                                        |
                                        v
  [vLLM Hybrid Spec Decode SW Divisor, FP8 IPC Cache & Lazy KV Offload (#60252, #64119, #59383)]
    |-- SW Divisor Block Sizing (#60252)--> divisor_of=1648 Avoids 6592 LCM (+27.4 pts Hit Rate, 1.305x Spark)
    |-- ModelOpt CUTLASS FP8 (#64119)   --> Block-FP8 (FP8_PB_WO) CUDA IPC Weight-Cache Engine Restarts (B300)
    +-- Lazy KV Offload Order (#59383)  --> BlockPool.free_blocks(evict_first=True) Prevents CPU Tier Churn
                                        |
                                        v
  [SGLang AITER Strided-Q, R3 Rank Skip, DSpark Mamba & Dynamo PoolRelay (#47822, #47553, #47494, #17716)]
    |-- AITER Strided-Q Extend (#47822) --> Skip q.contiguous() on Dense Token Slices (1.645x gfx1250 TP2 Window)
    |-- R3 Capture & DSpark (#47553)    --> Output-Rank-Only 1x12MB Buf (8.0x Reduction) + 1x Mamba Verify Track
    +-- Dynamo PoolRelay & Cancel (#17716)-> Shared Live PoolViews (2 Modes) + Remote Prefill Cancel Propagation
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182/run.py`](../../experiments/trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182/run.py) `--check`:
- `trtllm_kimi_k3_gb300_attn_res_max_speedup = 4.231`, `trtllm_kimi_k3_rubin_folded_moe_speedup = 2.124`, `trtllm_kimi_k3_rubin_mtp7_step_reduction_pct = 13.28`
- `trtllm_vision_gpu_norm_ttft_reduction_pct = 17.14`, `trtllm_vision_http_cpu_reduction_pct = 19.27`, `trtllm_vision_h2d_bandwidth_compression_ratio = 2.0`
- `vllm_sw_lcm_inflation_reduction_ratio = 4.0` (`6592 -> 1648`), `vllm_rtx_pro_5000_prefix_hit_gain_pts = 27.4`, `vllm_dgx_spark_agentic_replay_speedup = 1.305`
- `sglang_aiter_strided_q_tp2_window_speedup = 1.645`, `sglang_r3_tp8_buffer_reduction_ratio = 8.0`, `sglang_retirer_consolidated_finish_callsites = 7`
- `dynamo_pool_relay_connection_modes = 2`, `dynamo_orphaned_remote_prefill_on_compatible_cancel = 0`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Blackwell / Rubin Hybrid Attention Decode & Multimodal VLM Preprocessing** | Enable CuTe DSL PDL decode attention-residual mix + deferred MoE residual folding on GB300/Rubin Kimi-K3 (`M <= 64`, `#21399`, `4.231x`) and `vision_gpu_normalize: true` on Qwen3-VL / Gemma 4 (`#20973`, `-17.14%` TTFT). | Forcing `vision_gpu_normalize` on vision processors that resize in FP32 before normalization (e.g., Nemotron VL, which must stay on CPU). |
| **GDN + Sliding-Window Speculative Decoding & Tiered CPU KV Offload** | Use `divisor_of` SW block selection on GDN hybrid targets with DFlash2 drafters (`#60252`, `+27.4 pts` prefix hit rate) and `evict_first=True` lazy block return when CPU offload capacity is smaller than GPU KV (`#59383`). | Relying on `divisor_of` block scaling alone to mask unpatched Mamba state-column seeding bugs (`#55601` is still required for correctness). |
| **AMD `gfx1250` Unified Extend, MoE Routing Capture & Disaggregated Routing** | Enable `_unified_extend_q` zero-copy views on SGLang + AITER `gfx1250` (`#47822`, `1.645x` TP2 window), R3 non-output rank skip (`#47553`), and Dynamo `PoolRelay` (`#17716`). | Retiring source prefill KV during an in-flight async transfer on unpatched SGLang 0.5.23 images lacking `#47250` transfer-completion guards (`#14775`). |
