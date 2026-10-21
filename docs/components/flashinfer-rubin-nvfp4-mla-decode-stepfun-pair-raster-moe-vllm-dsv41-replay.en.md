# Deep Dive: FlashInfer Rubin (`sm_107a`) Cake NVFP4 MLA Decode, StepFun Pair-Raster/2-CTA MoE & vLLM DeepSeek-V4.1 Replay Fix

> **中文母稿**: [flashinfer-rubin-nvfp4-mla-decode-stepfun-pair-raster-moe-vllm-dsv41-replay.md](flashinfer-rubin-nvfp4-mla-decode-stepfun-pair-raster-moe-vllm-dsv41-replay.md)  
> **Phase 41 (`priority: 50`) · Deterministic Experiment**: [`experiments/flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41/run.py`](../../experiments/flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41/run.py)  
> **7-Card 4:5 Visual Report**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 1. Executive Summary

1. **FlashInfer Rubin (`sm_107a` / R200) Cake NVFP4 MLA Paged Decode (`PR #6931` / `adca7b87d2`)**:
   - Introduces the experimental **Cake dense NVFP4 MLA paged decode** backend on NVIDIA R200 (`sm_107a` / compute capability `10.7`, `208 SMs`, `120 MiB L2`, CUDA 13.4 PTX ISA 9.4) for absorbed multi-head latent attention (`512` latent + `64` rope channels, Kimi-K3 / DeepSeek-V3 geometry).
   - **Compact 352 B/token KV Cache & 356 B/row Query Quantization**: Reads a `352 B/token` paged cache (`256 B` packed E2M1 `ckv_cache` + `32 B` E4M3 `ckv_sf_cache` + `64 B` E4M3 `kpe_cache`, page size `64`) and quantizes BF16 `[.., 576]` queries into `356 B/row` (`256 B q_nope + 32 B q_sf + 64 B q_rope + 4 B FP32 q_scale`) with `q_scale = max(amax(nope)/(6*448), amax(rope)*kpe_scale/(448*ckv_scale))` so RoPE dot products land directly in the latent logit unit.
   - **Mixed 4-CTA / 2-CTA Cluster Execution**: Packs `50` four-CTA clusters and `4` two-CTA pair clusters on 208 SMs (`46` fours + `14` pairs on 212 SMs). Across 16 R200 configurations against TokenSpeed MLA 2 (2.0.3), Cake achieves a **`1.0436x`** overall geometric mean speedup (**`1.1039x`** on `Q=4` rows and **`1.2483x`** peak at `H96-Q4-B64-KV128K`: `1623.8 -> 1300.8 us`).
2. **FlashInfer Cake StepFun SwiGLU Fused MoE Regeneration (`PR #6934` / `fd4de5acce`)**:
   - Regenerates all 5 precision families (`bf16`, `fp8`, `mxfp8`, `nvfp4`, `nvfp4_bf16tok`, **`111` kernels/arch**) on `sm_100a` (B200) and `sm_103a` (GB300) for StepFun SwiGLU fused MoE (`H=4096, I=1536, E=64, top_k=8`).
   - **Pair-Rasterized BF16 FC1 & 2-CTA NVFP4 FC2**: Pair-rasterized BF16 FC1 tile-128 cuts L2 weight reads by **`2.0x`** across paired M-tiles, while the two-CTA `nvfp4_bf16tok` FC2 tile-256 kernel delivers **`1.0323x`** on B200 (`211.0 -> 204.4 us`) and **`1.0245x`** on GB300 (`196.8 -> 192.1 us`) at `T=1024`.
3. **vLLM DeepSeek-V4.1 `DecoderReplayLayers` Target-Token Scoring Fix (`PR #62652` / `85752a43c3`)**:
   - Updates `DeepseekV41ModelState.add_request` so `_keeps_rows_np[req_index]` is set when `params.prompt_logprobs is not None or params.prompt_logprob_token_ids is not None`, preserving all **`300` rows** across single-chunk (`300`) and two-chunk (`150+150`) prefills instead of trimming them to `WINDOW=64` (recovering **`236`** and **`172`** dropped rows respectively).

---

## 2. First-Hand Upstream Commits

| Repository | Commit / PR | Date | Core Files | Evidence ID |
|---|---|---|---|---|
| `flashinfer-ai/flashinfer` | [`adca7b87d2` (`#6931`)](https://github.com/flashinfer-ai/flashinfer/commit/adca7b87d23d426e7ae0eae6df3992d6b44532e8) | `2026-10-21` | `flashinfer/experimental/cake_rubin_mla_nvfp4_decode/{cake_backend.py,cake_jit.py}` | `EV-2026-355` |
| `flashinfer-ai/flashinfer` | [`fd4de5acce` (`#6934`)](https://github.com/flashinfer-ai/flashinfer/commit/fd4de5acce18fd1c747c8425555f082dd4452cc3) | `2026-10-21` | `csrc/cake_stepfun_moe/{dispatch.inc,jit_Compile.py}` | `EV-2026-356` |
| `vllm-project/vllm` | [`85752a43c3` (`#62652`)](https://github.com/vllm-project/vllm/commit/85752a43c3cee76d9234704e189cc2ca69f24712) | `2026-10-21` | `vllm/models/deepseek_v41/nvidia/model_state.py` | `EV-2026-357` |

---

## 3. Architectural Mechanics

### 3.1 Rubin (`sm_107a`) Cake NVFP4 MLA Paged Decode Pipeline

- **4-CTA + 2-CTA Mixed Cluster Scheduling**: Ranks 0–1 hold `128` query rows each in tensor memory and compute $S^T$ over 128-key tiles (2 pages) via two `kind::mxf4nvf4` `K=128` block-scaled MMAs plus one `kind::f8f6f4` RoPE MMA, then push E4M3 probabilities $P$ (with per-row UE8M0 scale) to Ranks 2–3, which convert $V$ on-chip to E4M3 and accumulate $O$ with `kind::mxf8f6f4` `K=64 M256 N256` MMAs.
- **256-Byte Aligned 4-Slot Caller Workspace**: Partitions `partial_o`, `partial_lse`, `lse_scratch`, and `plan_block` on `256 B` boundaries (`8,749,056 B` query workspace and `1,617,348,608 B` max decode workspace at `H=96, Q=4, B=64`), enabling zero-allocation, zero-sync `runner.launch()` inside CUDA Graphs.

### 3.2 StepFun Fused MoE & vLLM DeepSeek-V4.1 Replay Invariants

| Dimension | Metric | Deterministic / Reported Value | Status |
|---|---|---|---|
| Rubin NVFP4 MLA | KV Token Bytes / Query Row Bytes | `352 B` / `356 B` | `PASS` |
| Rubin NVFP4 MLA | R200 (`208 SM`) All-16 / Q4-8 Geomean Speedup | `1.0436x` / `1.1039x` | `PASS` |
| Rubin NVFP4 MLA | Peak Speedup at `H96-Q4-B64-KV128K` (`MLA2 -> Cake`) | `1.2483x` (`1623.8 -> 1300.8 us`) | `PASS` |
| StepFun MoE | Kernels per Arch / FC1 L2 Weight Read Reduction | `111` / `2.0x` | `PASS` |
| StepFun MoE | `nvfp4_bf16tok` `T=1024` Speedup (`B200` / `GB300`) | `1.0323x` / `1.0245x` | `PASS` |
| vLLM DSV4.1 Replay | Target-Token Replay Rows Preserved (`300` / `150+150`) | `300` (`+236`) / `[150, 150]` (`+172`) | `PASS` |

---

## 4. Production Adoption Recommendations

1. **Batch-Aware Dispatch on Rubin (`sm_107a`)**: Route `batch >= 4` (`q_len=4` MTP) traffic to `cake_rubin_mla_nvfp4_decode` for up to `1.25x` latency reduction while keeping `B=1` on TokenSpeed MLA 2.
2. **Graph-Safe Host Plan Preparation**: Always supply `seq_lens_host` to `prepare_cake_mla_nvfp4_paged_decode` outside CUDA Graph capture so `runner.launch()` remains strictly zero-sync.
3. **Mandatory vLLM Patch for Target-Token Scoring**: Deploy `85752a43c3` on all DeepSeek-V4.1 serving nodes using `prompt_logprob_token_ids` to prevent `WINDOW=64` truncation during `DecoderReplayLayers` execution.
