# TensorRT-LLM Rubin SM104 DeepSeek-V4/DSpark Indexer Gather & Capture Consolidation, vLLM D2D FlashInfer Warmup Seeding & Secondary-Tier KV Store-Failure Backpressure, SGLang MiniMax-M3.1 NVFP4 / MSA-v1 Decode / HiCache Page-Row Copy / Disagg KV Checksum Signature / DP-Attention Alignment, and AI Infra Innovator Unit Economics

> **Language / 语言**: [English](trtllm-rubin-vllm-seed-sglang-msa-nvfp4.en.md) | [中文](trtllm-rubin-vllm-seed-sglang-msa-nvfp4.md)
> **Phase**: Phase 73 (`priority: 82`, Track B)
> **Question ID**: `q-phase73-trtllm-rubin-vllm-seed-sglang-msa-nvfp4`
> **Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. Executive Summary & Architectural Insights

Between `2026-10-24` and `2026-10-25`, upstream commits across `NVIDIA/TensorRT-LLM`, `vllm-project/vllm`, and `sgl-project/sglang` landed major serving invariants across Rubin (`sm_104`) and Blackwell (`sm_100`/`sm_103`) sparse attention, daemon-to-daemon warmup seeding, KV offload failure backpressure, and disaggregated KV integrity verification:

1. **TensorRT-LLM DeepSeek-V4 / DSpark Optimizations on Rubin (`sm_104`)** (`TensorRT-LLM@d7e0ecac92`):
   - **Dead `latent_cache` Slice Elimination**: In `attentionOp.cpp`, passes `std::nullopt` for `latent_cache` whenever KV cache reuse or chunked context is active, avoiding unused tensor-slice construction on every step.
   - **Fused `invokeIndexerKCacheQuantAndScatter` with `sm != 104` Gate**: In `indexerKCacheGather.cu`, fuses per-token FP8 E4M3 quantization, scale store, and KV cache scatter into a single 8-warp (256-thread) kernel when `is_IndexK_fp8 && head_dim == 128 && quant_block_size == 128`, `max_Num_Tokens_Per_Block <= 8`, and **`sm != 104` (excluding Rubin SM104)**, falling back to separate quantize + scatter launches on `sm == 104` or large token-per-block ratios.
   - **Deferred DSpark `TargetCaptureExport.consolidate()`**: Moves `consolidate()` out of the per-layer decoder loop in `dspark.py` to execute once after all 61 layers finish (reducing consolidation calls from $4 \to 1$).
2. **vLLM D2D FlashInfer Warmup Cache Seeding & Secondary-Tier KV Store-Failure Backpressure** (`vllm@41dc477d24`, `vllm@2b1f75c80f`):
   - **D2D `SeedTransport` & Unanimous Rank Verification**: Bounds unauthenticated peer requests to a whole-message deadline (`_REMOTE_REQUEST_TIMEOUT_S = 10.0s`, `_MAX_REMOTE_REQUEST_SIZE = 64 KiB`) to prevent slow-trickle DoS on the serial `accept` loop, while relaxing authenticated transfers to `300.0s`. `_load_shared_autotune_cache` enforces unanimous cache hits across `tune_group` via `all_gather_object` before skipping the collective-bearing dummy-run sweep.
   - **Store-Failure Backpressure (`ENOSPC`/`EDQUOT` Guard)**: Fixes `KVOffloadManager._process_finished_jobs` so failed secondary-tier stores invoke `detector.on_store_failed()`. After `DEFAULT_FAILURE_THRESHOLD = 3` consecutive failures on `"fs"` or `"obj"` tiers, `EMABackpressureDetector` spikes the EMA to `_FAILURE_EMA_FACTOR = 4.0 * high_water_s` (`40 ms` for `fs`, `200 ms` for `obj`), causing `ThrottledDropPolicy(ramp_factor=2.0)` to drop 100% of stores until idle decay lets a probe store through.
3. **SGLang MiniMax-M3.1 NVFP4 + MSA-v1 Decode + HiCache `page_row_copy` + Disagg KV Signature + DP-Attention Alignment** (`sglang@a99c7d050f`, `174bd2976d`, `5d11819f3a`, `ad950cdcd9`, `9b230c1dae`):
   - **MiniMax-M3.1 NVFP4 Stack & `rcp.approx.ftz.f32` Bit-Identity**: Adds `minimax_m3_nvfp4` (`group_size=16`, MXFP8 projections + NVFP4 routed experts + NVFP4 KV/Index-K + `DSparkMiniMaxDraftModel`). `nvfp4_store_kv_index.cuh` packs each token-head into `64 B` E2M1 payload + `8 B` E4M3 scales = **`72 B` (`3.556x` compression over `256 B` BF16)** using `rcp.approx.ftz.f32` to avoid ~`0.1%` nibble tie-breaking drift vs. IEEE division; `msa_v1/attn/dequant.cuh` uses PTX `mul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4` on CUDA $\ge 13.4$ and feeds `FlashInferQ8K8Decode` zero-copy HND views.
   - **HiCache `page_row_copy` 2-CTA PCIe Saturation**: Uses just `kPageRowBlockQuota = 2` CTAs (`512` threads = 8 workers of 64 lanes, `kVecsPerLane = 16` `16 B` non-coherent loads/stores in flight per lane) to saturate B300 PCIe at **`51 GB/s` H2D and `52 GB/s` D2H** (vs. `46 / 35 GB/s` at `8 / 32` vecs/lane), leaving all other SMs for compute.
   - **Disagg `kv_layout_signature` & DP-Attention Bucket/SUM_LEN Alignment**: Ships a 64-bit odd `blake2b` layout signature alongside each request's Adler-32 KV checksum so heterogeneous TP/PP prefill-decode pairs log one warning and skip verification rather than aborting every request; aligns DP-Attention prefill graph token counts to the smallest common captured bucket (`[0, 310, 180, 420] -> [512, 512, 512, 512]`) and selects `SUM_LEN` in decode when `CustomAllReduceV2` can hold the summed buffer.

---

## 2. AI Infra Innovator Unit Economics Comparison (Rule 13)

| Dimension | Fireworks AI / Together AI (Blackwell B200/GB300 GPU Clouds) | Cerebras Wafer-Scale Engine (WSE-3) | Groq LPU (Deterministic Tensor Streaming) |
|---|---|---|---|
| **Memory Hierarchy & Bandwidth** | HBM3e (`8 TB/s` B200 / `288 GB` GB300) + PCIe Gen5/NVLink host tiered cache | Wafer-scale on-chip SRAM (`44 GB` on-wafer, `21 PB/s`) | On-chip SRAM (`230 MB/chip`, `80 TB/s/chip`) + synchronous C2C fabric |
| **NVFP4 KV (`72 B/tok/head`) & Sparse Attention Impact** | **Massive**: `3.56x` KV compression + 2-CTA `51–52 GB/s` `page_row_copy` multiplies concurrent long-context capacity per node by `3.5x` | **Bounded by on-wafer SRAM pool**: zero HBM offload overhead, but multi-tenant 128K+ KV pools hit on-wafer SRAM capacity limits first | **Ultra-low single-stream ITL, high chip-count scaling for long KV**: excels at low-batch high tok/s, but 128K+ KV caches require sharding across many LPUs |
| **Disaggregated PD & Elastic TP/EP** | Native heterogeneous prefill-TP / decode-DP-EP scaling with `kv_layout_signature` and RDMA/EFA transfer | Whole-wafer execution domain | Statically scheduled compiler pipeline; dynamic top-k sparse page gather gains are bounded by static worst-case slot reservation |

---

## 3. Reproducible Verification (`own-offline`)

```powershell
python experiments\trtllm_rubin_vllm_seed_sglang_msa_nvfp4_sim\run.py --check
```

| Module | Key Invariant / Metric | Verification Outcome | Evidence Tier |
|---|---|---|---|
| TensorRT-LLM Rubin `sm_104` Indexer Gather | Fused quant+scatter gated on `sm != 104 && max_tokens_per_block <= 8`; DSpark `consolidate()` moved post-loop | `sm=104` bypasses fusion; 61-layer 4-capture `consolidate` calls reduced `4 -> 1` | `implementation` + `own-offline` |
| vLLM D2D `SeedTransport` & KV Offload Backpressure | Unauth `10s / 64 KiB` cap; unanimous `tune_group` hit; 3 failed `fs`/`obj` stores spike EMA to `4.0x` high-water | `7/8` rank hit rejects skip; 3 failures raise EMA to `40 ms / 200 ms` (`100%` drop probability) | `implementation` + `own-offline` |
| SGLang MiniMax-M3.1 NVFP4 + MSA-v1 + Disagg + DP | `72 B/token/head` (`3.5556x` compression); `kVecsPerLane=16` hits `51/52 GB/s`; DP prefill `[0,310,180,420] -> [512]*4` | All byte ratios, odd `blake2b` layout signatures, and DP buckets verified | `implementation` + `author-benchmark` + `own-offline` |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise (vLLM Store-Failure Backpressure Probe Recovery)**: Once 3 consecutive `ENOSPC` errors raise an `fs` tier's EMA to `40 ms` (`4.0 * 10 ms`), `ThrottledDropPolicy(ramp_factor=2.0)` drops 100% of writes. How does the tier recover once disk space is freed?
   - *Answer Guide*: `EMABackpressureDetector` applies exponential idle decay to the stored EMA across elapsed time $\Delta t_{\text{idle}}$ Since the last store attempt; once the decayed EMA drops below `ramp_factor * high_water_s = 20 ms`, drop probability falls below `1.0`, admitting a probe write that updates the EMA with real I/O latency.
2. **Falsifiable Hypothesis (Disaggregated Radix Cache vs. Whole-Prompt KV Checksum)**: Enabling whole-prompt Adler-32 KV checksums alongside `--disaggregation-decode-enable-radix-cache` will falsely abort every prefix-cache hit because prefill only computes and transmits the uncached suffix pages.
