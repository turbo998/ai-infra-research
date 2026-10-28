# Deep Dive: FlashInfer CAKE `tinygemm2` `PACKED_M64_N8` / KCP Pool Sharing / Rubin `SM107` AOT / cuDNN 4-Slot Prefill Cache / Blackwell `bf16_fp4` / Mamba SSU / MiniMax-H3 `SM120` / DSV4, FlashMLA `SM100` WAR Barrier & TileLang Lowering Guards (Phase 123)

> **中文版**: [flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123.md](flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123.md)  
> **Question ID**: `q-gpu-flashinfer-tinygemm-kcp-cudnn-bf16fp4-mamba-h3-dsv4-flashmla-tilelang-phase123` (`priority: 132`)  
> **Component**: `gpu-software-stack`  
> **Evidence Tier**: `implementation` (first-hand GitHub commit & diff inspection) + `author-benchmark` (upstream R200/GB300/B200/RTX PRO 6000 benchmarks, explicitly separated from local synthetic models) + `own-offline` (zero-dependency Python 3.12 standard-library verification)  
> **Reproducible Experiment**: [`experiments/flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123/run.py`](../../experiments/flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123/run.py)

---

## 1. Executive Summary & Architectural Problem Statement

Across Blackwell (`SM100`/`SM103`/`SM120`) and Rubin (`SM107`, 212 SMs) GPU software stacks, Phase 123 addresses four critical kernel and compiler bottlenecks:
1. **Rubin `sm_107a` Narrow-Batch GEMM Occupancy & Missing AOT Modules**: On 212-SM Rubin R200 GPUs, 8-row unbiased GEMMs routed to `M128_N8_2CTA` launch only 64 CTAs and pay 2-CTA cluster barrier overhead; meanwhile `flashinfer/aot.py` omitted `has_sm107` when registering `SM100f` CUTLASS FP8/MoE modules, triggering a **789 s** runtime JIT compilation that caused vLLM DeepSeek-R1 FP8 GSM8K to exceed its 1,200 s startup timeout.
2. **KCP Communication Pool Redundancy & cuDNN Prefill Geometry Thrashing**: `cake_kda_kcp` allocated separate forward and backward send buffers plus eager ring buffers even under `all_to_all`; meanwhile `_cudnn_prepared_state` cached only a single `CudnnPrefillGraph`, forcing full graph preparation on every alternating geometry switch.
3. **Blackwell/SM120 Mixed-Precision & SSM Kernel Dispatch**: Pre-cluster-barrier weight/scale loads in `cake_blackwell_bf16_fp4`, `sm_100a` batch-64 z-gated Mamba2 `selective_state_update` routing, and warp-specialized `ATTENTION_WS_NODELTA_LEAD/TAIL` (`384` threads, `80,896 B` dynamic SMEM) for SM120 NVFP4 varlen attention on 188-SM RTX PRO 6000.
4. **FlashMLA `SM100` WAR Race & TileLang Type/TMA Guards**: `head64` sparse decode lacked a `NamedBarrier` between reading `rowwise_max_buf[idx ^ 64]` and overwriting `rowwise_max_buf[idx] = li`; TileLang multiplied modular index coefficients into 16-bit `DataType::lanes()`, overflowing past `32,767`.

---

## 2. First-Hand Code & Implementation Analysis

### 2.1 FlashInfer CAKE `tinygemm2` `PACKED_M64_N8` (`#7735`) & `cake_kda_kcp` Pool Sharing (`#7714`)
- **`PACKED_M64_N8` Single-CTA `tcgen05` Program (`1d622fdb`)**:
  - Adds a 9th `TinyGemm2ProgramId::PACKED_M64_N8 = 8` computing `(16, 8)` (`output_features × batch_rows`) per single CTA with `101,376 B` dynamic SMEM and `input_features_multiple = 64` on `sm_107a` (Rubin R200) for unbiased 8-row GEMMs in the window $(N=1536, K\in[2048,3072]) \lor (N=2048, K\in[2048,2560]) \lor (N=4096, K\in[3072,6144])$.
  - For `(8, 4096, 4096)`, `PACKED_M64_N8` launches $4096/16 = 256$ single-CTA blocks across 212 SMs instead of 64 2-CTA blocks, reducing R200 latency from `6.24 us` to `5.98 us` (**1.0435x**; 53 rerouted 8-row shapes improve by **1.01x–1.44x**, `author-benchmark`).
- **`cake_kda_kcp` Send-Pool Aliasing & Lazy Ring Buffers (`e4c7e4a6`)**:
  - Aliases `SHARED_POOLS = {"b1_send": "c1_send", "b2_send": "c2_send"}` and defers `LAZY_POOLS = ("ring_out", "ring_in")` until `exchange="ring"`, cutting distinct per-`(device, dtype)` buffers from 10 to **6** under `all_to_all` (**8.0 MiB** saved at $B=2, H=32, D=128, V=128$ BF16) and **8** under `ring`.

### 2.2 Rubin `SM107` AOT Registration (`#6795`) & cuDNN 4-Slot Prefill Cache (`#7720`)
- **Rubin `SM107` AOT Precompilation (`4d0f564e`)**: Registers `gen_cutlass_fused_moe_sm100_module` and `gen_gemm_sm100_module_cutlass_fp8` when `has_sm107` is set, eliminating the **789 s** runtime JIT compilation and cutting vLLM DeepSeek-R1 FP8 GSM8K startup+eval wall time from `1,254 s` (timeout `> 1,200 s`) to `465 s` (**2.6968x**).
- **4-Slot `_cudnn_prepared_cache` (`42494623`)**: Retains up to `_MAX_CUDNN_PREPARED_CACHE_ENTRIES = 4` prepared `CudnnPrefillGraph` entries and rebinds live plan metadata via `_rebind_prepared_prefill_state()`, cutting B200 first-layer geometry-switch host time from `121.69 -> 74.89 us` (**1.6249x**, saving **46.80 us** on B1) and `117.31 -> 69.64 us` (**1.6845x**, saving **47.67 us** on B3).

### 2.3 Blackwell `bf16_fp4` (`#7732`), Mamba2 SSU (`#7724`), MiniMax-H3 `SM120` (`#7730`) & DSV4 (`#7729`)
- **Blackwell `bf16_fp4` Pre-Barrier Loads (`e9908762`)**: Issues weight and scale loads before `barrier.cluster.arrive/wait` in the `M=128` 2-CTA pair-tile (`768x2112x2048`), improving GB300 (`9.264 -> 9.149 us`, **1.0126x**) and B200 (`10.101 -> 9.992 us`, **1.0109x**).
- **Mamba2 `selective_state_update` `sm_100a` (`6bac9852`)**: Routes batch-64 `64h/8g` z-gated decode on 148-SM B200 to `mtp_short_ring_heads64_groups8_t1_stp_z` (`40.440 -> 38.368 us`, **1.0540x**; with softplus `40.480 -> 38.472 us`, **1.0522x**).
- **MiniMax-H3 `SM120` NVFP4 Warp Specialization (`4b948ed8`)**: Adds `ATTENTION_WS_NODELTA_LEAD/TAIL` (`384` threads: 8 QK/softmax + 4 PV warps, `80,896 B` dynamic SMEM, `13 -> 15` programs) on 188-SM RTX PRO 6000 when $\sum L_i^2 \in [2^{23}, 24576^2]$ and weighted mean length $\le 40960$ (`single_4096` `603.01 -> 579.25 us`, **1.041x**; `seg3_ragged` `688.10 -> 671.70 us`, **1.024x**).
- **`cake_dsv4` Sparse MLA (`1531ff91`)**: 89/197 shapes faster on GB300, 36/197 faster on B200, **0 slower**.

### 2.4 FlashMLA `SM100` WAR Barrier (`#241`) & TileLang Guards (`#3453`, `#3484`, `#3369`, `#3367`)
- **FlashMLA `SM100` `head64` WAR Fix (`395cb3af`)**: Inserts `NamedBarrier::arrive_and_wait(128, NamedBarriers::wg0_sync)` between reading `plan.rowwise_max_buf[idx_in_warpgroup ^ 64]` and writing `plan.rowwise_max_buf[idx_in_warpgroup] = li`, preventing warps 0..1 from corrupting `rowwise_max_buf[0..63]` before warps 2..3 read it.
- **TileLang Lowering Guards**: Separates `scalar_read_coeff` from 16-bit `DataType::lanes()` (`#3453`, `0573808d`), verifies `CUtensorMap` declarations before recording `desc_name_map` (`#3484`, `10866657`), restricts CUDA `AtomicAddx4` to `float*` (`#3369`, `fab9a39f`), and checks `K % micro_size_k == 0` in `gemm_sp` (`#3367`, `fca340e5`).

---

## 3. Deterministic Offline Verification (`own-offline`)

Run `python experiments\flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123\run.py --check`:

| Subsystem & Metric | Baseline | Phase 123 Result | Architectural Impact |
| :--- | :--- | :--- | :--- |
| `tinygemm2` `(8, 4096, 4096)` (`sm_107a`) | `M128_N8_2CTA` (`64` CTAs, `6.24 us`) | `PACKED_M64_N8` (`256` CTAs, `5.98 us`) | **1.0435x** speedup without 2-CTA cluster sync |
| `cake_kda_kcp` buffers per `(device, dtype)` | `10` eager buffers | `all_to_all`: `6` / `ring`: `8` | Saves **4 buffers** (`8.0 MiB`) or **2 buffers** (`4.0 MiB`) |
| Rubin `SM107` DeepSeek-R1 FP8 startup | `1,254.0 s` (`789 s` JIT, timeout) | `465.0 s` (AOT `SM100f` registration) | **2.6968x** faster startup; passes `1,200 s` budget |
| cuDNN Prefill 4-slot cache (B200 B1 / B3) | `121.69 us` / `117.31 us` (0 hits) | `74.89 us` / `69.64 us` (5 hits) | Saves **46.80 us / 47.67 us** (**1.68x**) |
| FlashMLA `SM100` `head64` WAR reduction error | `120.0` (`li=128.0` overwrites `max=8.0`) | `0.0` (`NamedBarrier(128, wg0_sync)`) | Eliminates cross-warp WAR numerical corruption |
| TileLang `scalar_read_coeff` (`65536 × 4`) | Overflows 16-bit `DataType::lanes()` | Preserves `coeff=65536, lanes=4` | Enables large-stride vectorized scalar reads |
