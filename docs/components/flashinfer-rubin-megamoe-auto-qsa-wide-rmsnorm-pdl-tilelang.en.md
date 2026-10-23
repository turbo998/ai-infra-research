# Deep Dive into FlashInfer Rubin (`sm_107a`) MegaMoE `knobs="auto"` Collective Workspace Coordination, `cake_qsa_nvfp4` (`sm_80`) Wide Whole-Tile Decode, `cake_rmsnorm_train` Non-Cooperative PDL Backward Launch, and TileLang 16-Bit Inverse/`nextafter` Subnormal Math Lowering

> **Language / 语言**: [English](flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang.en.md) | [中文](flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang.md)  
> **Phase**: 70 (`priority: 79`, Track A — GPU Kernel & Quantization)  
> **Evidence Tiers**: `implementation` (fixed-commit source audit) + `author-benchmark` (author-measured A100-SXM4-80GB / A100-SXM4-40GB / GB300 / B200 reports) + `own-offline` (deterministic standard-library simulation `experiments/flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Recent commits across FlashInfer (`sm_107a` Rubin, `sm_100a`/`sm_103a` Blackwell, `sm_80` Ampere) and the TileLang CUDA/Ascend compiler address five critical kernel engineering problems:

1. **Rubin (`sm_107a`) MegaMoE `knobs="auto"` Cross-Rank Collective Workspace Coordination (`flashinfer-ai/flashinfer@3a4e382bc46e`, `#6535`)**: Under Expert Parallelism (EP), `RubinMegaMoEWorkspace.create(..., device_buffer_bytes=None)` allocates and frees symmetric heap memory collectively (`destroy()`). Why does rank-local cache invalidation or asymmetric entry into `autotune_v2()` cause immediate collective deadlocks or invalidate captured CUDA Graphs?
2. **`cake_qsa_nvfp4` Wide, Long-Stream Whole-Tile Decode Selection on `sm_80` (`flashinfer-ai/flashinfer@179af3f6e08c`, `#7133`)**: When `rows * num_kv_heads * splits` exceeds a full `sm_80` residency wave ($2 \times \text{sm\_count} = 216$ CTAs on A100) and each CTA streams across $\ge 16$ 64-token tiles, why does the K-first `decode` kernel's per-tile K/V shared-memory reuse barrier become a bottleneck, and how does `decode_whole_tile` achieve a **`1.060x–1.085x`** speedup on A100-80GB/40GB?
3. **`cake_rmsnorm_train` Backward Replacing Whole-Device Cooperative Launch with PDL Serialization (`flashinfer-ai/flashinfer@28b2c6d93f33`, `#7131`)**: Why does using `cudaLaunchAttributeCooperative` for single-kernel cross-CTA `dweight` reduction stall **`10x–280x`** behind a 1-thread resident side-stream kernel, and how does replacing it with a non-cooperative chunk-claiming main kernel plus a `cudaLaunchAttributeProgrammaticStreamSerialization` (`griddepcontrol.wait`) reduce kernel bound concurrent slowdown to **`<= 1.13x`**?
4. **`trtllm_fused_moe_workspace_size` Unshuffled FP8 Routing Initialization & Explicit `tactic` Sizing (`flashinfer-ai/flashinfer@75d4c6e61ddc`, `#7025`)**: Why did `trtllm_fused_moe_workspace_size` return `0` bytes for unshuffled FP8 (`ds_fp8` / `fp8_qdq`), and how is single-tactic post-tuning workspace sizing exposed?
5. **TileLang CUDA 16-Bit Inverse/Binary Math Lowering & Bit-Exact `nextafter` Subnormal Stepping (`tile-ai/tilelang@43f13c7ad1d1` `#3172`, `@f1a34efea68c` `#2860`, `@d85a70a97218` `#3582`)**: Why do floating-point comparisons in 16-bit `nextafter` flush FP16/BF16 subnormals under CUDA `--use_fast_math` (FTZ), and why must arch-less CUDA targets fail closed instead of falling back to TVM's silent `sm_50` default?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 Rubin (`sm_107a`) MegaMoE `knobs="auto"` & Collective EP Workspace Coordination (`#6535`)

In `flashinfer/fused_moe/rubin_megamoe.py`, `rubin_megamoe(..., knobs="auto")` integrates with `autotune_v2()` via `_RubinMegaMoERuntimeRunner`:

- **Two-Stage `DynamicKnobSpec` Filtering**:
  1. `knobs[0]` (`num_tokens_per_cta` $\in \{64, 128, 256\}$) selects the valid CTA token tile;
  2. `knobs[1]` (`option_idx`) indexes only the pipeline knob tuples (`prefetch_ahead`, `combine_slots`, `grad_prefetch_ahead`) valid for the chosen `num_tokens_per_cta`.
- **Versioned Tuning Cache Contract**: `BACKEND_REVISION = "sm107-block-scaled-1667b47a-runtime-options-v1"` is recorded in `TuningContext` and checked on `ConfigRecord` load so binary revisions automatically invalidate stale tuning records.
- **`coordinate_workspace_replacement` Collective EP Synchronization**:
  - In `_AutoWorkspaceHolder`, because symmetric workspace creation and destruction are collective across `ep_group`, `coordinate_workspace_replacement(ep_group, replace=not reusable)` executes `dist.all_gather_object(decisions, (context.tuning_enabled, replace), group=ep_group)`:
    - If only a subset of ranks has `tuning_enabled=True` (`any(tuning_ranks) and not all(tuning_ranks)`), it raises `RuntimeError` immediately rather than deadlocking when tuning ranks sweep all candidate configs while non-tuning ranks run once;
    - If any rank reports `replace=True`, every rank in `ep_group` replaces its workspace together.
- **CUDA Graph Workspace Preservation**:
  - When `torch.cuda.current_stream(device).is_capturing()` is true, online autotuning is disabled;
  - If a workspace used during stream capture is later replaced outside capture, the old workspace is appended to `_captured_workspaces` instead of calling `destroy()`, preserving symmetric heap addresses bound into captured CUDA Graphs.

### 2.2 `cake_qsa_nvfp4` (`sm_80`) Wide, Long-Stream Whole-Tile Decode (`#7133`)

In `flashinfer/experimental/cake_qsa_nvfp4/lib.py`, FlashInfer adds `decode_whole_tile` for `sm_80` (A100, `108` SMs) governed by `select_wide`:

- **Root Cause**: The baseline `sm_80` K-first `decode` kernel stages K and V sequentially through the same shared-memory buffer per 64-token tile, requiring an extra CTA barrier per tile. When `rows=128` (`splits=2`, `512` CTAs) streams across `33` tiles (`2051` selected tokens $\implies 17$ tiles/CTA), eliminating that per-tile K/V staging barrier via `decode_whole_tile` outweighs the slightly higher shared-memory footprint.
- **`select_wide` Selection Formula**:
  $$
  \operatorname{select\_wide} = (\text{form} == \text{"split"}) \land (\text{rows} \times H_{KV} \times \text{splits} > 2 \times N_{\text{SM}}) \land \left(\left\lceil \frac{\text{tiles}}{\text{splits}} \right\rceil \ge 16\right)
  $$
  where $\text{tiles} = \lceil N_{\text{selected}} / 64 \rceil$. On A100 ($2 N_{\text{SM}} = 216$), `rows=128, H_KV=8, selected_tokens=2051` ($\text{tiles}=33, \text{splits}=2$) yields $512 > 216$ CTAs and $\lceil 33/2 \rceil = 17 \ge 16$ tiles/CTA, selecting `decode_whole_tile`, whereas `rows=64` ($\text{splits}=4 \implies 9$ tiles/CTA) stays on `decode`.
- **Author-Measured A100 Benchmarks (`author-benchmark`, `rows=128, num_kv_heads=8, selected_tokens=2051`)**:
  - **A100-SXM4-80GB**: NHD `198.50 -> 182.91 us` (**`1.085x`**); HND `198.53 -> 185.63 us` (**`1.069x`**);
  - **A100-SXM4-40GB**: NHD `218.48 -> 201.83 us` (**`1.083x`**); HND `220.37 -> 207.99 us` (**`1.060x`**).

### 2.3 `cake_rmsnorm_train` Non-Cooperative PDL Backward Launch (`#7131`)

In `flashinfer/cake_rmsnorm_train.py` and `flashinfer/cake_rmsnorm_train_backend/`, FlashInfer replaces whole-device `cudaLaunchAttributeCooperative` backward launches on `sm_100a`/`sm_103a`/`sm_107a` (`H in {512, 1536, 2048, 4096, 6144}`):

- **Cooperative Launch Head-of-Line Blocking Hazard**: A cooperative launch requires every CTA in the full-device grid to co-reside simultaneously. A single 1-thread resident kernel on a side stream occupies one warp slot on one SM, blocking the cooperative launch until the side kernel exits and causing **`10x–280x`** latency spikes.
- **Two-Kernel Non-Cooperative + PDL Design**:
  - **Kernel 1 (Chunk-Claiming Main Kernel)**: Plain non-cooperative launch where CTAs claim row chunks via atomic `counters`, write `dx`, and accumulate chunk partial sums into `partial[n_chunks, H]`. If a side stream occupies an SM, remaining CTAs simply claim the extra chunks.
  - **Kernel 2 (`dweight` Reduce Kernel)**: Launched with `cudaLaunchAttributeProgrammaticStreamSerialization` (`programmaticStreamSerializationAllowed = 1`) and PTX `griddepcontrol.wait` to fold `partial -> dweight` without host/inter-kernel launch bubbles.
- **Author-Measured Benchmarks (`author-benchmark`, GB300 `152` SMs / B200 `148` SMs)**:
  - **Side-Stream Contention**: Reduces slowdown behind a 1-thread resident side kernel from **`10x–280x`** down to at most **`1.13x`** (at $T=65536$);
  - **Single-Stream Latency**: GB300 $(T=4096, H=512)$ improves `16.0 -> 13.7 us` (**`1.168x`**) and $(T=16384, H=2048)$ `27.6 -> 24.4 us` (**`1.131x`**); B200 $(T=4096, H=512)$ improves `18.8 -> 15.9 us` (**`1.182x`**) and $(T=16384, H=2048)$ `31.1 -> 27.9 us` (**`1.115x`**).

### 2.4 `trtllm_fused_moe_workspace_size` Unshuffled FP8 Fix & Explicit `tactic` Sizing (`#7025`)

In `csrc/trtllm_fused_moe_kernel_launcher.cu` and `flashinfer/fused_moe/core.py`:
- **Root Cause**: Calling `prepare_moe_runner(tactic)` before `prepare_routing()` left routing mode flags unset on unshuffled FP8 (`ds_fp8` / `fp8_qdq`), causing `get_Configs()` to find zero valid configs and return **`0` bytes**.
- **Fix & API Extension**: `scratch_.begin_measure()` and `prepare_routing()` now execute before runner measurement, and `trtllm_fused_moe_workspace_size(..., tactic=(tile_tokens_dim, config_index))` measures scratch for a single tuned tactic rather than the max across all tiles.

### 2.5 TileLang 16-Bit Inverse/Binary Math & `nextafter` Subnormal Stepping (`#3172`, `#2860`, `#3582`)

1. **16-Bit Math Intrinsics & `hnextafter` Bit Stepping (`tile-ai/tilelang@43f13c7ad1d1`, `#3172`)**:
   - Registers CUDA lowering rules for `asin, acos, asinh, acosh, atanh, log1p, atan2, hypot, copysign, ldexp, nextafter`, guarded by `CCCL_VERSION >= 3000000` (`TL_HAS_LIBCUDACXX_16BIT_MATH`) with FP32 fallbacks.
   - Implements `hnextafter` over raw 16-bit integer encodings (`x_mag = x_bits & 0x7FFF`, stepping `0x0000 -> (y_bits & 0x8000u) | 0x0001u` and `x_bits ± 1`) so CUDA fast-math FTZ cannot flush BF16/FP16 subnormals (`0x0001..0x007F`).
2. **Rejecting Arch-Less CUDA Targets (`@f1a34efea68c`, `#2860`) & Ascend PTO Signed Warp Reductions (`@d85a70a97218`, `#3582`)**:
   - `_cuda_target_from_arch` now raises an actionable error when a CUDA target lacks `-arch=sm_xx` instead of letting TVM silently default to `sm_50`; Ascend PTO `T.warp_reduce_max/min` now preserves signed integer semantics.

---

## 3. Deterministic Offline Verification (`own-offline`)

Experiment script: `experiments/flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang/run.py` (results saved in `summary.json` and guarded by `tests/test_phase70_experiments.py`):

| Verification Target | Parameters | Measured / Simulated Output | Architectural Takeaway |
|---|---|---|---|
| Rubin MegaMoE `coordinate_workspace_replacement` | 4 EP ranks, Rank 1 cache miss (`replace=True`) | `collective_replace=True`, `preserved_in_captured_list=True` | Prevents symmetric-heap deadlock and protects captured CUDA Graphs |
| Rubin MegaMoE mixed autotune guard | Rank 0/2 `tuning=True`, Rank 1/3 `False` | `mixed_autotune_error=True` | Fails fast if EP ranks enter `autotune_v2()` asymmetrically |
| `cake_qsa_nvfp4` (`sm_80`) `select_wide` | A100 (`108` SMs), `selected_tokens=2051` (`33` tiles) | `rows=128 -> decode_whole_tile` (`17` tiles/CTA, `1.085x`); `rows=64 -> decode` (`9` tiles/CTA) | Uses whole-tile decode only when CTAs exceed 2 waves and stream $\ge 16$ tiles |
| `cake_rmsnorm_train` PDL launch pair | GB300 (`152` SMs) / B200 (`148` SMs), $T \in [4096, 65536]$ | Side-stream stall reduced from `10x–280x` to `<= 1.13x`; GB300 $(4096, 512)$ `1.168x` | Eliminates whole-device cooperative launch starvation under multi-stream overlap |
| `trtllm_fused_moe_workspace_size` | `quant_mode="ds_fp8"`, `use_shuffled_weight=False` | Legacy `0 B` $\to$ Fixed all-tiles `24,096,768 B`, `tactic=(16,0)` `15,184,896 B` | Restores non-zero workspace sizing on unshuffled FP8 and enables single-tactic sizing |
| TileLang `hnextafter` bit stepping | `+0.0 -> -1.0`, `0x007F -> 1.0`, `0x0080 -> 0.0` | `0x8001` (`-min_subnormal`), `0x0080` (`min_normal`), `0x007F` (`max_subnormal`) | Preserves exact BF16 subnormal and signed-zero transitions under FTZ |

---

## 4. Solutions Architect Exercises & Falsifiable Questions

1. **Sizing Exercise**: On an `sm_80` GPU with `108` SMs running `cake_qsa_nvfp4` (`form="split"`), what is the minimum `selected_tokens` count required for `rows=128, num_kv_heads=8, splits=2` to select `decode_whole_tile`?
   - *Answer*: Since $\text{ctas} = 128 \times 8 \times 2 = 2048 > 216$, `select_wide` requires $\lceil \text{tiles} / 2 \rceil \ge 16 \implies \text{tiles} \ge 31 \implies \text{selected\_tokens} \ge 30 \times 64 + 1 = 1921$ tokens.
2. **Falsifiable Hypothesis**: Does keeping `cudaLaunchAttributeCooperative` for RMSNorm backward remain safe if communication kernels run on a lower-priority CUDA stream?
   - *Falsification*: Any resident warp on any stream—regardless of stream priority—prevents a full-device cooperative launch from acquiring all SMs simultaneously until that warp exits, producing `10x–280x` stalls that the non-cooperative chunk-claiming + PDL pair avoids.
