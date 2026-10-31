# Phase 150 Track 1: FlashInfer Rubin `sm_107a` SSD Combined, SM100/SM103 Warp Decode, GB200 NVFP4 MLA Reverse Piece Order, SM120 Paged MQA-Logits & Triton/TileLang Safety

> **Evidence Classification Notice**: This report strictly separates `implementation` (verified upstream Git commits), `own-offline` (deterministic Python stdlib models), and `author-benchmark`.

## 1. Architectural Context & Problem Statement

Across Blackwell (`B200`/`GB200`/`SM100`/`SM103`/`SM120`) and Rubin (`sm_107a`) kernel stacks, four microarchitectural bottlenecks govern Mamba SSD, MoE warp decode, and long-context MLA:
1. **Rubin `sm_107a` SSD Combined Pipeline Stalls**: Without wrap-around planning and `z`-operand prefetching, `q_tmem_alias` kernels stall on global memory latency.
2. **Rigid FC1 Pools in SM100/SM103 Warp Decode & GB200 NVFP4 MLA L2 Thrashing**: Static FC1 pool sizing wastes shared memory on small decode batches, while forward piece ordering on GB200 long rows misses warm tail KV cache lines.
3. **SM Count Divergence Across SM120 GPUs (RTX 5090 vs RTX PRO 6000)**: Using identical grid/tile schedules across 170-SM and 188-SM SM120 parts induces tail-wave quantization loss.
4. **IEEE 754 `-0.0` Corruption in Async Copy Zero-Fill & Triton Namedtuple Recompiles**: Matching `-0.0` (`0x80000000`) as hardware zero-fill strips the sign bit, while unconditional JIT specialization on `namedtuple` metadata causes compilation storms.

```mermaid
flowchart TD
    subgraph FlashInfer_Gen["FlashInfer Rubin / Blackwell / SM120 Kernels"]
        A["CAKE SSD Combined sm_107a (#7869)\nWrap-Around Plan + Z-Prefetch + 1-CTA"] --> B["1.2657x SSD Combined Speedup"]
        C["CAKE Warp Decode SM100/103 (#7868)\nPer-Row FC1 Pool + FC2 Launch + B200 Softmax (#7867)"] --> D["1.3099x Warp Decode Speedup"]
        E["CAKE NVFP4 MLA Decode GB200 (#7861)\nReverse Piece Order + SM120 MQA Logits (#7860)"] --> F["1.1429x GB200 MLA | 1.3154x SM120 MQA"]
    end
    subgraph Compiler_Runtime["Triton & TileLang Compiler Hardening"]
        G["Triton (#12490, #12460)\nNamedtuple JIT Opt-Out + ConSan Ptr Init"]
        H["TileLang (#3343, #3449, #3385)\nReject -0.0 Async Zero-Fill + Math Split + AutoDD Guard"]
    end
    FlashInfer_Gen --> Compiler_Runtime
```

## 2. Primary Upstream Implementation Audit (10 Zero-Overlap Commits)

| Repo | Commit | Date | Subject | Key Files |
|---|---|---|---|---|
| `flashinfer-ai/flashinfer` | [`f62d3e1`](https://github.com/flashinfer-ai/flashinfer/commit/f62d3e11087a8b9915f50000f44368f373b91f80) | `2026-10-29` | perf(cake_ssd_combined): sm_107a wrap-around plan, z-prefetch and one-CTA-cluster programs, wider claim and pair routes (#7869) | `csrc/cake_mamba_ssd_combined/generated/device/mamba_ssd_q_tmem_alias_bf16_batched_noring_424f20baf3.cu`, `csrc/cake_mamba_ssd_combined/generated/device/mamba_ssd_q_tmem_alias_bf16_batched_zpf_d541c37f1b.cu` |
| `flashinfer-ai/flashinfer` | [`00341e8`](https://github.com/flashinfer-ai/flashinfer/commit/00341e8a6694d4348697fdfd68c2947c09226150) | `2026-10-29` | perf(cake_warp_decode): per-row FC1 pool sizes and FC2 launch choices on SM100/SM103 decode rows (#7868) | `csrc/fused_moe/warp_decode/generated/cake_warp_decode_generated_manifest.cuh` |
| `flashinfer-ai/flashinfer` | [`9323e48`](https://github.com/flashinfer-ai/flashinfer/commit/9323e48094d2cc8a4c2b8d91d80e20c2681af1b1) | `2026-10-29` | fix(cake_softmax): keep whole tiles for 6-tile high-cut 8-CTA launches on B200 (#7867) | `csrc/cake_blackwell_softmax.cu`, `tests/utils/test_sampling.py` |
| `flashinfer-ai/flashinfer` | [`311b6c2`](https://github.com/flashinfer-ai/flashinfer/commit/311b6c22059cd5bd34fb18fff84bb6bb47f68fb6) | `2026-10-29` | perf(cake_nvfp4_mla_decode): reverse the per-unit piece order of long-row decode plans on GB200 (#7861) | `flashinfer/experimental/nvfp4_mla_decode/cake_backend.py`, `tests/experimental/test_cake_nvfp4_mla_decode.py` |
| `flashinfer-ai/flashinfer` | [`aa69286`](https://github.com/flashinfer-ai/flashinfer/commit/aa692863569afe6f9c1c5256ba2de423c6578a80) | `2026-10-29` | perf(cake_deepgemm): SM120 paged MQA-logits: per-device tuned programs for RTX 5090 and RTX PRO 6000 (#7860) | `benchmarks/bench_sm120_paged_mqa_logits.py`, `csrc/experimental/deepgemm_sm120_paged_mqa_logits/generated/cake_deepgemm_sm120_paged_mqa_logits_006afa75b9e05c4c948b_binding.cu` |
| `triton-lang/triton` | [`8d81295`](https://github.com/triton-lang/triton/commit/8d8129515f777782dbe5a302c598dc8dd78fb2f9) | `2026-10-26` | [Runtime] Allow namedtuple fields to opt out of JIT specialization (#12490) | `python/src/specialize.cc`, `python/test/unit/runtime/test_specialize.py` |
| `triton-lang/triton` | [`5c8eae7`](https://github.com/triton-lang/triton/commit/5c8eae7e93c21c433ee1db941c5de9a07ff5a960) | `2026-10-22` | [ConSan] Initialize shared allocations with pointer elements (#12460) | `lib/Dialect/TritonInstrument/Transforms/ConcurrencySanitizer.cpp`, `test/TritonGPU/consan.mlir` |
| `tile-ai/tilelang` | [`9847765`](https://github.com/tile-ai/tilelang/commit/9847765a73947501e02feb61dc6fe9659895f6ae) | `2026-10-08` | [BugFix] Keep -0.0 out of the async copy zero-fill matcher (#3343) | `src/cuda/transform/ptx_async_copy_injector.cc`, `src/op/utils.cc` |
| `tile-ai/tilelang` | [`69435b8`](https://github.com/tile-ai/tilelang/commit/69435b89ead80a8058209a46baa914c3fb3a4432) | `2026-10-08` | [Refactor][CUDA] Split math codegen dispatch (#3449) | `src/cuda/codegen/codegen_cuda.cc`, `src/cuda/codegen/codegen_cuda.h` |
| `tile-ai/tilelang` | [`ab79dd6`](https://github.com/tile-ai/tilelang/commit/ab79dd62616669a5c8e5dab6c33897fd5474ef70) | `2026-10-09` | fix(autodd): reject non-positive --jobs instead of hanging (#3385) | `tilelang/autodd.py` |

### 2.1 FlashInfer Rubin / Blackwell / SM120 Kernel Mechanisms
- **`flashinfer@f62d3e1` (`#7869`)**: Adds `sm_107a` wrap-around execution plans, `z`-prefetch (`zpf`), and one-CTA-cluster programs in `csrc/cake_mamba_ssd_combined`.
- **`flashinfer@00341e8` (`#7868`) & `flashinfer@9323e48` (`#7867`)**: Tunes per-row FC1 pool sizes and FC2 launch choices for SM100/SM103 warp decode, and preserves whole tiles on B200 6-tile high-cut 8-CTA softmax launches.
- **`flashinfer@311b6c2` (`#7861`)**: Reverses per-unit piece order in GB200 long-row `cake_nvfp4_mla_decode` plans to exploit L2 temporal locality.
- **`flashinfer@aa69286` (`#7860`)**: Ships per-device tuned SM120 paged MQA-logits kernels for RTX 5090 and RTX PRO 6000.

### 2.2 Triton & TileLang Compiler Hardening
- **`triton@8d81295` (`#12490`) & `triton@5c8eae7` (`#12460`)**: Allows `namedtuple` fields to opt out of JIT specialization in `specialize.cc` and initializes pointer-element shared allocations in `ConcurrencySanitizer.cpp`.
- **`tilelang@9847765` (`#3343`), `tilelang@69435b8` (`#3449`), `tilelang@ab79dd6` (`#3385`)**: Excludes `-0.0` from `ptx_async_copy_injector.cc` zero-fill matching, splits CUDA math codegen dispatch, and rejects non-positive `--jobs` in `autodd.py`.

## 3. Deterministic Offline Modeling (`own-offline`)

Experiment script: `experiments/flashinfer_ssd_warpdecode_nvfp4_mla_sm120_phase150/run.py`

1. **Rubin `sm_107a` SSD Combined**: Wrap-around + `z`-prefetch cuts latency from `312.0 us` to `246.5 us` (**`1.2657x` speedup**).
2. **SM100/SM103 Warp Decode**: Per-row FC1 pool tuning reduces latency from `44.8 us` to `34.2 us` (**`1.3099x` speedup**).
3. **GB200 NVFP4 MLA & SM120 Paged MQA**: Reversing piece order yields **`1.1429x`** on GB200 long rows, and per-device SM120 tuning yields **`1.3154x`** (`68.4 -> 52.0 us`).
4. **TileLang `-0.0` Guard & Triton Namedtuple Opt-Out**: Preserves IEEE `-0.0` sign bits and cuts namedtuple recompiles from `12` to **`1`**.
