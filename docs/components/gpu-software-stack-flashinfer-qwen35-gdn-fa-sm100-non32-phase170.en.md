# GPU Software Stack Deep Dive (Phase 170): FlashInfer Qwen 3.5 GDN `frost` SM100 Backend, Cake Softmax Chunk Rules & FA SM100 Non-Multiple-of-32 `head_dim` Backward

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `56472bf4fc` | 2026-10-31 | `[Experimental] Qwen 3.5 GDN block: decode and MTP verify behind a backend registry, with an SM100 CuTe DSL backend (#8131)` |
| `flashinfer-ai/flashinfer` | `caa3fd0f20` | 2026-10-29 | `perf(cake_softmax): widen balanced per-CTA chunk rules on B200 and GB300; narrow 6-tile 8-CTA rules on GB300 (#7822)` |
| `flashinfer-ai/flashinfer` | `861d03c456` | 2026-10-29 | `perf(cake_mhc): cut the eager host time of the mHC with-grad entry points (#7823)` |
| `flashinfer-ai/flashinfer` | `c58617e6da` | 2026-10-29 | `test(comm): fix Ulysses initialization fault injection (#7876)` |
| `Dao-AILab/flash-attention` | `94345b4568` | 2026-09-08 | `[CuTe,Bwd,Sm100] Support head_dim not a multiple of 32 in backward (#2698)` |
| `Dao-AILab/flash-attention` | `f4c68c8a74` | 2026-09-08 | `[CuTe] Include callable defaults in JIT cache hashes (#2773)` |
| `NVIDIA/cutlass` | `c900acd605` | 2026-07-08 | `Preserve Blackwell example performance across CUDA backends (#3370)` |
| `triton-lang/triton` | `2122e24efc` | 2026-10-16 | `[Backend] Avoid emitting llvm bitcast when loading (#12319)` |

---

## 2. Architectural Analysis

### 2.1 FlashInfer Qwen 3.5 GDN Fused Block (`56472bf4fc`) & Cake Softmax Rules (`caa3fd0f20`)
- **Single-Call Qwen 3.5 GDN Block (`#8131`)**: Introduces `flashinfer.qwen_3_5_gdn_block` with a pluggable backend registry (`register_backend`) and the `frost` SM100 CuTe DSL backend, fusing BA projection, causal conv update, gating, gated delta rule over the SSM state pool, gated RMSNorm, and static FP8 output into 1 call (**`2.23x`** faster: `46.8 us` -> `21.0 us`).
- **Blackwell Cake Softmax Chunking (`#7822`)**: Uses `2048`-element tiles and `ceil(vocab / CTAs)` rounded up to `32` elements (`19008` elements for `vocab=152064, CTAs=8`), while narrowing 6-tile 8-CTA dispatch rules on GB300 (`sm_103a`).

### 2.2 FlashAttention CuTe SM100 Non-Multiple-of-32 `head_dim` Backward (`94345b4568`) & JIT Hash (`f4c68c8a74`)
- **Non-Multiple-of-32 `head_dim` Backward (`#2698`)**: Fixes SM100 backward crashes and corrupt `dK/dV` for `head_dim % 32 != 0` (e.g., ViT `head_dim=72, 104`) by slicing `predicate_k` per-`m` and aligning the `dQ` TMEM accumulator to a multiple of 32 (`[96, 128]`).
- **Callable Defaults in JIT Cache Hashes (`#2773`)**: Hashes callable default argument values so score-modification callables with changed defaults never collide in JIT cache (`0` collisions).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/flashinfer_qwen35_gdn_cake_softmax_fa_sm100_non32_phase170/run.py --check`:
- `gdn_speedup_x`: **`2.23x`** (`6` -> `1` kernel)
- `balanced_cta_chunk_elems`: **`19008`** (`gb300_narrowed_max_tiles_8cta = 6`)
- `padded_tmem_hdims`: **`[96, 128]`** (`non32_bwd_max_err = 0.0`, `jit_callable_default_collisions = 0`)
