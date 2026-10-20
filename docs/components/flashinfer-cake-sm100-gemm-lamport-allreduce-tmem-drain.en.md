# FlashInfer Cake Blackwell (`SM100`/`SM103`) Fused TP GEMM + MNNVL Lamport All-Reduce, `tcgen05.wait::ld` TMEM Drain, and 2-CTA Producer-Tail Synchronization Contracts

> **中文版**: [flashinfer-cake-sm100-gemm-lamport-allreduce-tmem-drain.md](flashinfer-cake-sm100-gemm-lamport-allreduce-tmem-drain.md) | **Associated Experiment**: [`experiments/flashinfer_cake_gemm_lamport_allreduce/README.md`](../../experiments/flashinfer_cake_gemm_lamport_allreduce/README.md)

### One-Sentence Takeaway

FlashInfer's **Cake fused tensor-parallel GEMM + MNNVL Lamport All-Reduce (`cake_gemm_allreduce`)** on Blackwell (`SM100` / `SM103`) collapses a row-parallel linear layer $\text{out} = \sum_{r=0}^{W-1} X_r W_r^T$ into a single kernel launch per rank (`192` threads / 6 warps, 64-column $\times$ `token_tile` `tcgen05` accumulator tiles): it dispatches **One-Shot NVLS multicast publish** for $M \le 16$ and **Two-Shot scatter / owner-reduce / multicast broadcast** for $16 < M \le 128$, sanitizes BF16 `-0.0` (`0x8000` $\to$ `0x0000`) to eliminate collisions with the 32-bit Lamport dirty sentinel `0x80000000u`, and accumulates cross-rank partials in fixed rank order ($r = 0 \dots W-1$) via `add.rn.f32.bf16` with a single final `__float22bfloat162_rn` rounding to guarantee **bitwise-identical output across all ranks** (**1.1x–1.3x** faster than `mm_bf16 + allreduce_fusion` for $M \le 64$, level at $M = 128$, and superseded by the two-call path by **~1.3x** at $M \ge 256$ due to per-tile weight re-streaming); alongside this, two PTX `tcgen05` memory-consistency fixes add `tcgen05.wait::ld.sync.aligned` before `tcgen05.dealloc` across 39 persistent Kimi-K3 AttnRes kernels (`1.0000x` mean speed ratio on B300 `sm_103a`) and enforce a `ring_depth` (`5..12`) producer-tail `mbarrier_wait` drain before the peer CTA load warp exits across 9 2-CTA (`cta_group::2`) mixed FP8$\times$FP4 GEMM kernels.

### Component Responsibilities and Upstream Locations

| Module & File | Upstream Repository & Commit / Blob SHA | Core Engineering Responsibility |
| :--- | :--- | :--- |
| `flashinfer/comm/cake_gemm_allreduce.py` & `flashinfer/jit/cake_gemm_allreduce.py` | `flashinfer-ai/flashinfer` commit `e563bf9867ba6cf5ef0f7becff219343c6ac519e` / blobs `d41ff11eac`, `b6bef816ac` | Defines the public `cake_gemm_allreduce` API, the 12-entry `ROUTES` table (`world_size` $\in \{2,4,8\} \times$ `token_tile` $\in \{16,32,64,128\}$), `select_route`, and triple-buffered Lamport workspace sizing `required_workspace_bytes`. |
| `csrc/cake_gemm_allreduce/cake_gemm_allreduce_binding.cu` & `cake_gemm_allreduce_kernels.cu` | `flashinfer-ai/flashinfer` commit `e563bf9867ba6cf5ef0f7becff219343c6ac519e` / blobs `380a309fef`, `9b2ef6d494` | Encodes 3D K-major `SWIZZLE_128B` TMA tensor maps via `EncodeKMajorMap` and implements the 12 `sm_100a`/`sm_103a` kernels (6-stage TMA ring, 32-column TMEM accumulators, `-0.0` BF16 sanitization, Lamport `0x80000000u` polling reduction, background dirty-buffer clearing, and PDL support). |
| `csrc/experimental/cake_kimi_k3_attn_res/*_kernel.cu` | `flashinfer-ai/flashinfer` commit `3bf6d3202239a107989451257c6f594cb5d2f008` | Inserts `tcgen05.wait::ld.sync.aligned` immediately before `barrier.sync 9, 288` and `tcgen05.dealloc` in 39 288-thread persistent Kimi-K3 AttnRes kernels to close TMEM anti-dependency hazards. |
| `csrc/experimental/deepgemm_mixed_gemm/*_kernel.cu` | `flashinfer-ai/flashinfer` commit `b815220c91936cd7e11a0b91fddb3307a4317a26` | Appends a `ring_depth` (`5..12`) producer-tail `mbarrier_wait(mma_done_addr + load_stage * 8)` loop after the TMA load warp tile loop in 9 2-CTA (`cta_group::2`) mixed FP8$\times$FP4 GEMM kernels. |

### Core Implementation Paths, Data Layout, and Synchronization Contracts

1. **12-Route Dispatch Table, 3D K-Major TMA Descriptors, and the Decode-Band Regime (`cake_gemm_allreduce.py:24-85`, `cake_gemm_allreduce_binding.cu:40-165`)**:
   - Each rank holds row-parallel shards $X_r \in \text{BF16}[M, K_r]$ ($K_r \bmod 128 = 0$) and $W_r \in \text{BF16}[N, K_r]$ ($N \bmod 64 = 0$).
   - `select_route(M)` selects `(token_tile=16, twoshot=False)` (one-shot) for $M \le 16$, and `(token_tile, twoshot=True)` with `token_tile` $\in \{32, 64, 128\}$ for $16 < M \le 32$, $32 < M \le 64$, and $M > 64$.
   - `EncodeKMajorMap` builds 3D K-major `CUtensorMap` descriptors with `global_dim = {64, rows, K / 64}`, `global_strides = {K * 2, 128}` bytes, `box_dim = {64, box_rows, 2}`, and `CU_TENSOR_MAP_SWIZZLE_128B`.
   - The launch grid is `gridDim = (N / 64, ceil(M / token_tile), 1)` with `192` threads per CTA (Warps `0..3` epilogue/all-reduce consumers, Warp `4` `tcgen05` MMA issuer, Warp `5` 6-stage TMA loader). For $M \le 128$, `gridDim.y = 1`, so each column tile streams the weight shard $W_r$ once; for $M \ge 256$, `gridDim.y = ceil(M / 128) >= 2`, re-streaming $W_r$ per token tile and making `mm_bf16 + allreduce_fusion` ~1.3x faster.
2. **Lamport Dirty Sentinel `0x80000000u`, BF16 `-0.0` Sanitization, and Bitwise-Identical Cross-Rank Reduction (`cake_gemm_allreduce_kernels.cu:250-450`)**:
   - Epilogue warps `0..3` read FP32 accumulators from 32 TMEM columns via `tcgen05.ld.sync.aligned.16x256b.x2.b32` (`n16`) or `.x4.b32` (`n32`), convert to BF16 into padded shared-memory scratch `xpose_h[tok * 66 + col]` (stride `66` prevents bank conflicts), and after `barrier.sync 8, 128` pack 8 BF16 values (`int4` / 16B) per thread.
   - **Why BF16 `-0.0` (`0x8000`) Must Be Sanitized**: The MNNVL Lamport buffer uses `0x80000000u` as its 32-bit dirty sentinel. In a packed BF16 pair (`uint32_t`), `0x80000000u` corresponds to `lo = 0x0000` (`+0.0`) and `hi = 0x8000` (`-0.0`). Before publishing, `_packed16_sanitize` replaces any 16-bit half equal to `0x00008000u` with `0u` (`+0.0`), preserving exact IEEE-754 arithmetic equivalence while guaranteeing no valid published word ever matches `0x80000000u`.
   - **Bitwise-Identical Reduction**: Every rank polls `ld.volatile.global.v4.b32` until none of the four 32-bit words equals `0x80000000u`, accumulates across ranks in fixed order $r = 0, 1, \dots, W-1$ using inline PTX `add.rn.f32.bf16` into FP32 registers, and applies `__float22bfloat162_rn` once at the end. Meanwhile, all CTAs cooperatively clear the previous dirty Lamport buffer and `(blockIdx.x == 0, blockIdx.y == 0, tid_1 == 0)` advances `(buffer_idx + 1) % 3` once `buffer_flags[8] == gridDim.x * gridDim.y`.
3. **Blackwell `tcgen05` Hardware Synchronization Fixes (`#6740` & `#6819`)**:
   - **TMEM Anti-Dependency Drain (`3bf6d32022`)**: Per the PTX ISA `tcgen05` memory consistency model, a true register dependency on a `tcgen05.ld` result orders only the register value, **not** anti-dependencies against `tcgen05.dealloc`. Inserting `tcgen05.wait::ld.sync.aligned` before `barrier.sync 9, 288` across 39 persistent 288-thread Kimi-K3 AttnRes programs prevents consumer warp 0 from deallocating TMEM columns while another warp's `tcgen05.ld` is still in flight (`0.9984..1.0024` speed ratio, `1.0000x` mean on B300 `sm_103a`).
   - **2-CTA Cluster Producer-Tail Drain (`b815220c91`)**: In `cta_group::2` GEMMs, the leader CTA's MMA warp releases each TMA ring stage via `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster` (mask `0b11`). Because PTX provides no ordering between separate `tcgen05.commit` operations (§9.7.18.6), the peer CTA's load warp could exit after its last refill while the last `ring_depth` (`5..12`) multicast arrives are still in flight toward its shared memory (undefined behavior under PTX ISA §2.2.2). Adding a `ring_depth` `mbarrier_wait` producer tail after the tile loop drains all in-flight arrives with zero overhead on GB300 (`8.16 -> 8.04 us` at $M=512, N=5120, K=2304$).

### Key Formulas, Units, and Engineering Assumptions

1. **Per-Buffer Lamport Workspace Bytes (`required_workspace_bytes`)**:
   $$\text{Bytes}_{\text{buffer}}(M, N, W) = \begin{cases} 2 \cdot M \cdot N \cdot W & \text{if } M \le 16 \text{ (One-Shot)} \\ 4 \cdot \lceil M / W \rceil \cdot W \cdot N & \text{if } M > 16 \text{ (Two-Shot)} \end{cases}$$
   Total triple-buffered capacity is $3 \times \text{Bytes}_{\text{buffer}}(M, N, W)$.
2. **Dynamic Shared Memory Allocation**:
   $$\text{SMEM}_{\text{total}}(\text{tile}) = 1024 + 6 \times 16384 + 6 \times (256 \cdot \text{tile}) + \text{XPOSE}(\text{tile})$$
   Yielding `126,080 B`, `152,704 B`, `206,080 B`, and `214,528 B` for `tile` $\in \{16, 32, 64, 128\}$.
3. **Unawaited 2-CTA Multicast Commits at Loop Exit**:
   $$N_{\text{unawaited\_without\_tail}} = \min(N_{\text{tiles}}, D_{\text{ring}}), \quad N_{\text{unawaited\_with\_tail}} = 0$$

### Synthetic Experiment and Verification Results (`experiments/flashinfer_cake_gemm_lamport_allreduce`)

Running `python experiments/flashinfer_cake_gemm_lamport_allreduce/run.py --check` verifies:

| Metric / Scenario ($N=7168, K_r=2304, W=8$) | $M=16$ (One-Shot) | $M=32$ (Two-Shot) | $M=64$ (Two-Shot) | $M=128$ (Two-Shot) | $M=256$ (Out-of-Band) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `(token_tile, twoshot)` | `(16, False)` | `(32, True)` | `(64, True)` | `(128, True)` | `(128, True)` |
| `gridDim` / Total CTAs | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 1, 1]` (`112`) | `[112, 2, 1]` (`224`) |
| Dynamic SMEM Bytes | `126,080 B` | `152,704 B` | `206,080 B` | `214,528 B` | `214,528 B` |
| Single / Triple Lamport Buffer | `1,835,008 B` / `5.51 MB` | `3,670,016 B` / `11.01 MB` | `7,340,032 B` / `22.02 MB` | `14,680,064 B` / `44.04 MB` | `29,360,128 B` / `88.08 MB` |
| Weight Re-Stream Factor / MiB | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `1x` (`31.50 MiB`) | `2x` (`63.00 MiB`) |
| Fused vs Two-Call Speedup / Route | **`1.289x`** (`cake_gemm_allreduce`) | **`1.242x`** (`cake_gemm_allreduce`) | **`1.168x`** (`cake_gemm_allreduce`) | **`1.057x`** (`cake_gemm_allreduce`) | **`0.785x`** (`mm_bf16+allreduce_fusion`) |

- **Sentinel Collision & Bitwise-Identical Reduction Verification**: For partial `(0.0, -0.0)`, unsanitized packing yields `0x80000000` (`unsanitized_dirty_sentinel_collision = True`), whereas `_packed16_sanitize` eliminates the collision (`sanitized_dirty_sentinel_collision = False`). Under high-dynamic-range partials `(65536.0, 0.00390625, -65536.0)`, ring-order summation produces `2` distinct outputs across 8 ranks, while Cake's fixed rank-order FP32 summation yields `0x80800080` identically across all 8 ranks (`cake_bitwise_identical = True`).

### Architect Design Exercises and Falsifiable Questions

1. **Sizing Exercise**: Why does Two-Shot routing at $W=8, N=7168, M=20$ require $2 \times \lceil 20 / 8 \rceil \times 8 \times 7168 \times 2 = 6,881,280\text{ B}$ per Lamport buffer instead of $2 \times 20 \times 7168 \times 2$?
   - *Answer*: Phase 1 scatters tokens by `dest_rank = tok % W` and `dest_token = tok / W`, so each owner rank needs $\lceil M / W \rceil = 3$ token slots across all $W$ ranks for both Stage 0 (scatter gather) and Stage 1 (broadcast), totaling $2 \times 3 \times 8 \times 7168 \times 2 = 6,881,280\text{ B}$.
2. **Falsifiable Hypothesis**: Does the `ring_depth` producer-tail loop hang on short problems where $N_{\text{tiles}} < D_{\text{ring}}$?
   - *Verification*: No—stages that were never filled retain their initial phase-0 free credits, so their `mbarrier_wait` calls pass immediately while the $N_{\text{tiles}}$ filled stages properly drain in-flight multicast arrives.
