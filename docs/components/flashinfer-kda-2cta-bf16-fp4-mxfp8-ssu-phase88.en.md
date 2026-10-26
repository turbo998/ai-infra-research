# FlashInfer KDA 2-CTA CP State Walks, BF16×FP4 Early TMEM Free & Preloop Scales, Grouped MXFP8 Integer UE8M0 Quantize, Mamba2 SSU P-Split/TMA-Ring & TileLang SM120 Register-A GEMM Unroll (Phase 88)

> **Language / 语言**: [中文 (`flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88.md`)](flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88.md) | **English (Current)**

- **Queue ID**: `flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88` (`priority: 97`)
- **Evidence ID**: `ev-flashinfer-kda-2cta-bf16-fp4-mxfp8-ssu-phase88-2026`
- **Verification Status**: `own-offline` (Primary upstream source audit + deterministic Python 3.12 stdlib verification in [`experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py`](../../experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py); GPU hardware timings are upstream `author-benchmark` results on B200/GB300/SM107/SM120)

---

## 1. Executive Summary & Architectural Problem

On NVIDIA Blackwell architectures (`sm_100a` B200, `sm_103a` GB300, `sm_107a`, and workstation `sm_120` RTX PRO 6000 Blackwell), hybrid linear attention (Kimi Delta Attention, KDA / Mamba2 Selective State Update, SSU) and low-precision (NVFP4 W4A4, BF16×FP4 W4A16, MXFP8 E4M3+UE8M0) kernels face five critical microarchitectural bottlenecks:
1. **SM Residency Under-Utilization in Context-Parallel (CP) KDA State Walks**: Each `(segment, head)` maintains a $128 \times 128$ FP32 recurrent state ($65,536\text{ B}$). A full-width 1-CTA record fits only 1 CTA per SM; when a CP rank holds only 1–2 long document chains, many SMs sit idle.
2. **Cluster Barrier Tail Latency & Redundant Scale Loads in Blackwell `mm_bf16_fp4`**: Waiting for a final cluster-wide barrier before freeing 2-CTA `tcgen05` TMEM stalls epilogue overlap, while small-$M$ split-K register-streaming kernels repeatedly load and convert E4M3 block scales inside the main loop.
3. **FP32 Division Bottleneck in Grouped MXFP8 Quantization**: Computing one UE8M0 scale byte per 32 elements via FP32 division at $K \in \{2048, 4096, 7168\}$ throttles throughput on sparse MoE masks.
4. **CTA-Pair (`cta_group::2`) All-Gather Matmul Multicast Arrive Race (PTX §2.2.2)**: The leader CTA's MMA warp releases TMA ring stages via multicast `tcgen05.commit` to the `mma_done` mbarrier of both CTAs. Because the peer CTA's load warp only waited before refills, it could exit the CTA while the last `ring_depth` multicast arrives were still in flight toward its shared memory.
5. **Local-Memory Spills in SM120 Register-A Block-Scaled GEMM**: Looping over multiple 64-wide $K$ atoms with a runtime index forces the register fragment $A$ into local memory, and reading shared-memory scale words three times per MMA pair wastes shared-memory bandwidth.

### Primary Source Matrix

| Upstream Commit / PR | Target Operator / Path | Microarchitectural Mechanism | Measured / Verified Gain |
|---|---|---|---|
| `flashinfer-ai/flashinfer` `#7415` (`182159eb7a`, 2026-10-26) | `csrc/fused_moe/warp_decode/cake_warp_decode_contract.cuh` | Moves 100 `sm_103a` rows to fused route-pack FC1 + K512 MMA-U2 FC2, 20 rows to packed fixed top-10 vector finalize, and 24 rows (`sm_103a` & `sm_100a`) to linear-grid K256 FC2 | Eliminates separate route-pack launch (`3 -> 2` launches, `1.50x` reduction) with bitwise-identical outputs |
| `flashinfer-ai/flashinfer` `#7436` (`4f14b25dc6`, 2026-10-26) | `flashinfer/experimental/cake_kda_train/cake_kcp.py` | Adds 2-CTA CP state walk records (64 value columns / $32,768\text{ B}$ per CTA, 2 CTAs/SM), selected per launch by chain-length fit rule | Up to `1.0754x` (B1 forward) and `1.0512x` (B3 fwd+replay+bwd) across 144 GB300 rows; 0 rows >2% slower |
| `flashinfer-ai/flashinfer` `#7419` (`328e862b34`, 2026-10-26) | `csrc/blackwell_bf16_fp4/` | Frees 2-CTA $M=128$ `tcgen05` TMEM while last stores are in flight; pre-loads/converts all E4M3 block scales before the main loop in 7 small-$M$ split-K programs | `1.099x` (GB300) / `1.085x` (B200) at $768/2112/2048$; `1.166x` (GB300) / `1.205x` (B200) at $16/1024/1024$ |
| `flashinfer-ai/flashinfer` `#7431` (`bdaa2401ad`, 2026-10-26) | `csrc/cake_grouped_mxfp8_quantize/` | Adds per-$K$ row-tiled programs ($K=4096$ 256-thread 2-row CTA, $K=2048$ 128-thread CTA) and integer UE8M0 scale extraction at $K=7168$ | `1.3552x` (GB300) / `1.3632x` (GB200) on $[32, 1024, 4096]$ with 16 valid rows/group |
| `flashinfer-ai/flashinfer` `#7407` (`fdd121f46b`, 2026-10-26) | `csrc/cake_selective_state_update/` | Mamba2 SSU 2/4-way P-Split along head dim (`2x–4x` CTAs), TMA-ring 1–2 token speculative decode, and persistent 2-head z-gated decode | Fills B200 (148 SMs) and GB300 (152 SMs) tail waves and removes proxy fences on short speculative steps |
| `flashinfer-ai/flashinfer` `#7417` (`a8d23b20bf`) & `#7434` (`9e656abf02`) | `cake_rmsnorm_train_backend/` & `cake_minimax_h3_nvfp4_pre_attention/` | Staged `dweight` slab reduction (8/16 trips/wave on `sm_107a`); faster MiniMax-H3 NVFP4 pre-attention norm+QKV GEMM | Up to `1.18x` on `sm_107a` (`1.07x` on `sm_103a`) for RMSNorm bwd; up to `1.1198x` on GB300 for MiniMax-H3 |
| `flashinfer-ai/flashinfer` `#6885` (`0af9594727`, 2026-10-26) | `csrc/cake_all_gather_matmul/` | Adds `producer_tail` in 7 CTA-pair programs so peer load warp waits on `mma_done` for the last `ring_depth` stages before CTA exit | Eliminates PTX §2.2.2 undefined async arrive on exited CTA SMEM (375 GB300 tests, 0 racecheck hazards) |
| `tile-ai/tilelang` `#3286` (`32f56122e2`) & `#3649` (`89b5b5b9ad`) | `tilelang/cuda/intrinsics/macro/mma_sm120_macro_generator.py` & `tilelang/jit/adapter/` | Unrolls $K$-atom loop (`T.unroll(explicit=True)`) and loads compact SFA/SFB scale packages once per atom on SM120; allocates `out_idx` on caller's launch stream | Eliminates local-memory spills and cuts shared scale loads per lane per atom by `6.0x` (`12 -> 2`); `0` cross-stream overwrites |

---

## 2. Implementation Mechanics, Data Layout & Synchronization

### 2.1 `cake_kda` Context-Parallel Two-CTA State Walks (PR `#7436`)

In packed-document KDA training ($D_k = D_v = 128$, chunk size $64$), each head maintains a $128 \times 128$ FP32 recurrent state ($65,536\text{ B}$). Because the recurrence updates value columns independently:
$$H_t[:, c] = \text{diag}(\alpha_t) H_{t-1}[:, c] + k_t \cdot v_{t, c}^{\top}, \quad c \in \{0..63\} \cup \{64..127\}$$
PR `#7436` splits each `(segment, head)` walk across two CTAs, each owning a 64-column half ($32,768\text{ B}$) and allowing **2 CTAs per SM** on GB300 (152 SMs). Tape walks (`kcp_fwdh_tape`, `kcp_bwd_tape`) always use the two-CTA record, while offset walks (`kcp_fwdh_offset`, `kcp_bwd_offset`) invoke `cake_kcp_jit.walk_form` to select two-CTA vs full-width per launch based on chain-length list scheduling, achieving up to **1.0754x** forward (B1) and **1.0512x** forward+replay+backward (B3) speedup at `CP=16`.

### 2.2 Blackwell `mm_bf16_fp4`, Grouped MXFP8 & CTA-Pair `producer_tail` (PRs `#7419`, `#7431`, `#6885`)

- **Early TMEM Deallocation & Pre-Loop Scale Conversion (`#7419`)**: Two-CTA $M=128$ `tcgen05` kernels free pair TMEM while the final TMA stores are still in flight rather than after a trailing cluster barrier (**1.099x** on GB300), and small-$M$ split-K kernels convert all E4M3 scales before the main loop (**1.205x** on B200 at $M=16, N=1024, K=1024$).
- **Integer UE8M0 Scale Extraction (`#7431`)**: Extracts the IEEE-754 biased exponent $\lceil \log_2(\max |x_i| / 448.0) \rceil + 127$ using integer bit operations and packed BF16/FP16 pairs, reaching **1.3552x** (GB300) and **1.3632x** (GB200) on $[32, 1024, 4096]$ with 16 rows/group.
- **CTA-Pair `producer_tail` (`#6885`)**: Because PTX ISA §9.7.18.6 does not order one `tcgen05.commit` after another, waiting on the epilogue accumulator barrier does not drain earlier multicast `mma_done` commits. Waiting for the last `ring_depth` stages in the peer load warp before CTA exit prevents undefined async arrives (PTX §2.2.2).

### 2.3 TileLang SM120 Register-A GEMM & Stream-Safe Allocation (PRs `#3286`, `#3649`)

Unrolling the $K$-atom loop (`T.unroll(explicit=True)`) gives every register fragment access a compile-time index, eliminating local-memory spills for $K \in \{192, 256\}$. Loading $\lceil \text{warp\_rows}/2 \rceil$ SFA words and $\lceil n_{8\text{-blocks}}/4 \rceil$ SFB words once per $K$ atom reduces shared-memory scale loads per lane per atom from $12$ to $2$ (**6.0x** reduction). PR `#3649` wraps `torch.empty` output allocations in `with torch.cuda.stream(stream):` when a non-default stream is passed to `cython`, `nvrtc`, or `cutedsl` adapters.

---

## 3. AI Infra Innovators Comparison (Fireworks / Together / Cerebras / Groq)

| Dimension | FlashInfer Cake + TileLang (Blackwell B200/GB300/SM120) | Fireworks AI (FireAttention / FP4-FP8 Stack) | Together AI (Turbo / ThunderKittens / FA4) | Groq LPU (Deterministic On-Chip SRAM Streaming) | Cerebras WSE-3 (Wafer-Scale Dataflow Engine) |
|---|---|---|---|---|---|
| **Low-Precision GEMM & Block Scaling** | Native `tcgen05` NVFP4 W4A4, BF16×FP4 W4A16 (early TMEM free), and integer UE8M0 grouped MXFP8 | Custom fused FP8/FP4 block-scaled MoE decode kernels minimizing launch gaps | Co-designed FA4 CuTe-DSL and register-tile primitives for high-throughput quantized GEMMs | Hardware mixed INT4/INT8/FP8/FP16 streaming MXM units with compiler-scheduled scale streams | Wafer-scale sparse/dense tensor cores with weights resident in on-chip SRAM |
| **Hybrid Linear Attention (KDA / Mamba2 SSU)** | 2-CTA 64-column CP state walks (2 CTAs/SM) + P-Split/TMA-Ring speculative decode | Unified SSM state and paged KV cache tiering for low-TTFT long-context serving | Chunked state replay and prefix alignment for hybrid Mamba-Transformer models | $230\text{ MB}$ on-chip SRAM holds recurrent state in place with zero external DRAM traffic | $44\text{ GB}$ on-chip SRAM holds full model weights and recurrent state without inter-node CP |
| **Multi-Stream / Multi-Core Synchronization** | Hardware `mbarrier`, multicast `tcgen05.commit`, `producer_tail` drain, and PDL | End-to-end CUDA Graph capture eliminating host launch overhead and stream races | Explicit CuTe barriers and static descriptors preventing register spills | Compiler-scheduled cycle-exact deterministic execution (zero runtime barriers) | On-wafer 2D mesh hardware fabric with self-synchronizing color micro-packets |

---

## 4. Architect Exercises & Falsifiable Follow-Up Questions

1. **Quantitative Exercise**: Verify that for a 32-head KDA layer at `CP=16` with $S_{\text{seg}}=2$ segments per rank on GB300 (152 SMs), splitting each `(segment, head)` walk from 1 CTA ($65,536\text{ B}$ state) to 2 CTAs ($32,768\text{ B}$ state each) doubles active CTAs from 64 to 128 and raises single-wave SM coverage from $42.1\%$ to $84.2\%$.
2. **Falsifiable Follow-Up**: At what expert-token skew threshold does fixed 2-row-per-CTA tiling in `cake_grouped_mxfp8_quantize` ($K=4096$) lose to a prefix-sum compacted worklist dispatch?
