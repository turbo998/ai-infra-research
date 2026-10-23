# Deep Dive: FlashInfer SM120 FP8 256/256 MLA & GQA Prefill, Hopper MXFP4×FP8 K128 Folded MegaMoE, Native NoPE Sparse MLA Training, Cake Ulysses Interleaved-Stripe NVLink All-to-All, and TileLang Top-K / 2:4 Sparse Bijection Guards

> **Research Question ID**: `q-flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang` (Priority 69, Phase 60 Track A)  
> **Upstream Commits Covered**:
> - `flashinfer-ai/flashinfer@8e8342f909d4422c67d3e83b94aa8a5e94a68824` (`#6466`, 2026-10-23)
> - `flashinfer-ai/flashinfer@b17eaa011241866e45a3ea06797c5fecaf6c7bc7` (`#6465`, 2026-10-23)
> - `flashinfer-ai/flashinfer@592deecaff131b07bff5d53c4f61c8f95621f65a` (`#6929`, 2026-10-22)
> - `flashinfer-ai/flashinfer@49aa24fbc4b30a2ba0be61062342f891e8af2972` (`#6887`, 2026-10-20)
> - `flashinfer-ai/flashinfer@6a4e51eb81a108c0d44e32d74c7713da6577d92c` (`#6489/#6836`, 2026-10-20)
> - `flashinfer-ai/flashinfer@60b1652f903bc6169b7ac1a4c9968e42e58cd9b2` (`#6864`, 2026-10-20)
> - `tile-ai/tilelang@4da574c01371a79825c91492f09be6a83a4cb400` (`#3280`, 2026-10-22)
> - `tile-ai/tilelang@ced0ca1a30356d5834fe38b1d4257a09773e467a` (`#3125`, 2026-10-22)
> - `tile-ai/tilelang@1a52a1cd0628f59cbc45c5d29505fab0e9f08b92` (`#3595`, 2026-10-21)  
> **Companion Deterministic Experiment**: `experiments/flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang/run.py`

---

## 1. Architectural Motivation & Problem Statement

Across Blackwell workstation/edge GPUs (`SM120`/`SM121`), Hopper (`SM90a`), and Blackwell/Rubin datacenter GPUs (`SM100a`/`SM103a`/`SM107a`), five kernel-level bottlenecks govern long-context prefill, MoE routing, and sequence-parallel training efficiency:

1. **Missing SM120 FP8 Prefill Attention Coverage for 256/256 MLA and GQA**: On SM120 (e.g., 5K Pro GPU with 110 SMs and a 99 KiB per-SM shared memory ceiling), the FMHA v2 warp-specialized FP8 E4M3-to-BF16 kernel previously supported only DeepSeek-V3.2's `192/128` MHA geometry. It lacked both the 4-K-step `256/256` pipeline required by **GLM-4.7-Flash** (`20/20` heads, 192 NoPE + 64 RoPE) and the separate KV head stride indexing required by **GLM-4.7** GQA (`128/128`, 96 Q heads / 8 KV heads, $G=12$).
2. **Hopper MXFP4 × FP8 MegaMoE Candidate Starvation on Non-256-Divisible Intermediate Sizes**: When a fine-grained MoE model uses intermediate size $I = 640$ (`gate_up = 1280`), $640 \bmod 256 = 128 \neq 0$ disqualifies every `K256` folded tuning candidate in `can_implement`, leaving only **3** out of 11 candidates eligible and causing severe wave quantization at medium-to-high token counts.
3. **Redundant RoPE Compute and HBM Traffic in Zero-RoPE (NoPE) Sparse MLA Training**: **GLM-5.3-Flash**'s DeepSeek Sparse Attention (DSA) drops the 64-dim RoPE suffix and uses a pure 512-dim shared latent vector across 64 heads. Zero-padding into a 576-dim RoPE kernel wastes $64/576 = 11.11\%$ of HBM bandwidth and executes unnecessary RoPE backward reductions.
4. **NVLink Contention in Sequence-Parallel All-to-All**: Naive contiguous peer writes during Ulysses `scatter_heads` / `gather_heads` create remote HBM channel hotspots across 2/4/6/8 NVLink ranks.
5. **TileLang Fixed-SMEM Top-K Overflow and Non-Bijective 2:4 Sparse Metadata Layouts**: DeepSeek-V3.2's top-k selector hardcoded `SMEM_INPUT_SIZE = 4096` for radix bucket candidates, overflowing shared memory when `seq_len = 32768` contains $>4096$ tied scores; meanwhile, `make_metadata_layout` on SM8x 2:4 sparse GEMM produced non-bijective index maps whenever $M \bmod \text{group} \neq 0$ or $K \bmod 2 \neq 0$.

---

## 2. Core Implementation Mechanics & Source-Level Analysis

```
+--------------------------------------------------------------------------------------------------+
| Phase 60 Track A: GPU Kernel & Quantization Pipeline Architecture                                |
+--------------------------------------------------------------------------------------------------+
| 1. SM120 FP8 E4M3->BF16 FMHA v2 (#6466)                                                          |
|    - GLM-4.7-Flash (256/256, 20/20 MHA): 233 regs/thread, 64 KiB SMEM, 1 CTA/SM -> 1.53x-1.61x   |
|    - GLM-4.7 (128/128, 96/8 GQA): head_idx = (qkv_offset==0) ? bidh : (bidh / h_q_per_kv)        |
|                                                                                                  |
| 2. Hopper MXFP4 x FP8 MegaMoE Folded K128 Candidates (#6465)                                     |
|    - I=640 (640 % 256 != 0) filters out K256; adds M256/N64/K128 (clusters 1x1x1 & 1x2x1)        |
|      + M256/N16/K128 -> eligible candidates 3 -> 8; -17.72% compute / -14.93% E2E at 1024 tok    |
|                                                                                                  |
| 3. Blackwell/Rubin Native NoPE DSA Training (#6836) & Cake Ulysses A2A (#6864)                   |
|    - cake_dsa_h64_nope_train (512 latent, dkv = dK + dV in FP32 [S,512]): 1.74x fwd on GB300     |
|    - Interleaved-stripe NVLink scatter/gather (24 variants): 1.33x-2.50x on B200/B300            |
|                                                                                                  |
| 4. TileLang Guardrails (#3280, #3125, #3595)                                                     |
|    - Dynamic global workspace [B, 2, seq_len] replaces fixed 4096 SMEM buffer for >4K ties       |
|    - sm8x 2:4 sparse metadata enforces M % group == 0 (32/16) & K % 2 == 0 for bijection         |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 SM120 FP8 256/256 MLA and GQA Prefill Attention (`flashinfer@8e8342f909`)

In `csrc/fmha_v2/fmha/smem_tile_o.h` and `fused_multihead_attention_kernel_1xN_reload.h`, FlashInfer adds two architectural extensions for SM120:

1. **GLM-4.7-Flash `256/256` Warp-Specialized Pipeline**:
   - Both QK and V head dimensions are `256` (`4` MMA K-steps of `64`). To fit within SM120's 99 KiB shared memory limit, dynamic SMEM is bounded at **64 KiB** (1 CTA/SM) and register usage is **233 regs/thread** (whereas the matched BF16 reload kernel uses **247 regs/thread** with 64B stack spill).
   - In `Smem_tile_o_row_major`, `BYTES_PER_STG == 16` packs 8 BF16 elements per thread across 32 quad pairs for `HEAD_DIM_V = 256`, issuing wide 16B stores without shared-memory bank conflicts.
2. **GLM-4.7 GQA Head Stride Indexing**:
   - `Fused_multihead_attention_params_v2` adds `h_kv` and `h_q_per_kv = h / h_kv`.
   - `Hopper_qkv_tile_params` selects the head index via:
     $$\text{head\_idx} = \begin{cases} \text{bidh}, & \text{if } \text{qkv\_offset} = 0 \text{ (Query)} \\ \lfloor \text{bidh} / h_{\text{q\_per\_kv}} \rfloor, & \text{if } \text{qkv\_offset} \in \{1, 2\} \text{ (Key / Value)} \end{cases}$$

### 2.2 Hopper MXFP4 × FP8 MegaMoE Folded K128 Candidates (`flashinfer@b17eaa0112`)

In `flashinfer/moe_ep/hopper_mxfp4_megamoe.py`, folded `M256/N64` tiles were previously paired only with `K256`. Because `M256/N64/K128` folds to a valid 4-stage `M128/N128/K128` Hopper WGMMA mainloop requiring only $I \bmod 128 = 0$, this commit adds 5 new `K128` tuning candidates:
- Three `M256/N64/K128` single-CTA cluster `(1,1,1)` candidates with `group_hint` `132`, `264`, and `330`;
- One `M256/N64/K128` two-CTA cluster `(1,2,1)` tail-pair candidate;
- One `M256/N16/K128` fine-N candidate.

For $I = 640$, eligible tuning candidates increase from **3 to 8**.

### 2.3 Native NoPE Sparse MLA Training (`flashinfer@6a4e51eb81`) & Ulysses Interleaved-Stripe A2A (`flashinfer@60b1652f90`)

- **Native NoPE Sparse MLA Training (`cake_dsa_h64_nope_train`)**: Targets GLM-5.3-Flash (64 heads, 512 shared latent dim, 0 RoPE dim) on `sm_100a` (GB200), `sm_103a` (GB300), and `sm_107a` (R200). Because $K$ and $V$ alias the same `[S, 512]` latent tensor, backward passes accumulate a single combined FP32 gradient $\text{dkv} = dK + dV \in \mathbb{R}^{S \times 512}$, eliminating the 64-dim RoPE `tma_red_f32` path.
- **Cake Ulysses Interleaved-Stripe NVLink All-to-All (`#6864`)**: Delivers all 24 variants (`bf16/fp16/fp32` × `scatter/gather` × `2/4/6/8 ranks`) on `sm_100a`/`sm_103a`, achieving **1.33x–2.49x** speedup on 8×B200 and **1.29x–2.50x** on B300.
- **Cake BF16 Fused QK RMSNorm + NeoX RoPE + Paged KV Append (`#6887`)**: Unifies SM90/SM100/SM103 under a single kernel launch, achieving 28-shape geomean speedups of **1.0623x** on H100, **1.0738x** on H200, and **1.14x** on B200/B300/GB300.

### 2.4 TileLang Top-K Overflow Fix, 2:4 Sparse Bijection Guard, and Shift-And BF16x2 Conversion (`tilelang@4da574c013`, `ced0ca1a30`, `1a52a1cd06`)

1. **DeepSeek-V3.2 Top-K Selector Dynamic Workspace (`#3280`)**: Replaces the hardcoded shared-memory `s_input_idx[2, 4096]` candidate queue with a global tensor `workspace[batch, 2, seq_len]`, preventing shared-memory corruption when $>4096$ elements share the threshold radix bucket (e.g., at `seq_len = 32768`).
2. **SM8x 2:4 Sparse Metadata Bijection Guard (`#3125`)**: Enforces `ICHECK_EQ(m % group, 0)` (`group = 32` for 16-bit, `16` for 8-bit) and `ICHECK_EQ(k % 2, 0)` in `make_metadata_layout`, guaranteeing that the 2:4 sparse metadata swizzle is strictly bijective.
3. **Permutation-Free `bf16x2 -> f32x2` (`#3595`)**: Replaces byte-permutation intrinsics in `tl::__tl_cvt_bf16x2_to_f32x2` with `in_bits << 16` (`shl.b32`) and `in_bits & 0xffff0000u` (`and.b32`).

---

## 3. Quantitative Evaluation & Benchmark Comparison

Verified via `experiments/flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang/run.py`:

### Table 1: SM120 (5K Pro GPU, 110 SMs) GLM-4.7-Flash `256/256` FP8 Prefill Latency (`#6466`)

| Heads ($H_q/H_{kv}$) | Seq Len ($S$) | FP8 FMHA v2 ($\mu\text{s}$) | Matched BF16 FMHA v2 ($\mu\text{s}$) | BF16 FA2 ($\mu\text{s}$) | Speedup vs Matched BF16 | Speedup vs BF16 FA2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `20 / 20` (1 Node) | 1024 | **68.88** | 105.33 | 103.54 | **1.53x** | **1.50x** |
| `20 / 20` (1 Node) | 4096 | **782.40** | 1241.56 | 1110.74 | **1.59x** | **1.42x** |
| `20 / 20` (1 Node) | 8192 | **2975.75** | 4690.40 | 4077.53 | **1.58x** | **1.37x** |
| `5 / 5` (TP=4) | 1024 | **30.35** | 47.07 | 43.59 | **1.55x** | **1.44x** |
| `5 / 5` (TP=4) | 4096 | **243.48** | 387.44 | 377.05 | **1.59x** | **1.55x** |
| `5 / 5` (TP=4) | 8192 | **803.80** | 1290.77 | 1133.70 | **1.61x** | **1.41x** |

### Table 2: 2×H200 (`EP=2, E=512, topk=10, H=2560, I=640`) MXFP4×FP8 MegaMoE K128 Tuning (`#6465`)

| Tokens / Rank | Orig Compute ($\mu\text{s}$) | PR Compute ($\mu\text{s}$) | Compute Reduction | Orig E2E ($\mu\text{s}$) | PR E2E ($\mu\text{s}$) | E2E Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 128 | 339.216 | **316.272** | **-6.76%** | 501.264 | **475.856** | **-5.07%** |
| 1024 | 593.264 | **488.128** | **-17.72%** | 756.176 | **643.296** | **-14.93%** |
| 4096 | 1405.728 | **1315.200** | **-6.44%** | 1573.472 | **1475.216** | **-6.24%** |
| 8192 | 2670.336 | **2343.024** | **-12.26%** | 2891.616 | **2560.176** | **-11.46%** |
| 16384 | 5223.024 | **4442.880** | **-14.94%** | 5628.496 | **4873.776** | **-13.41%** |

### Table 3: GB300 (`sm_103a`) & R200 (`sm_107a`) Native NoPE Sparse MLA Training vs Zero-Padded RoPE (`#6836`)

| Architecture | Workload ($S \times S_{kv}, K$) | Forward NoPE (ms) | Forward RoPE (ms) | Fwd Speedup | Step NoPE (ms) | Step RoPE (ms) | Step Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GB300 (`sm_103a`)** | `doc_4096` ($K=2176$) | **0.716** | 1.246 | **1.74x** | **3.707** | 5.150 | **1.39x** |
| **GB300 (`sm_103a`)** | `doc_16384` ($K=128$) | **0.655** | 1.201 | **1.83x** | **2.443** | 4.774 | **1.95x** |
| **GB300 (`sm_103a`)** | `short_1024_k128` ($K=128$) | **0.049** | 0.558 | **11.43x** | **0.323** | 0.878 | **2.72x** |
| **GB300 (`sm_103a`)** | `glm53_a` ($16263 \times 269564, K=2176$) | **2.836** | 3.801 | **1.34x** | **15.674** | 20.181 | **1.29x** |
| **R200 (`sm_107a`)** | `doc_16384` ($K=128$) | **0.387** | 0.602 | **1.56x** | **1.508** | 3.515 | **2.33x** |
| **R200 (`sm_107a`)** | `glm53_a` ($16263 \times 269564, K=2176$) | **1.706** | 2.369 | **1.39x** | **9.109** | 13.923 | **1.53x** |

---

## 4. Architect Design Takeaways & Falsifiable Questions

1. **SM120 99 KiB SMEM Ceiling**: When deploying GLM-4.7-Flash (`256/256` head dim) on SM120 workstation GPUs, BF16 reload kernels hit 247 regs/thread with 64B stack spill; switching to FP8 E4M3-to-BF16 FMHA v2 keeps dynamic SMEM at 64 KiB and registers at 233 regs/thread, delivering **1.53x–1.61x** prefill speedups.
2. **MoE Intermediate-Size Divisibility Audit**: Fine-grained MoE architectures with $I \bmod 256 \neq 0$ (such as $I=640$) require explicit `K128` folded WGMMA tiles (`M256/N64/K128`) and `(1,2,1)` cluster tail-pairing to avoid losing 15%–18% compute throughput at `tokens_per_rank >= 1024`.
3. **Falsifiable Follow-Up Question**: On Rubin `sm_107a`, when native NoPE sparse MLA training merges `dK + dV` into a single `[S, 512]` FP32 `tma_red_f32` atomic reduction, at what sparsity ratio $K/S$ does L2 atomic reduction throughput overtake Tensor Core MMA as the primary backward bottleneck?
