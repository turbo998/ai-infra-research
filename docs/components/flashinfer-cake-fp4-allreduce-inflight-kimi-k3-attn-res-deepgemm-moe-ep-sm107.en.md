# Deep Dive: FlashInfer Cake FP4 AllReduce In-Flight Scale & Shared-Memory Value Reuse, Kimi-K3 Attention-Residual Round-2, Self-Contained Cake DeepGEMM, and SM107 MoE-EP

> **Research Phase**: Phase 47 (`priority: 56`) · **Question ID**: `q-phase47-flashinfer-cake-fp4-allreduce-inflight-kimi-k3-attn-res-deepgemm-moe-ep-sm107` · **Deterministic Script**: [`experiments/flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107/run.py`](../../experiments/flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107/run.py)

---

## 1. Executive Summary & First-Hand Upstream Commits

Across NVIDIA Blackwell (`sm_100a` B200 / `sm_103a` B300 / GB300) and `sm_107a` (GR100), six upstream commits in `flashinfer-ai/flashinfer` on 2026-10-21 advance collective communication-norm fusion, multi-candidate residual attention, self-contained Blackwell GEMM/attention libraries, and SM107 MoE-EP compilation:

| Commit | Date | Architectural Mechanism | Measured / Verified Impact |
|---|---|---|---|
| `flashinfer@c7d1eac` (`#6875`) | 2026-10-21 | `cake_fp4_allreduce` TP2 in-flight `os_rs` scale load + `j0==0` warp vote; TP4 pass-2 shared-memory value reuse | **`1.018x / 1.018x / 1.023x`** NVFP4 geomean on `sm_100a/103a/107a`; **`1.0384x`** peak (`18.15 -> 17.47 us`) on GB300 `1Kx6144` BF16; **`80%`** L2 value traffic cut in TP4 pass 2 |
| `flashinfer@3b65505` (`#6904`) | 2026-10-21 | `cake_kimi_k3_attn_res` Round-2 (`133` programs, `161` libraries: `75` `sm_100a` + `86` `sm_103a`) across `582` contract rows | **`1.0072x`** GB300 geomean (**`1.0888x`** at `serving_m5_k5`, `4.448 -> 4.096 us`); **`1.1034x`** peak on B200 `k_sweep_m1_k6` |
| `flashinfer@595bc57` (`#6928`) | 2026-10-21 | `cake_deepgemm_batched_projection_gemm` 6 variants (`bhr,hdr->bhd`, `K=4096, N=1024`, swapped-AB `M=16`) | **`9.727 us`** on GB300 at `T=2` swapped-AB `M=16` (**`8.78%`** faster than standard `M=64` `10.663 us`) |
| `flashinfer@597e3a6` (`#6927`) & `edfb3f7` (`#6902`) | 2026-10-21 | Self-contained `cake_deepgemm_fp8_gemm_1d1d` (FProp/WGrad) and `cake_nvfp4_attention` (3-stage persistent) | GB300 `4096x7168x4096` FProp/WGrad **`66.801 / 81.440 us`**; NVFP4 Attention `B=4,H=8,S=4096` **`114.161 us`** |
| `flashinfer@9d3837c` (`#6621`) | 2026-10-21 | Consolidates `flashinfer.moe_ep` and adds native `sm_107a` compilation for GR100 | Fixes `Mercury Validator: Validation failed for SM 107` caused by `sm_100f` lacking CUTLASS `Sm107` guards |

---

## 2. `cake_fp4_allreduce`: In-Flight Scale Overlap & Shared-Memory Value Reuse (`#6875`)

1. **Branchless FP4 Wire `-0.0` (`0x8`) Nibble Sentinel Detector**: Each 32-bit packed wire word holds eight E2M1 nibbles where `-0.0` (`0x8`) denotes an unwritten slot. The kernel tests all 8 nibbles simultaneously via:
   $$\text{pending} = \big(w \ \& \ \sim((w \ \& \ \text{0x77777777}) + \text{0x77777777})\big) \ \& \ \text{0x88888888}$$
2. **TP2 In-Flight Peer Scale Read**: Instead of spin-waiting on `os_rs + m_row` (`0x80000000u` sentinel) before entering the packet sweep, the kernel issues a single non-blocking `ld.volatile.global.u32` into `rsw[0]`, sweeps the first packet group (`j0 == 0`), and then resolves `rsw[0]` via `__any_sync(0xFFFFFFFF, rsw[0] == 2147483648u)` only if any lane still saw the sentinel.
3. **TP4 Two-Shot Pass-2 Shared-Memory Value Reuse**: Pass 1 already validates and stages all 4 ranks' 16-byte FP4 value packets (`64B` total) in shared memory `smem_u`. Pass 2 re-reads those validated values from `smem_u` and fetches only the four 4-byte scale words (`16B`) from the L2 Lamport buffer, cutting pass-2 L2 traffic by **`80.0%`** (`80B -> 16B` per step).

---

## 3. `cake_kimi_k3_attn_res` Round-2 Vendorable Package (`#6904`)

For hidden size $D = 7168$ and history depth $K \in [0, 8]$, `cake_kimi_k3_attn_res` fuses residual addition $u_i = \text{hidden}_i + \text{residual}_i$, optional `u_out` writeback, dual input RMSNorms ($w_r, w_u$), $K+1$ candidate dot-product scoring against query $q$, softmax $p_{0..K}$, convex combination $h_i = \sum_{k=0}^{K-1} p_{k,i} R_{k,i} + p_{K,i} u_i$, and optional output RMSNorm into a single kernel across `133` programs (`161` libraries: `75` `sm_100a` + `86` `sm_103a`) and `582` contract rows (`291` shapes $\times$ `2` PDL modes).

---

## 4. Self-Contained Cake DeepGEMM / NVFP4 Attention & SM107 MoE-EP (`#6928`, `#6927`, `#6902`, `#6621`)

- **Swapped-AB `M=16` Batched Projection GEMM (`#6928`)**: Swapping A and B operands for small batch/token dimensions (`T in {1, 2, 4}`) enables an `M=16` MMA tile for `128`-head `K=4096, N=1024` projections, achieving **`9.727 us`** at `T=2` on GB300 (`8.78%` faster than `M=64`).
- **Native `sm_107a` MoE-EP Compilation (`#6621`)**: Consolidated `flashinfer.moe_ep` adds explicit `sm_107a` code generation for GR100 (`SM 10.7`), preventing `sm_100f` fallback binaries from failing the driver's `Mercury Validator`.
