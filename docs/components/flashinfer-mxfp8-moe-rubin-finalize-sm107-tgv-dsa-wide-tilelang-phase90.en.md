# FlashInfer MXFP8 CuTe-DSL Fused MoE, Rubin Unfused Deterministic Finalize, SM107 Deeper TGV Pipelines, Cake DSA Wide-First Indexer, Kimi-K3 TP12 Tail, 5-CTA MoE Finalize AllReduce, Blackwell Cached-Cluster Softmax & TileLang SM8x/Reduce Deep Dive (Phase 90)

> **中文版本**: [flashinfer-mxfp8-moe-rubin-finalize-sm107-tgv-dsa-wide-tilelang-phase90.md](flashinfer-mxfp8-moe-rubin-finalize-sm107-tgv-dsa-wide-tilelang-phase90.md)  
> **Evidence Tier**: `implementation` (first-hand commit inspection) + `own-offline` (deterministic Python 3.12 stdlib model in `experiments/flashinfer_mxfp8_moe_rubin_finalize_sm107_tgv_dsa_wide_tilelang_phase90/run.py`). No live GPU/RDMA hardware benchmarks are claimed.

## 1. First-Hand Code Evidence & Implementation Paths

1. **CuTe-DSL Fused MoE `MXFP8 x MXFP8` Mode for Nemotron** (`flashinfer-ai/flashinfer` PR `#6793`, commit `b92f112955`):
   - Adds `quant_mode="mxfp8"` (E4M3 weights and activations with 1-byte `E8M0` block-32 scales) to `cute_dsl_fused_moe` on SM100/SM103 by reusing the `W4A8` kernel family with an FP8 B operand and wiring `QuantConfig(MXFP8, MXFP8)` into the unified MoE API.
   - **Bandwidth Formula**: Effective bytes per element $B_{\text{eff}} = 1 + \frac{1}{32} = 1.03125\text{ B}$, yielding a $\frac{2.0}{1.03125} \approx 1.9394\times$ bandwidth reduction vs. BF16 ($2.0\text{ B}$) with $7168 / 32 = 224$ scale blocks per row at $K=7168$.
2. **Rubin (SM107) Unfused Deterministic Finalize for CuTe-DSL Fused MoE** (`flashinfer-ai/flashinfer` PR `#6950`, commit `5240384d8f`):
   - Previously, SM107 only implemented the fused `atomicAdd` scatter-add finalize, causing `use_fused_finalize=False` (the SGLang default and mandatory `--enable-deterministic-inference` path) to raise `NotImplementedError`. This commit implements the unfused contiguous output + ordered reduction path on SM107.
3. **SM107 TGV GEMM Tactics (`29..46`) with Deeper A/B Pipelines** (`flashinfer-ai/flashinfer` PR `#7049`, commit `f2ebfdc8c4`):
   - Appends 18 SM107-specific TGV tactics (IDs `29..46`) exploiting SM107's larger shared memory ($320\text{ KiB}$ vs. $228\text{ KiB}$ in our model) to deepen A/B pipeline stages from 4 to 6 ($1.5\times$ pipeline depth).
4. **Cake DSA Indexer `sm_103a` Wide-First Dispatch (`>= 512` Mean Unit Tiles)** (`flashinfer-ai/flashinfer` PR `#7406`, commit `35cba307a0`):
   - Updates the host plan on `sm_103a` (GB300) so calls with $\ge 512$ mean key tiles per work unit dispatch the unrolled 8-query wide scan program ahead of the 3-warp narrow program.
5. **Cake Kimi-K3 TP12 Tail Four-Regime Dispatch & Persistent K3** (`flashinfer-ai/flashinfer` PR `#7402`, commit `8707500798`):
   - Dispatches 12-rank NVL72 BF16 LatentMoE tail kernels by token count $M$: one-shot launch for $M \le 8$, one-shot ESS K1 + K3-ESS for $M \in [9, 16]$, two-shot ESS K1 + K3-ESS for $M \in [17, 960]$, and two-shot persistent K3 for $M > 960$.
6. **Cake MoE Finalize + One-Shot Lamport AllReduce + Ordered RMS Exchange (`5 CTAs/SM`)** (`flashinfer-ai/flashinfer` PR `#7392`, commit `ee376e9737`):
   - Fuses Top-K expert reduction, optional shared expert, one-shot Lamport AllReduce, residual add, and ordered RMSNorm at $H=7168$ across 32 specialized programs with `5 CTAs/SM` residency.
7. **Cake Blackwell Cached-Cluster Softmax Residency Cap & PDL Twins** (`flashinfer-ai/flashinfer` PR `#7404`, commit `01a04018fc`):
   - Caps cluster spread to active SM residency and adds `sm_103a` PDL kernel twins across 14 Blackwell FP32 row-major softmax programs (7 row-caching cluster + 2 shared-tile programs).
8. **TileLang SM8x Sparse Metadata Guard, DeepSelect Top-K & Sub-Byte/Non-Finite Reduce Fix** (`tile-ai/tilelang` PRs `#3335` `80364b3d97`, `#3304` `f8053068c7`, `#3189` `1a0830a50d`):
   - Explicitly rejects unsupported operand dtypes (e.g., `float8_e4m3fn`) in `make_cutlass_metadata_layout_sm8x`; adds deterministic `DeepSelect` Top-K; clamps non-finite `fp8_e4m3`/`float4` reduction initializers to max-finite ($448.0$ / $6.0$) instead of emitting invalid `inff`, and rejects $<8\text{-bit}$ accumulators.

## 2. Architect Exercises & Falsifiable Hypotheses

- **Sizing Exercise**: For $M=512, \text{top\_k}=8, K=7168$, the Rubin SM107 unfused finalize BF16 intermediate buffer requires $512 \times 8 \times 7168 \times 2 = 56\text{ MiB}$.
- **Falsifiable Hypothesis**: Dispatching the 8-query wide scan on `sm_103a` when mean unit tiles $< 512$ degrades tail latency due to register pressure and under-utilized query lanes.
