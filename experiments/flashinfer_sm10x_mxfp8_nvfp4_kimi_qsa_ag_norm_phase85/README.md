# FlashInfer SM100/SM103/SM107 MXFP8 Gated-Act, NVFP4 Quant+GEMM, Kimi-K3 Router & AttnRes, SM107 QSA, AG+GEMM & Norm-Combine (Phase 85)

Deterministic Python 3.12 standard-library model reproducing the 7 FlashInfer upstream kernel advances audited in Phase 85 (`priority: 94`):

1. **`cake_gated_act_mxfp8` Narrow `32x32` / `32x64` Tiles & 4-Row/Thread Large-Shape Programs (`0b352a63`, PR #7346)**:
   - Models `silu_and_mul_mxfp8_quantize` and `silu_and_mul_mxfp8_quantize_backward` across `sm_100a` (B200, 148 SMs), `sm_103a` (GB300, 152 SMs), and `sm_107a` (R200, 208 SMs).
   - Verifies narrow `32x32` tile speedups at `128x128` (`1.260x` B200 `fwd colwise`, `1.121x` GB300 `fwd colwise`, `1.116x` R200 `bwd colwise`), the `sm_103a` `32x64` band ($2^{22}$ to $2^{24}$ elements), and 4-row/thread speedups at `131072x8192` (`1.020x` GB300 `fwd both`, `1.096x` R200 `fwd rowwise`).
2. **`cake_all_gather_matmul` SM103a Last-Pass N-Split Twin at `M=1025`, `ws=8` (`e7f5b1e1`, PR #7360)**:
   - Models the final peer pass splitting $P$ tiles into $2P$ half-width `128x128` tiles when $2P \le \text{num\_ctas}$ (`192` threads, `197,632 B` dynamic SMEM, `TMEM_NCOLS=512`), achieving `1.0116x` (`341.6 -> 336.6 us` node 1) and `1.0209x` (`340.7 -> 337.7 us` node 2) on 8x B300.
3. **`cake_mm_fp4` 4-CTA Row-Split NVFP4 Quantizer & `N16_SPLITK3_M` GEMM (`941fda25`, PR #7356)**:
   - Models the `_cs4` 4-CTA row-split quantizer (`1.255x` B200 at `K=28672, M=17`: `3.601 -> 2.874 us`; `1.243x` GB300 at `K=28672, M=1`: `3.235 -> 2.604 us`), `N16_SPLITK3_M` cluster occupancy guard (`148`/`152` SMs), and `1.144x` GEMM (`519.5 -> 454.5 us`) / `1.140x` fused (`540.6 -> 474.0 us`) on GB300.
4. **`cake_kimi_k3_fused_router` Regenerated 28-Cell Router Schedule (`ee50fd01`, PR #7355)**:
   - Models the 896-expert top-16 Kimi-K3 fused router across all 28 `(num_tokens, block_m)` cells on `sm_100a` (`1.06x–1.73x`) and `sm_103a` (`1.10x–1.80x`, `40.66 -> 22.61 us` at `tokens=8192, block_m=16`).
5. **`cake_qsa_nvfp4` SM107 (`cc=10.7`) Persistent Walk Within a 3-Wave Bound (`1cfd7312`, PR #7353)**:
   - Models the 15-bit SMEM descriptor mask `(((smem_q_addr) >> 4) & 0x7FFF)` under `__CUDA_ARCH__ == 1070`, the 3-wave bound `1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= 3` on R200 (`212` SMs), and `2` partial slots per `(row, KV head)`, delivering `1.17x–1.21x` speedups at `rows=128` (`55.71 -> 46.05 us` HND).
6. **`cake_warp_decode` 5-Deep-Ring K256 FC2 & Vectorized Top-10 Finalize (`ea95347a`, PR #7357)**:
   - Models `NUM_WORK_PIPE_STAGES: 3 -> 5`, the `512`-row (`8,192 B`) SMEM route-tile metadata table (`SMEM_TOTAL: 179,456 -> 187,648 B`), and the 4-feature/thread finalize kernel (`1.112x` on B200 and `1.104x` on GB300 at `M=4`).
7. **`cake_fused_norm_combine` 6-Program Dispatch & `cake_kimi_k3_attn_res` `_cta2` / `_crfnone` (`20ede145` PR #7352 & `e9261a5e` PR #7338)**:
   - Models the 6-program token-band dispatch (`3 CTAs/SM` on `593..1772` tokens, `1.088x` B200 / `1.092x` B300 at `tokens=1`) and `cake_kimi_k3_attn_res` 2-CTA-per-SM (`_cta2`) + fence-free (`_crfnone`) persistent programs (`181` programs / `211` libraries, up to `1.1297x` B200, `1.1219x` B300, `1.1097x` GB300).

## Run

```bash
python experiments/flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85/run.py --check
```
