# FlashInfer MegaMoE EP16 `fence.proxy.alias`, MiniMax-H3 SM120 Out-Proj, Grouped FP8 Direct-Store, MM-FP4 Ring-Overlay, MoE AR Union, NVFP4 MLA QK-Ahead & TileLang Compiler Evolution (Phase 97)

> **中文版**: [flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97.md](flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97.md)
> **Priority**: `106` · **Question ID**: `flashinfer-megamoe-fence-minimax-sm120-grouped-fp8-mm-fp4-ar-union-mla-tilelang-phase97` · **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Across 10 zero-overlap commits in `flashinfer-ai/flashinfer` and `tile-ai/tilelang`, Phase 97 establishes key Blackwell (`sm_100a`/`sm_103a`/`sm_120a`) kernel and compiler invariants:

1. **Ordering Peer-Mapped VA Writes Against Local VA Reads via `fence.proxy.alias` (`flashinfer#7378`, `364b53a4bf`)**: In `CakeMxfp8MegaMoeEp16` (16 ranks, 512 experts = 32/rank, top-8, `H=3072, I=5120`), peer-mapped virtual writes (`remote_mc_va`) and local virtual reads alias the same physical HBM pages. Placing `fence.proxy.alias` immediately before `fence.acq_rel.sys` prior to cross-rank barrier arrival guarantees alias visibility before any peer consumes the dispatched rows.
2. **RTX PRO 6000 Blackwell (`sm_120a`, 188 SMs) Out-Projection Epilogue & Paired 16B Scale Loads (`flashinfer#7389`, `b986d13848`)**: `cake_minimax_h3_sm120_quant_out_proj` routes `sm_count == 188` to `PROGRAM_1_SMS = 188`, staging each MMA warp's epilogue slice in private SMEM and loading NVFP4 scale rows via one 16-byte SMEM load per stage pair (`3705.5 -> 3630.0 us`, `1.0208x` for FP8 at `M=33472`; `1.040x–1.046x` for NVFP4).
3. **Grouped FP8 GEMM Direct-Store & 256-Row 2-CTA Routing (`flashinfer#7386`, `63b0744f22`)**: Adds `deepk_c2_kg1_direct_store_m32`, `deepk_c2_ab6_scale1_direct_store_m32` (`M <= 32, CTAs <= 16`), and `deepk_c2_kg1_expert_aligned`, plus a 14-route layout-aware dispatcher enforcing 256-row alignment on 2-CTA routes.
4. **SM100 `cake_mm_fp4` Ring-Overlay `128x128` 1-CTA Tile (`flashinfer#7380`, `3508ca8255`)**: Overlays output staging onto the mainloop SMEM ring when `M <= 128` and `ceil(N / 128) > num_sms` (`224 > 148` on B200 for `N=28672`), expanding the mainloop pipeline from `5` to `6` stages (`1.019x–1.024x`).
5. **Interleaved Two-Shot Reduce-Scatter in `cake_trtllm_moe_allreduce_union` (`flashinfer#7387`, `5a052b767f`)**: Shortens first-token load chains and adds an interleaved two-shot reduce-scatter kernel for `WS=8, M in {64, 128}` (`H=7168`), accelerating all 30 benchmarked shapes (`1.02x–1.11x` on B200, `7.87 -> 7.49 us` at `WS=2, M=1`; `1.01x–1.12x` on B300).
6. **SM100 `cake_nvfp4_mla_decode` 2-Tile QK-Ahead & Split-KV Combine (`flashinfer#7385`, `aa3d76f9ba`)**: Launches `decode_qk_ahead` (issuing QK MMAs 2 KV tiles ahead of PV) when `unit_tiles >= 512`, reaching `1.039x` (64K) and `1.055x` (128K) on B200 and `1.016x–1.026x` on GB300 (`11.97 -> 11.78 us` at split-2).
7. **Removing Redundant Trailing Cluster Barriers in `mm_bf16_fp4` (`flashinfer#7376`, `de80f27971`)**: Eliminates end-of-kernel `barrier.cluster.arrive/wait` across 6 split-K `(1,1,4)` and `(1,1,8)` kernels, yielding `1.075x–1.078x` on `1x4096x4096` and `1.139x–1.140x` on `16x1024x1024` leader-mailbox.
8. **TileLang In-Register BF16x2 Conversion, Transitive Pipeline Copy Dependencies & Batched Reducer Lowering (`tilelang#3595/#2822/#3070`)**: Replaces `__bfloat1622float2` with `shl.b32 %0, %2, 16; and.b32 %1, %2, 0xffff0000;`, preserves transitive scalar/buffer pipeline copy waits in a single backward worklist, and fixes batched `tl.finalize_reducer` lowering via linear offsets (`b * layout_batch_size`).

---

## 2. Primary Source Evidence Matrix

| # | Source ID / Commit | Mechanism & Hardware Contract | Quantitative Result / Invariant |
| :--- | :--- | :--- | :--- |
| 1 | `flashinfer-cake-megamoe-ep16-alias-proxy-fence-pr7378-2026` (`364b53a4bf`) | `fence.proxy.alias` immediately before `fence.acq_rel.sys` on EP16 sync path | Orders peer-mapped VA writes before local VA reads across 16 ranks / 512 experts |
| 2 | `flashinfer-cake-minimax-h3-sm120-outproj-rtxpro6000-pr7389-2026` (`b986d13848`) | `PROGRAM_1_SMS=188` warp-private epilogue + 16B paired scale row loads | FP8 `M=33472`: `3705.5 -> 3630.0 us` (`1.0208x`); NVFP4 up to `1.046x` |
| 3 | `flashinfer-cake-grouped-fp8-gemm-direct-store-fp32-scale-pr7390-2026` (`63b0744f22`) | 3 FP32-scale direct-store/expert-aligned kernels + 14-route dispatcher | Direct store for `M<=32, CTAs<=16`; 256-row alignment check for 2-CTA routes |
| 4 | `flashinfer-cake-mm-fp4-ring-overlay-128wide-sm100-pr7380-2026` (`3508ca8255`) | `128x128` 1-CTA output staging overlaid onto mainloop SMEM ring | Expands pipeline `5 -> 6` stages; `1.019x–1.024x` on `K=8192, N=28672, M=128` |
| 5 | `flashinfer-cake-trtllm-moe-allreduce-union-interleaved-2shot-pr7384-2026` (`5a052b767f`) | Shortened first-token load chain + interleaved 2-shot RS (`H=7168`) | `1.02x–1.11x` on B200 (`7.87 -> 7.49 us`); `1.01x–1.12x` on B300 |
| 6 | `flashinfer-cake-nvfp4-mla-decode-qkahead-split-kv-pr7385-2026` (`aa3d76f9ba`) | `decode_qk_ahead` (`unit_tiles >= 512`) + split-KV combine | `1.039x` (64K) and `1.055x` (128K) on B200; `1.026x` on GB300 |
| 7 | `flashinfer-cake-blackwell-bf16-fp4-splitk-cluster-pr7376-2026` (`de80f27971`) | Removes redundant trailing `barrier.cluster.arrive/wait` | `1.078x` on `1/4096/4096`; `1.140x` on `16/1024/1024` leader-mailbox |
| 8 | `tilelang-cuda-vectorized-bf16-to-fp32-conversion-pr3595-2026` (`1a52a1cd06`) | `__tl_cvt_bf16x2_to_f32x2` via `shl.b32` + `and.b32` | Unpacks `bf16x2 -> float2` in-register without permutation |
| 9 | `tilelang-preserve-transitive-pipeline-copy-deps-pr2822-2026` (`4b2032a6ce`) | Unified backward worklist + materialized thread-domain region conflict | Zero dropped async copy waits across transitive stages |
| 10 | `tilelang-fix-batched-finalize-reducer-lowering-pr3070-2026` (`6db1fc771a`) | `layout_batch_size > 0` check + flattened linear offset per batch | Exact per-batch `tl.finalize_reducer` lowering |

---

## 3. Reproducible Offline Verification

- **Script**: `experiments/flashinfer_megamoe_fence_minimax_sm120_grouped_fp8_mm_fp4_ar_union_mla_tilelang_phase97/run.py`
- **Command**: `python experiments\flashinfer_megamoe_fence_minimax_sm120_grouped_fp8_mm_fp4_ar_union_mla_tilelang_phase97\run.py --check`
- **Unit Test**: `tests/test_phase97_experiments.py`
