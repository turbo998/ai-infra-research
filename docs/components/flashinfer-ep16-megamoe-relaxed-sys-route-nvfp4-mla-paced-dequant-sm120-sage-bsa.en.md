# Deep Dive: FlashInfer EP16 MegaMoE Relaxed-Sys Sentinel Return, NVFP4 MLA Paced Dequant, and SM120 Block-Sparse Sage Attention

> **Status**: Phase 80 (`priority: 89`) Primary-Source Code Audit  
> **Bilingual Pair**: [`flashinfer-ep16-megamoe-relaxed-sys-route-nvfp4-mla-paced-dequant-sm120-sage-bsa.md`](flashinfer-ep16-megamoe-relaxed-sys-route-nvfp4-mla-paced-dequant-sm120-sage-bsa.md)  
> **Reproducible Experiment**: [`experiments/flashinfer_mxfp8_megamoe_ep16_relaxed_route_nvfp4_mla_paced_dequant_sage_sm120/run.py`](../../experiments/flashinfer_mxfp8_megamoe_ep16_relaxed_route_nvfp4_mla_paced_dequant_sage_sm120/run.py)

---

## 1. Primary-Source Evidence Matrix

| Evidence ID | Upstream Commit | Key Module & Change | Category |
| --- | --- | --- | --- |
| `src-flashinfer-mxfp8-megamoe-ep16-relaxed-route-2026` | `flashinfer-ai/flashinfer@0604b07a46e1` (`#7323`) | `CakeMxfp8MegaMoeEp16`: direct `st.relaxed.sys` route-term return with `0xFFFFFFFFu` sentinel, split rank barrier, deferred FC1 prefetch, `evict_first` weights | `implementation` + `author-benchmark` |
| `src-flashinfer-nvfp4-mla-decode-paced-dequant-2026` | `flashinfer-ai/flashinfer@0cdbe67b8c46` (`#7343`) | `cake_nvfp4_mla_decode`: `nanosleep.u32 200` dequant warp pacing, `sm_103a` elision of `fence.proxy.async.shared::cta`, loader cleanup | `implementation` + `author-benchmark` |
| `src-flashinfer-nvfp4-mla-buildjson-curdevice-2026` | `flashinfer-ai/flashinfer@9bad274a7d86` (`#7327`) | Package `build.json` in wheel and query `torch.cuda.current_device()` for workspace sizing | `implementation` |
| `src-flashinfer-kimi-k3-tp12-tail-k23-doc-2026` | `flashinfer-ai/flashinfer@a47e1701a0e0` (`#7316`) | `cake_kimi_k3_tp12_tail`: document `K23_MAX_TOKENS = 8` fused tail cutoff (`M <= 8` vs `9 <= M <= 960`) | `implementation` |
| `src-flashinfer-sm120-sage-block-sparse-attn-2026` | `flashinfer-ai/flashinfer@c4b6ec9e36b2` (`#7303`) | SM120 (`sm_120a`) `bsa_attn_sm120_blk64_sage_fwd`: INT8 QK + FP8 PV 64-token block-sparse attention with 20 specializations | `implementation` + `author-benchmark` |

---

## 2. `CakeMxfp8MegaMoeEp16`: Why a `0xFFFFFFFFu` Sentinel Eliminates a System Fence and NVLink Barrier

In `flashinfer-ai/flashinfer@0604b07a46e1` (`#7323`), the 16-rank expert-parallel MegaMoE kernel (`hidden=3072, intermediate=5120, 512 experts = 32/rank, top_k=8`, BF16 activations, MXFP8 E4M3+E8M0 block-32 weights on `sm_100a`/`sm_103a`) lands three coordinated optimizations:

1. **Direct `st.relaxed.sys` Route-Term Return + `0xFFFFFFFFu` Sentinel Polling**:
   - Before the publication barrier, each rank initializes its incoming route-term buffer words to the all-ones sentinel `0xFFFFFFFFu`.
   - Each 32-bit word packs two IEEE-754 BF16 values. Because BF16 `-0.0` is `0x8000` (`0x80008000 != 0xFFFFFFFF`) and canonical quiet NaN is `0x7FFF` (`0x7FFF7FFF != 0xFFFFFFFF`), no finite BF16 pair or canonical NaN pair collides with `0xFFFFFFFFu`.
   - The FC2 epilogue writes each route term directly to the source rank via `st.relaxed.sys`. The source rank's `cake_megamoe_wait_route_terms` spins with `ld.relaxed.sys.global.v4.u32` until none of the 8 `uint4` words equals `0xFFFFFFFFu`, reduces the `top-8` expert outputs, and re-arms the `0xFFFFFFFFu` sentinel. This eliminates the end-of-kernel system fence (`fence.acq_rel.sys`), the return grid counter, and one NVLink barrier phase (`3 -> 2` phases).
2. **Deferred First FC1 Prefetch**: Moves `cp.async.bulk.prefetch.tensor` for the first FC1 work item after the publication grid barrier so L2 prefetch traffic does not delay cross-rank publication.
3. **L2 `evict_first` Weight Hints**: Tags streaming FC1 and FC2 weight TMA loads with `evict_first`.

### Measured Latency on 16× NVIDIA GB300 (`sm_103a`, NVL72 4×4, CUDA 13.4, Cold L2)

| Tokens / Rank (Global) | Routing | Previous ($\mu\text{s}$) | New ($\mu\text{s}$) | Speedup | Saved ($\mu\text{s}$) |
| --- | --- | --- | --- | --- | --- |
| `16` (`256`) | `balanced` / `hotset50` | `313.3` / `313.3` | `299.0` / `299.1` | **1.048x** / **1.047x** | `-14.3` / `-14.2` |
| `32` (`512`) | `balanced` / `hotset50` | `316.4` / `315.9` | `301.0` / `300.9` | **1.051x** / **1.050x** | `-15.4` / `-15.0` |
| `64` (`1024`) | `balanced` / `hotset50` | `317.4` / `318.0` | `303.1` / `303.9` | **1.047x** / **1.046x** | `-14.3` / `-14.1` |

---

## 3. `cake_nvfp4_mla_decode`: Dequant Warp `nanosleep` Pacing & `sm_103a` Proxy-Fence Elision

In `flashinfer-ai/flashinfer@0cdbe67b8c46` (`#7343`), the DeepSeek-V4 NVFP4 MLA decode kernel paces the dequantization transform warp with `asm volatile("nanosleep.u32 %0;" :: "r"(200));` right after `mbarrier_wait` on `raw_full_addr` and `v_empty_addr`, elides the redundant `fence.proxy.async.shared::cta` when `__CUDA_ARCH__ == 1030` (GB300), and removes an unused `first_page_l` load from the loader start chain. Combined with `#7327` (`current_device()` workspace query and `build.json` wheel packaging), this delivers **1.012x–1.056x** speedup on GB300 (`1125.70 -> 1066.37 us` at `batch 32, 128K KV`) and **1.012x–1.032x** on B200.

---

## 4. SM120 (`sm_120a`) Block-Sparse Sage Attention & Kimi-K3 TP12 Tail Cutoff

- **SM120 Block-Sparse Sage Attention (`#7303`)**: Implements `bsa_attn_sm120_blk64_sage_fwd(..., backend="cake")` on RTX PRO 6000 Blackwell (`sm_120a`, 188 SMs) using INT8 $Q K^T$ on `mma.sync` and FP8 E4M3 $P V$ across 64-token blocks with **20 kernel specializations**, achieving **1.0047x–1.0169x** speedup across $S \in [1024, 65536]$.
- **Kimi-K3 TP12 Tail Boundary (`#7316`)**: Documents `K23_MAX_TOKENS = 8` (`M <= 8` routes to the fused single-launch `k23_fused` tail kernel; `8 < M <= 960` routes to grouped ESS).
