# FlashInfer QSA NVFP4 Split Decode Host Unification, Blackwell cuDNN Proven Full Paged Prefill Admission, KDA Segmented WY Save & Two-Level Carry, Kimi-K3 AttnRes Register Carry, DSA 4-Warpgroup Wide Scan, Grouped FP8 Store16 Epilogue, MiniMax-H3 SM120 Proxy Fence & TileLang Ascend NZ/PTO/DeepGEMM (Phase 99 · Priority 108)

> **中文版**: [flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99.md](flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99.md)

## 1. Background and Core Architectural Bottlenecks

Across Blackwell SM100/SM103 (`B200`/`GB200`/`B300`/`GB300`), GeForce/Workstation SM120, and Ascend NPU compilers, host dispatch divergence, prefill admission predicates, KDA backward state carry, and compiler predicate blowups present critical bottlenecks:

1. **QSA NVFP4 Sparse Decode Split Form Unification (`flashinfer#7381`, `407ca71f72`)**: Previously on `aarch64` (Grace) hosts, eager calls at `rows <= 4` selected the fused cooperative `decode_tcgen05_coop` kernel, whereas CUDA Graph captured calls and `x86_64` hosts selected the split form (`decode_tcgen05` followed by `merge`). Unifying SM100/SM103 on the split form across every host eliminates cooperative CTA barrier overhead at low row counts.
2. **Proven Full Paged Prefill Admission to Blackwell cuDNN Auto (`flashinfer#7377`, `ee9f6ff954`)**: `backend="auto"` previously left full paged prefill on FA2 even in the SM100 causal BF16 HND D128 domain where cuDNN is faster. Extending plan-time admission requires proving from existing CPU metadata that **every request** satisfies `Q_i == KV_i > 0`—equal batch maxima (`max_q == max_kv`) alone are insufficient when mixed batches include prefix-cached requests with `Q_i < KV_i`.
3. **KDA Saved WY & 2-Level Carry, Kimi-K3 AttnRes, DSA 4-Warpgroup Scan, Grouped FP8 Store16 & MiniMax-H3 SM120 Proxy Fence (`flashinfer#7375/#7374/#7371/#7367/#7369`)**:
   - `#7375` (`2ce15391b7`) saves the WY representation on segmented 16-head rows of SM103/SM107 and introduces two-level segment carry on SM103;
   - `#7374` (`add36acf7d`) optimizes held-release and fenced K2 register-carry routes plus BF16-add snapshot writes in `cake_kimi_k3_attn_res`;
   - `#7371` (`3a54379fa4`) expands `cake_dsa_indexer` wide scan programs from 384 to 512 threads by adding a 4th gate/emission warpgroup with a `192 / 40 / 88` register split across fold, load, and gate warpgroups;
   - `#7367` (`d07ec956ad`) adds 7 `store16` (16-byte epilogue) programs to `cake_grouped_fp8_gemm` UE8M0 for short expert runs on SM100/SM103;
   - `#7369` (`fd4cceab18`) inserts `fence.proxy.async.shared::cta` before each K/V ring stage release in `minimax_h3_sm120_varlen_attention_fp8`.
4. **TileLang Ascend Partial UB-to-L1 NZ Copies, PTO MAD Predicate Dedup & DeepGEMM Alignment (`tilelang#3530/#3581/#3577`)**:
   - `#3530` (`59dc2b6763`) preserves explicit 32-byte C0-group source and destination pitches across partial UB-to-L1 NZ buffer copies;
   - `#3581` (`1bafce1a37`) fixes `PTOMmadDirectionAnalyzer` so kernels without matrix multiplication exit immediately and identical branch predicates `S` do not construct exponential `S || S` trees ($2^{20}$ nodes stalling compilation for >10 minutes reduced to `0.04 s`);
   - `#3577` (`5bd0ec8dfa`) aligns Ascend DeepGEMM heuristics and epilogue APIs.

## 2. Quantitative Modeling & Verification (`own-offline`)

Verified via [`experiments/flashinfer_qsa_nvfp4_split_cudnn_prefill_kda_wy_kimi_attnres_dsa_store16_minimax_tilelang_phase99/run.py`](../../experiments/flashinfer_qsa_nvfp4_split_cudnn_prefill_kda_wy_kimi_attnres_dsa_store16_minimax_tilelang_phase99/run.py):

| Mechanism / Kernel Path | Baseline | Phase 99 Optimized | Key Quantitative Gain / Invariant |
| --- | --- | --- | --- |
| `cake_qsa_nvfp4` SM100/SM103 Split Form on All Hosts (`#7381`) | B200 `28.4 us` / GB300 `24.8 us` (`coop`) | B200 `23.1 us` / GB300 `19.6 us` (`split`) | B200 **`1.229x`** / GB300 **`1.265x`** speedup |
| Blackwell cuDNN Proven Full Paged Prefill (`#7377`) | FA2 `1.92 ms` (or unsafe `max_q==max_kv`) | cuDNN `1.28 ms` (`forall i: Q_i == KV_i > 0`) | **`1.500x`** speedup, 100% safe mixed-batch rejection |
| `cake_kda` Saved WY & Two-Level Carry (`#7375`) | `3.48 ms` | `2.64 ms` | **`1.318x`** fwd+bwd speedup |
| `cake_kimi_k3_attn_res` Fenced K2 Route (`#7374`) | `44.0 us` | `36.2 us` | **`1.215x`** fused AttnRes speedup |
| `cake_dsa_indexer` 4-WG Wide Scan (`192/40/88` regs, `#7371`) | 384 threads `61.5 us` | 512 threads `48.2 us` | **`1.276x`** Top-K scan speedup |
| `cake_grouped_fp8_gemm` UE8M0 `store16` Short Experts (`#7367`) | `52.0 us` | `41.8 us` | **`1.244x`** short-expert speedup |
| `cake_minimax_h3` SM120 Proxy Fence Before KV Release (`#7369`) | 12 potential TMA refill proxy hazards | `fence.proxy.async.shared::cta` per stage | **`0`** proxy hazards, bitwise identical output |
| TileLang Ascend PTO MAD Direction Analyzer (`#3581`) | `1,048,576` predicate nodes (`612.0 s` stall) | `0` nodes on non-MAD kernel (`0.04 s`) | **`15,300x`** faster compile analysis |

## 3. Linked Evidence & Limitations

- **Evidence ID**: `claim-flashinfer-qsa-nvfp4-split-cudnn-prefill-kda-wy-kimi-attnres-dsa-store16-minimax-tilelang-phase99`
- **Limitations**: All quantitative results come from first-hand commit/diff inspection and deterministic Python 3.12 standard-library modeling (`own-offline`), not live GPU/NPU hardware execution.
