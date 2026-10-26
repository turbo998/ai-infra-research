# FlashInfer Decode BMM Swap-AB, Cake NVFP4 Attention, Hopper MSA Consolidation & TileLang Barrier Isolation (Phase 87)

> **Evidence Classification**: First-hand Implementation Audit (`implementation` + `author-benchmark` + `own-offline` deterministic Python 3.12 stdlib model `experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py`).

## 1. Key Architectural Findings & First-Hand Source Paths

| Upstream Commit / PR | Source Paths | Architectural Change & Quantitative Metrics |
|---|---|---|
| `flashinfer-ai/flashinfer#7351` (`563a9021ad`, 2026-10-25) | `csrc/cake_bf16_bmm_swap_ab/*` | Adds `tcgen05` swap-AB programs for decode-row `flashinfer.gemm.bmm_bf16` on SM100/SM103 (weight rows fill MMA $M$, decode tokens fill MMA $N$, optional cluster split-K, single rounding of FP32 accumulator to BF16/FP16/FP32). Across 9 routed decode shapes per architecture ($K=1024, M \in [8,33], \text{batch} \in [1,4]$), kernels run **1.01x–1.20x** faster on B200 (`5.50 -> 4.59 µs` at `batch=2, M=33, N=1032`) and **1.03x–1.22x** faster on GB300 (`5.15 -> 4.22 µs`) than the hand-written `mma.sync` Cake kernels. |
| `flashinfer-ai/flashinfer#7362` (`f83439d2a6`, 2026-10-25) | `flashinfer/experimental/nvfp4_attention/*` | Ships Cake NVFP4 non-causal dense MHA ($D=128$, BF16 in/out, FP32 softmax, on-GPU statistics + smoothing/quantization on every call) across SM100/SM103 (4-kernel chain: NVFP4 QK + NVFP4 PV) and SM120 (3-kernel chain: INT8 QK + NVFP4 PV). Under CUDA-graph replay, beats FlashAttention-4 BF16 by **1.01x–1.06x** on B200 and **1.03x–1.08x** on GB300 for long sequences `R2–R7` ($S \in \{32760, 75600\}$), while `R1` ($S=4800$) is `0.72x/0.70x` due to on-GPU quantization overhead; on RTX PRO 6000 Blackwell (`sm_120a`), beats PyTorch cuDNN SDPA BF16 by **1.41x–1.84x** across all `R1–R7`. |
| `flashinfer-ai/flashinfer#7332` (`01b18dc98d`, 2026-10-25) | `csrc/cake_hopper_msa/*`, `flashinfer/jit/cake_hopper_msa.py`, `flashinfer/msa_ops/cake_hopper_sm90.py` | Four-commit mechanical consolidation of the 89 SM90 Hopper MSA JIT programs: (1) replaces 89 bindings with 15 role-shared `-D`-parameterized bindings (74 deleted); (2) folds 38 constant-clone kernels into 20 shared sources; (3) extracts the preamble and 13 device helpers into `cake_hopper_msa_device.h`; (4) removes dead host code (including the unused `done` scratch buffer) and compacts 89 `arg_plan` records into 7 `_ARG_PLANS` entries. Reduces `csrc/cake_hopper_msa/` from **179 to 68 files** (`+2,313 / -79,655`, net **-77,342 lines**) with proven SASS and symbol-table identity across all 89 programs. |
| `flashinfer-ai/flashinfer#7331` (`85fcf53577`, 2026-10-25) & `#7317` (`506fac6203`, 2026-10-25) | `flashinfer/cudnn/{decode,prefill,utils}.py` | `#7317` caches `_cudnn_version_supports_shape_override()` with `functools.cache` (`0.897 -> 0.072 µs` on EPYC 9554, **12.46x** faster); `#7331` avoids Python `torch.cuda.Stream` object construction on warm calls by querying the raw stream pointer, cutting B200 warm host enqueue from `32.796 -> 31.154 µs` (`-5.0%`) for ragged prefill and `35.717 -> 32.706 µs` (`-8.4%`) for HND decode. |
| `flashinfer-ai/flashinfer#7320` (`a02c8c5e6b`, 2026-10-25) & `#7354` (`66a780899a`, 2026-10-25) | `flashinfer/diffusion_ops/cake_minimax_h3_sm120_quant_varlen_attention.py`, `flashinfer/cake_causal_conv1d_train_backend/build.json` | `#7320` validates 32-bit signed offset bounds (`_MAX_ELEMENTS = (1 << 31) - 1`) at plan construction time for SM120 MiniMax-H3 varlen attention (max safe padded tokens = **299,520** at 56 heads) and statistics kernels; `#7354` raises `min_cuda` from `"13.0"` to `"13.4"` for all 24 `sm_107a` entries in `cake_causal_conv1d_train/build.json`. |
| `tile-ai/tilelang#3643` (`efa1ffc1e3`, 2026-10-25), `#3193` (`35ee295075`, 2026-10-23), `#3589` (`baceed6676`, 2026-10-23) | `src/transform/thread_storage_sync.cc`, `tilelang/jit/adapter/nvrtc/adapter.py`, `examples/ascend/deepgemm/*` | `#3643` fixes a warp-specialized H100 deadlock in `ThreadPartialSyncRewriter` where consecutive `ThreadSync("shared")` and `ThreadSync("shared.dyn")` passes reused named barrier IDs (`bar.sync 3`) across disjoint warpgroups (`[256, 384)` vs `[0, 128)`); `#3193` fixes NVRTC autotune result persistence after disk-cache hits; `#3589` adds FP4xFP4 support to Ascend DeepGEMM examples. |

## 2. Mathematical Formulation & Architect Sizing Rules

### 2.1 Decode-Row `bmm_bf16` Swap-AB Tensor Memory Mapping
For decode-phase batched GEMM $C_{[B, M, N]} = A_{[B, M, K]} \times W_{[B, K, N]}$ with small decode token counts $M \in [8, 33]$, mapping $M$ directly to the `tcgen05.mma` hardware row dimension wastes $50\%\text{–}87.5\%$ of Tensor Core lanes. Applying the transpose identity:
$$C^T_{[B, N, M]} = W^T_{[B, N, K]} \times A^T_{[B, K, M]}$$
places weight rows $N \in [40, 1032]$ into the MMA $M$ dimension and decode tokens $M \in [8, 33]$ into the MMA $N$ dimension, yielding **1.03x–1.22x** cold-L2 speedups on GB300.

### 2.2 SM120 Varlen Attention 32-Bit Signed Indexing Invariant
With block padding $B_N = 128$, head count $H = 56$, and head dimension $D = 128$:
$$T_{\text{pad}} = \max\left(1, \left\lceil \frac{T}{128} \right\rceil\right) \times 128$$
All three 32-bit signed offset domains must stay within $2^{31} - 1 = 2,147,483,647$:
$$T_{\text{pad}} \cdot H \cdot D \le 2^{31} - 1, \quad S_{\text{seg}} \cdot H \cdot 128 \le 2^{31} - 1, \quad N_{\text{tiles}} \cdot H \cdot 256 \le 2^{31} - 1$$
At $H=56, D=128$, the maximum safe padded token count is **299,520** ($T=299,521$ rounds up to $299,648 \times 7,168 = 2,147,876,864 > 2^{31}-1$ and is rejected during `MiniMaxH3VarlenPlan.__init__` with zero per-call overhead).

## 3. Reproducible Verification (`experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87`)

Running `python experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py --check` validates all deterministic metrics in `summary.json`.
