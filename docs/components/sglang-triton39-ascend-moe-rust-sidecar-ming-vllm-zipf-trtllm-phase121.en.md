# SGLang Triton 3.9 Block-Pointer Compat, Ascend Ops MoE Hybrid All-to-All, Rust Server Per-Node Sidecar & Abort Stream Framing, Ming-Omni 3840-Width Ordered RMSNorm, vLLM Anthropic Inline-System Probe, Rust KV-Fetch Stage Metrics, Zipf MoE Routing Benchmark & TensorRT-LLM Inductor Pool Keep-Alive and MiniMax-H3 DiT TP (Phase 121)

> **Module Positioning**: Written for cloud and AI infrastructure solution architects designing production LLM/DiT serving runtimes (SGLang, vLLM, TensorRT-LLM), multi-node Rust control planes, and realistic MoE benchmark harnesses. Based on 10 primary upstream commits landed on 2026-10-28.
> **Evidence Scope**: `implementation` (verified against pinned commit diffs) + `author-benchmark` + `own-offline` (reproducible Python standard-library modeling in this repository).

---

## 1. Architectural Overview & 10 Upstream Commits

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Scope |
| :--- | :--- | :--- | :--- | :--- |
| B1 | `sgl-project/sglang` [`0b158f7`](https://github.com/sgl-project/sglang/commit/0b158f75e79696ca87158bf65c20a40393e817e9) | 2026-10-28 | `triton_block_ptr_compat.py`: Restore `tl.make_block_ptr` and `tl.advance` on Triton 3.9 via a 10-tuple `_BlockPtr` aggregate | `implementation` |
| B2 | `sgl-project/sglang` [`bacf65f`](https://github.com/sgl-project/sglang/commit/bacf65f8f1cd24744aa60d735f4ecb8b7cb50d76) | 2026-10-28 | `ascend_ops`: Add Ascend NPU MoE All-to-All backend and `--moe-ascend-ops-config` (`{"prefill":"alltoallv","decode":"low_latency"}`) | `implementation` |
| B3 | `sgl-project/sglang` [`b7209cf`](https://github.com/sgl-project/sglang/commit/b7209cf53465dc2298c74719b303fdf44a80750b) | 2026-10-28 | `rust_server`: Launch exactly one `--sidecar` per node against the node's first Rust listener (`--grpc-port`) | `implementation` |
| B4 | `sgl-project/sglang` [`4b43572`](https://github.com/sgl-project/sglang/commit/4b43572e522912641ca7c83c5045f8c92bdad4c4) | 2026-10-28 | `diffusion/ming`: Extend `rmsnorm_cast_before_weight_bitexact` to width `3840` (`WARP_REDUCE_WIDTH=4`) and parameterized `EPS` | `implementation` |
| B5 | `sgl-project/sglang` [`6a473b2`](https://github.com/sgl-project/sglang/commit/6a473b2ac75184f0fa635481b6e7a6c8b1dfcb60) | 2026-10-28 | `rust-server`: Convert scheduler abort frame (`CoreError::RuntimeRejected`) into `CoreEvent::Finished` on native `/generate` streams | `implementation` |
| B6 | `vllm-project/vllm` [`b766f15`](https://github.com/vllm-project/vllm/commit/b766f15983bd7b926c93015dd8551a6101a2ca20) | 2026-10-28 | `anthropic/protocol.py`: Widen `_detect_merge_inline_system` catch to `except Exception:` so `tools | tojson` `TypeError` falls back to `True` | `implementation` |
| B7 | `vllm-project/vllm` [`68eca9a`](https://github.com/vllm-project/vllm/commit/68eca9af01b8b323667cb4b87148547e8b98eef6) | 2026-10-28 | `vllm-rust`: Export `vllm:num_requests_kv_fetch_by_stage` (`waiting_to_start`, `in_progress`, `completed_waiting`) | `implementation` |
| B8 | `vllm-project/vllm` [`6282c4c`](https://github.com/vllm-project/vllm/commit/6282c4c5cf7c58b1b0dfeefb0ff78bc32e6d8a9e) | 2026-10-28 | `forward_perf.py`: Default to `zipf_topk_ids` (`alpha=1.25`, per-layer shuffled expert popularity) for dummy weights | `implementation` |
| B9 | `NVIDIA/TensorRT-LLM` [`c9d2dcc`](https://github.com/NVIDIA/TensorRT-LLM/commit/c9d2dcc26e2bb79f7b7fd4327bfd5c7a69ccd55b) | 2026-10-28 | `_torch/AutoDeploy`: Set `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` via `setdefault` to prevent TP collective stalls on late recompile | `implementation` |
| B10 | `NVIDIA/TensorRT-LLM` [`a8ce107`](https://github.com/NVIDIA/TensorRT-LLM/commit/a8ce1074a66b30aa48cb7cb175e3582f45c3c9d3) | 2026-10-28 | `minimax_h3_dit`: Relax constraint to `ulysses_size * tp_size == world_size` for hybrid Ulysses + TP DiT parallelism | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 Triton 3.9 Block-Pointer Compat & Ming-Omni Width-3840 Bit-Exact RMSNorm (`0b158f7`, `4b43572`)

- **Triton 3.9 Block Pointers (`0b158f7`)**: Reconstructs `tl.make_block_ptr` and `tl.advance` on Triton 3.9 using a 10-field `tl.tuple` aggregate `_BlockPtr(base, s0, s1, d0, d1, o0, o1, b0, b1, order_flag)` and `_materialize` pointer/mask synthesis, keeping FLA backward/forward kernels (`chunk_bwd_dv_local`, `wy_fast`, `solve_tril`) bit-exact without kernel rewrites.
- **Ming-Omni Width-3840 Ordered RMSNorm (`4b43572`)**: Supports hidden dimension $H = 3840$ by setting `WARP_REDUCE_WIDTH = 4` ($3840 / (32 \times 30) = 4$ lanes remaining after 32-way warp reduction) and parameterizing `EPS` (`1e-5` / `1e-6`) behind `_MING_RMSNORM_FUSION`.

### 2.2 vLLM Zipf MoE Routing Skew & TensorRT-LLM Inductor Pool Keep-Alive (`6282c4c`, `c9d2dcc`)

- **Zipf MoE Routing (`6282c4c`)**: Replaces uniform dummy-weight expert selection with `zipf_topk_ids` ($w_r = r^{-1.25}$ with per-layer random permutation $\pi_l$). Across $E = 64, K = 8$, the top 8 experts hold **70.76%** of the unnormalized routing mass (vs **12.50%** uniform), and the hottest expert has **20.1027x** the mean single-draw weight.
- **Inductor Async Compile Pool Keep-Alive (`c9d2dcc`)**: Setting `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` prevents PyTorch from shutting down and re-spawning the 1-process compile pool on single-rank dynamic-shape recompiles, eliminating multi-rank TP collective stalls.

---

## 3. Reproducible Offline Verification (`own-offline`)

Experiment directory: `experiments/sglang_triton39_ascend_moe_rust_sidecar_ming_vllm_zipf_trtllm_phase121/`

- **Command**: `python experiments/sglang_triton39_ascend_moe_rust_sidecar_ming_vllm_zipf_trtllm_phase121/run.py --check`
- **Verified Sealed Outputs**:
  - `sglang_triton39_and_ascend.step3_valid_rows = 4`, `step3_masked_rows = 28`;
  - `sglang_rust_and_ming.new_sidecars_per_node = 1`, `ming_warp_reduce_width = 4`;
  - `vllm_frontend_and_zipf.zipf_top8_mass_share = 0.7076`, `zipf_max_to_mean_expert_ratio = 20.1027`;
  - `trtllm_inductor_and_minimax.minimax_dit_supports_hybrid_ulysses_tp = true`.

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: Under Zipf routing ($E = 64, K = 8, \alpha = 1.25$) with batch size $B = 512$ tokens ($4,096$ total assignments), compare uniform expected tokens per expert against the hottest expert's inclusion probability ($p_1 = 0.3141$).
   - *Answer*: Uniform assigns $4096 / 64 = 64$ tokens/expert; the #1 Zipf expert is selected in $> 1 - (1 - 0.3141)^8 \approx 95.1\%$ of tokens ($\ge 487$ tokens).
2. **Falsifiable Question**: How much resident host RAM does keeping `TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0` alive consume per rank on an 8-GPU node?
