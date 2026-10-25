# Deep Dive: FlashAttention-4 SM100 Decode S/P TMEM Ping-Pong, hd256 2-CTA Backward Unification, CUTLASS CLC Epilogue Counter, and veRL ModelOpt QAT Child-Quantizer Ignore

> **Status**: Phase 80 (`priority: 89`) Primary-Source Code Audit  
> **Bilingual Pair**: [`fa4-sm100-decode-sp-pingpong-hd256-2cta-bwd-cutlass-clc-epi-verl-qat-slime.md`](fa4-sm100-decode-sp-pingpong-hd256-2cta-bwd-cutlass-clc-epi-verl-qat-slime.md)  
> **Reproducible Experiment**: [`experiments/fa4_sm100_sp_pingpong_hd256_2cta_bwd_cutlass_clc_epi_verl_qat_slime/run.py`](../../experiments/fa4_sm100_sp_pingpong_hd256_2cta_bwd_cutlass_clc_epi_verl_qat_slime/run.py)

---

## 1. Primary-Source Evidence Matrix

| Evidence ID | Upstream Commit | Key Module & Change | Category |
| --- | --- | --- | --- |
| `src-fa4-sm100-decode-sp-pingpong-splitkv-2026` | `Dao-AILab/flash-attention@e9515d5dee6a` (`#2928`, `#2920`) | SM100 Decode (`q_stage=1`) S/P TMEM ping-pong (`QK(i+1)` before `PV(i)`) & `num_splits_heuristic >= 1` clamp | `implementation` |
| `src-fa4-sm100-hd256-sm103-pingpong-2cta-2026` | `Dao-AILab/flash-attention@ab3e5bd987de` (`#2951`) | Enable S ping-pong on SM103 (`GB300`); `hd256` 2-CTA for varlen/decode, disable CLC on 2-CTA, keep 1-CTA with caller metadata | `implementation` |
| `src-fa4-sm100-hd256-bwd-cta-schedule-tmem-2026` | `Dao-AILab/flash-attention@76d933102819` (`#2964`) | `CtaSchedule` enum (`ONE_CTA`, `TWO_CTA_PIPELINED`, `TWO_CTA_SERIAL`) and named TMEM column map | `implementation` |
| `src-fa4-paged-kv-cudagraph-capture-fix-2026` | `Dao-AILab/flash-attention@73019affed8c` (`#2984`) | Skip host-synchronizing `.max().item()` bounds check during CUDA graph capture in `mha_fwd_kvcache` | `implementation` |
| `src-cutlass-cutedsl-clc-epilogue-tile-counter-2026` | `NVIDIA/cutlass@84a62ec421c2` (`#3636`) | Track `num_tiles_executed` locally in epilogue warps so CLC dynamic GEMM rotates TMA-store SMEM stages | `implementation` |
| `src-cutlass-cutedsl-grouped-gemm-2cta-codegen-2026` | `NVIDIA/cutlass@af3d4367a80c` (`#3639`) | Select vectorized RMEM + `-O2` for 2-CTA and scalar RMEM + `-O3` for 1-CTA on CUDA 12.9 grouped GEMM | `implementation` |
| `src-verl-modelopt-qat-module-ignore-children-2026` | `volcengine/verl@5d09d065b2a6` (`#8329`) | Append `f"{key}.*"` in `_ignore_patterns_to_quant_cfg` to disable `.input_quantizer` and `.weight_quantizer` | `implementation` + `author-benchmark` |
| `src-verl-device-agnostic-linear-ce-tests-2026` | `volcengine/verl@d430157d1551` (`#8284`) | Migrate `LinearCrossEntropy` and unit tests to `verl.utils.device` accelerator abstraction | `implementation` |
| `src-slime-prefilter-raw-reward-logging-2026` | `THUDM/slime@57ff5d669be6` (`#2475`) | Compute and log `raw_reward` before dynamic rollout filtering in `slime/ray/rollout.py` | `implementation` |

---

## 2. FlashAttention-4 SM100 Decode S/P TMEM Ping-Pong & SplitKV Clamp

In `Dao-AILab/flash-attention@e9515d5dee6a` (`#2928`, `#2920`) and `ab3e5bd987de` (`#2951`), Blackwell SM100/SM103 (`B200`/`GB300`) decode (`q_stage = 1`) overlaps $Q K^T$ with softmax across two TMEM S/P slots:

- **Dual-Slot S/P Ping-Pong (`s_p_o_stage = 2`)**: Previously at `q_stage = 1`, the second TMEM S/P slot sat unused and the MMA warp waited for softmax to drain $S_i$ and write $P_i$ before issuing $Q K_{i+1}^T$. With `use_s_ping_pong = True` (gated to `S_PING_PONG_MIN_N_BLOCKS_PER_SPLIT = {64: 16, 128: 64}`), KV tiles load in $K_0, K_1, V_0, K_2, V_1, \dots$ order and $Q K_{i+1}^T$ is issued into the alternate TMEM slot **before** $P_i V_i$, reducing an 8-tile modeled mainloop from `224` to `168` units (**1.333x** speedup).
- **`num_splits_heuristic >= 1` (`#2920`)**: Clamps `max(1, min(num_SMs // total_mblocks, max_splits, num_n_blocks))` so `total_mblocks > num_SMs` (`256 > 148`) returns `1` split instead of `0`.
- **SM103 & `hd256` 2-CTA Guards (`#2951`, `#2964`, `#2984`)**: Enables S ping-pong on SM103 (`arch // 10 in (10, 11)`), disables CLC for `hd256` 2-CTA, preserves 1-CTA when reusing caller-built `scheduler_metadata`, unifies backward 2-CTA schedules under `CtaSchedule`, and skips `.max().item()` during CUDA graph capture.

---

## 3. CUTLASS CuTeDSL Epilogue-Local CLC Counter (`#3636`) & 2-CTA Codegen (`#3639`)

- **Epilogue-Local Tile Counter (`#3636`)**: In `dense_gemm_persistent_dynamic.py`, only the scheduler warp issues CLC advance queries, leaving `tile_sched.num_tiles_executed == 0` in epilogue warps and causing every tile to reuse SMEM stage `0` while prior TMA stores were still in flight on short-K shapes (`8256x8448x128`). Incrementing a local `num_tiles_executed` in the epilogue restores proper 4-stage (`0, 1, 2, 3`) rotation.
- **CUDA 12.9 CTA-Mode Codegen (`#3639`)**: Selects vectorized RMEM + `-O2` for 2-CTA grouped GEMM and scalar RMEM + `-O3` for 1-CTA.

---

## 4. veRL ModelOpt QAT Child-Quantizer Ignore (`#8329`) & Slime Pre-Filter Reward (`#2475`)

- **veRL ModelOpt QAT (`#8329`)**: Appends `{"quantizer_name": f"{key}.*", "enable": False}` in `_ignore_patterns_to_quant_cfg` so module-level ignore globs (`*self_attention.linear_proj`) disable `.input_quantizer` and `.weight_quantizer` without matching sibling prefixes (`in_proj_ab`). Verified on H100 across 10 GRPO steps on Qwen3-0.6B: active quantizers on 28 ignored `o_proj` layers dropped from **56/56 to 0/56**, and max QDQ error dropped from **0.0859375 to 0.0**.
- **Slime Pre-Filter Reward Logging (`#2475`)**: Computes `raw_reward` statistics prior to dynamic rollout filtering so filtered-out trajectories do not skew reported training rewards.
