# Deep Dive: SGLang PD+DP Two-Phase Role Switch, TRT-LLM Attention-DP KV Event Collective Alignment & Mooncake Store Zero-Copy VMM

> **中文母稿**: [sglang-pd-dp-attention-role-switch-trtllm-dp-kv-event-flush-mooncake-zerocopy.md](sglang-pd-dp-attention-role-switch-trtllm-dp-kv-event-flush-mooncake-zerocopy.md)  
> **Phase 41 (`priority: 50`) · Deterministic Experiment**: [`experiments/sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc/run.py`](../../experiments/sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc/run.py)  
> **7-Card 4:5 Visual Report**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 1. Executive Summary

1. **SGLang Two-Phase PD Role Switch Under DP & DP Attention (`PR #44215` / `af49ada16d`)**:
   - Enables runtime Prefill ↔ Decode role switching when `dp_size > 1` or `--enable-dp-attention` is active (rejecting only `--enable-dp-attention-local-control-broadcast`).
   - **Two-Phase `prepare_only=True` Preflight**: When `num_dp_ranks > 1`, `TokenizerControlMixin.pd_role_switch` first broadcasts `PdRoleSwitchReqInput(prepare_only=True)` to verify idle state and target-role CUDA Graph memory headroom across all DP ranks before any worker tears down its KV cache, transfer backend, or CUDA graphs. If any rank fails preflight, `0` ranks tear down state (`split_brain=False, safe_to_restore=True`), eliminating partial-rank teardown split-brain.
2. **TensorRT-LLM Attention-DP Overlap Scheduler KV Event Collective Alignment (`PR #20314` / `ec782566f5`)**:
   - Moves `self._add_kv_cache_events()` (which executes an attention-DP `kv_event_allgather`) out of `_process_previous_batch()` into the rank-symmetric overlap loop body and ensures `_consume_previous_batch_for_rebalance()` runs transfer/response/KV-event collectives on ADP ranks even when `previous_batch is None`.
   - Eliminates multi-rank collective hangs across all divergent ADP rank state combinations (`("absent", "processed")`, `("processed", "deferred")`, and `("deferred", "absent")`).
3. **TensorRT-LLM Mooncake Store Zero-Copy GPUDirect RDMA & `cuMemCreate` Boundary Splitting (`PR #20108` / `69e06b8afb`)**:
   - Flips `MooncakeStoreConnectorConfig.stage_through_host` default from `True` to **`False`**.
   - Merges overlapping V2 KV pool regions (`merge_intervals`) and cuts virtual VMM reservations at `cuMemCreate` physical allocation boundaries (`allocation_spans` and `split_at_cuts`), auto-enables `WITH_NVIDIA_PEERMEM=0` (`ibv_reg_dmabuf_mr`) on GB200/GB300 open-kernel hosts without `nvidia_peermem`, and saves **`1,024 MiB`** of pinned host staging DRAM on an 8-GPU node.

---

## 2. First-Hand Upstream Commits

| Repository | Commit / PR | Date | Core Files | Evidence ID |
|---|---|---|---|---|
| `sgl-project/sglang` | [`af49ada16d` (`#44215`)](https://github.com/sgl-project/sglang/commit/af49ada16d9b4e7535fd6ff542247f50b8c92c9f) | `2026-10-21` | `python/sglang/srt/disaggregation/role_switch.py`, `tokenizer_control_mixin.py` | `EV-2026-358` |
| `NVIDIA/TensorRT-LLM` | [`ec782566f5` (`#20314`)](https://github.com/NVIDIA/TensorRT-LLM/commit/ec782566f50ad9548e99818fa1d74c7c66cc572d) | `2026-10-21` | `tensorrt_llm/_torch/pyexecutor/py_executor.py` | `EV-2026-359` |
| `NVIDIA/TensorRT-LLM` | [`69e06b8afb` (`#20108`)](https://github.com/NVIDIA/TensorRT-LLM/commit/69e06b8afbffc4726a63b43fb4acb9fda340f3a7) | `2026-10-21` | `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/{addressing.py,gpudirect.py}` | `EV-2026-360` |

---

## 3. Deterministic Reproduction & Validation Metrics

| Subsystem | Metric | Deterministic Value | Status |
|---|---|---|---|
| SGLang PD+DP Switch | Single-Phase vs Two-Phase Partial Failure (`3/4` Ready) | `split_brain: True (3 torn down)` vs `False (0 torn down)` | `PASS` |
| SGLang PD+DP Switch | `--enable-dp-attention` Allowed / `local-control-broadcast` Rejected | `True` / `True` | `PASS` |
| TRT-LLM ADP Events | Overlap Loop Deadlock-Free Scenarios (`absent/processed/deferred`) | `1/3 -> 3/3 (100%)` | `PASS` |
| TRT-LLM ADP Events | KV Pool Rebalance (`absent` + `processed`) Deadlock-Free | `False -> True` | `PASS` |
| Mooncake Zero-Copy | `96 MiB` Pool Merged Spans / `32 MiB` `cuMemCreate` Registration Pieces | `1` / `3` | `PASS` |
| Mooncake Zero-Copy | Straddling `4 MiB` Page Split / 8-GPU Pinned DRAM Saved | `[2 MiB, 2 MiB]` / `1,024 MiB` | `PASS` |

---

## 4. Production Adoption Recommendations

1. **Global Control Broadcast for Elastic PD+DP**: Keep `--enable-dp-attention-local-control-broadcast` disabled when using SGLang runtime PD role switching so the tokenizer process coordinates the two-phase `prepare_only` barrier across all DP ranks.
2. **Zero-Copy V2 KV Pools on GB200/GB300**: Retain the new `stage_through_host=False` default in TRT-LLM Mooncake Store; `PageAddressing` automatically splits pages crossing `cuMemCreate` physical boundaries and falls back to `ibv_reg_dmabuf_mr` when `nvidia_peermem` is absent.
