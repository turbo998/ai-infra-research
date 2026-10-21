# SGLang PD Disaggregation Two-Phase Role Switch under DP / DP-Attention and vLLM DeepSeek-V4.1 Replay Target-Token Row Preservation

> **中文版本**: [sglang-pd-dp-role-switch-vllm-dsv41-replay-rows.md](sglang-pd-dp-role-switch-vllm-dsv41-replay-rows.md)  
> **Question ID**: `sglang-pd-dp-role-switch-vllm-dsv41-replay-rows` (`priority: 51`, Phase 42)  
> **Offline Experiment**: [`experiments/sglang_pd_dp_role_switch_vllm_dsv41_replay/run.py`](../../experiments/sglang_pd_dp_role_switch_vllm_dsv41_replay/run.py) · [`summary.json`](../../experiments/sglang_pd_dp_role_switch_vllm_dsv41_replay/summary.json)  
> **Evidence Tier**: `implementation` (primary source audit) + `own-offline` (deterministic Python stdlib model)

---

## 1. Architectural Context & Problem Statement

Production LLM serving stacks operating elastic Prefill/Decode (PD) disaggregation and multi-pass decoder layer replay face two subtle control-plane and tensor-state hazards:

1. **SGLang PD Role Switch Blocking & Split-Brain Hazard under `--dp-size > 1` / `--attn-dp-size > 1`**: SGLang's `--enable-pd-role-switch` previously rejected any configuration with `--dp-size > 1` or `--attn-dp-size > 1`, preventing Data Parallel (DP) and DP-Attention deployments from dynamically shifting workers between Prefill and Decode roles. Furthermore, across multiple DP workers, broadcasting an immediate single-phase teardown-and-switch command risks **split-brain corruption** if a subset of DP workers tear down their KV caches and flip roles while one worker rejects the switch.
2. **vLLM DeepSeek-V4.1 Target-Token Scoring Prompt Row Truncation**: In vLLM's DeepSeek-V4.1 NVIDIA backend (`vllm/models/deepseek_v41/nvidia/model_state.py`), decoder layer replay prunes intermediate prompt rows to `1` (`_keeps_rows_np[req_index] = False`) unless full prompt outputs are needed. When a caller requested target-token scoring via `params.prompt_logprob_token_ids` while leaving `params.prompt_logprobs = None`, `ModelState.add_request` omitted `prompt_logprob_token_ids`, discarding intermediate prompt token rows during replay.

This document analyzes two upstream commits landed on `2026-10-21`:
- **`sgl-project/sglang` Commit `af49ada16d9b4e7535fd6ff542247f50b8c92c9f` (PR `#44215`)**: Enables PD runtime role switching with `--dp-size > 1` and `--attn-dp-size > 1`, guarded by a two-phase `prepare_only=True` preflight broadcast.
- **`vllm-project/vllm` Commit `85752a43c3cee76d9234704e189cc2ca69f24712` (PR `#62652`)**: Preserves prompt rows in `ModelState.add_request` when `params.prompt_logprob_token_ids is not None`.

---

## 2. SGLang Two-Phase PD Role Switch under DP & DP-Attention

### 2.1 Refined Admission Boundary (`pd_disaggregation_hook.py`)

In `python/sglang/srt/disaggregation/pd_disaggregation_hook.py:39-94`, `validate_pd_role_switch_config` lifts the blanket ban on `view.dp_size > 1` and `attn_dp_enabled_of(view)`, rejecting only topologies that lack centralized control-plane fan-out or require cross-rank state repartitioning:

$$\text{Reject} \iff \begin{cases} \text{disaggregation\_mode} \notin \{\text{prefill}, \text{decode}\} \\ \text{attn\_dp\_enabled\_of}(v) \land v.\text{enable\_dp\_attention\_local\_control\_broadcast} \\ v.\text{ep\_size} > 1 \;\lor\; v.\text{moe\_a2a\_backend} \neq \text{"none"} \\ v.\text{pp\_size} > 1 \;\lor\; v.\text{dcp\_size} > 1 \end{cases}$$

### 2.2 Two-Phase `prepare_only` Preflight Protocol (`tokenizer_control_mixin.py` & `role_switch.py`)

PR `#44215` adds `prepare_only: bool = False` to `PdRoleSwitchReqInput` (`io_struct.py:1945-1960`) and implements a two-phase protocol across `TokenizerControlMixin.pd_role_switch` (`tokenizer_control_mixin.py:424-456`) and `Scheduler.pd_role_switch` (`role_switch.py:67-115`):

1. **Phase 1 (`prepare_only=True` when `num_dp_ranks > 1`)**: The tokenizer manager broadcasts `PdRoleSwitchReqInput(new_role=obj.new_role, prepare_only=True)` to all DP workers and waits for all `num_dp_ranks` responses. Each scheduler validates role-switch eligibility without setting `_pd_role_switch_in_progress` or tearing down KV/transfer engines, returning `PdRoleSwitchReqOutput(success=True, message="ready", safe_to_restore=True)`. If any rank fails (`not all(r.success for r in results)`), the switch aborts cleanly with zero state mutation on any worker.
2. **Phase 2 (`prepare_only=False`)**: Only after all DP ranks report ready does the tokenizer manager broadcast the commit request (`prepare_only=False`), drain requests, flip each worker's role, and update `self.disaggregation_mode = DisaggregationMode(obj.new_role)`.

---

## 3. vLLM DeepSeek-V4.1 Replay Row Preservation for Target-Token Scoring (`#62652`)

In `vllm/models/deepseek_v41/nvidia/model_state.py:425-429`, `ModelState.add_request` determines whether a request must retain all prompt rows during decoder layer replay:

```python
self._keeps_rows_np[req_index] = (
    params.prompt_logprobs is not None
    or params.prompt_logprob_token_ids is not None
    or self.enable_return_routed_experts
)
```

Adding `or params.prompt_logprob_token_ids is not None` ensures that target-token scoring requests (which set `prompt_logprob_token_ids` while leaving `prompt_logprobs=None`) preserve all $L_{\text{prompt}}$ rows across replay, while normal generation requests (`prompt_logprobs=None` and `prompt_logprob_token_ids=None`) continue to prune replay rows to `1`.

---

## 4. Deterministic Offline Verification (`own-offline`)

Running `python experiments/sglang_pd_dp_role_switch_vllm_dsv41_replay/run.py --check` confirms:
- **DP / DP-Attention Validation**: `dp_size=4, attn_dp_size=4` is admitted (`True`) when `enable_dp_attention_local_control_broadcast=False` and rejected (`False`) when `True`.
- **Zero Split-Brain under Partial DP Failure**: When rank 3 of 4 rejects a switch (`[True, True, True, False]`), single-phase execution causes a 3-decode / 1-prefill split-brain (`split_brain=True`), whereas the two-phase `prepare_only=True` preflight aborts before any rank mutates state (`split_brain=False`, all 4 workers remain `prefill`).
- **DeepSeek-V4.1 Target-Token Scoring Alignment**: For `prompt_len=16` with `prompt_logprob_token_ids=[101, 202, 303]`, the fix preserves all `16` replayed rows (`target_positions_valid=True`) vs `1` row before the fix.
