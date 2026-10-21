from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATHS = [ROOT / "summary.json", ROOT / "results" / "summary.json"]


def validate_pd_role_switch_config(
    enable_pd_role_switch: bool,
    disaggregation_mode: str,
    dp_size: int = 1,
    attn_dp_size: int = 1,
    enable_dp_attention_local_control_broadcast: bool = False,
    ep_size: int = 1,
    moe_a2a_backend: str = "none",
    pp_size: int = 1,
    dcp_size: int = 1,
) -> tuple[bool, list[str]]:
    # Mirrors validate_pd_role_switch_config in sglang/srt/disaggregation/pd_disaggregation_hook.py (#44215, af49ada16d)
    if not enable_pd_role_switch:
        return True, []
    errors: list[str] = []
    if disaggregation_mode not in ("prefill", "decode"):
        errors.append("--enable-pd-role-switch requires --disaggregation-mode prefill or decode")
    if attn_dp_size > 1 and enable_dp_attention_local_control_broadcast:
        errors.append("local DP-attention control broadcast (--enable-dp-attention-local-control-broadcast)")
    if ep_size > 1:
        errors.append("expert parallelism (--ep-size > 1)")
    if moe_a2a_backend != "none":
        errors.append("MoE all-to-all dispatch (--moe-a2a-backend != none)")
    if pp_size > 1:
        errors.append("pipeline parallelism (--pp-size > 1)")
    if dcp_size > 1:
        errors.append("decode context parallelism (--dcp-size > 1)")
    return len(errors) == 0, errors


def simulate_dp_role_switch_protocol(
    num_dp_ranks: int,
    initial_role: str,
    new_role: str,
    worker_can_switch: list[bool],
    use_two_phase_prepare: bool,
) -> dict[str, object]:
    # Mirrors TokenizerControlMixin.pd_role_switch in tokenizer_control_mixin.py and Scheduler.pd_role_switch in role_switch.py (#44215)
    worker_roles = [initial_role] * num_dp_ranks
    tokenizer_role = initial_role

    if use_two_phase_prepare and num_dp_ranks > 1:
        # Phase 1: broadcast prepare_only=True
        prepare_results = []
        for rank in range(num_dp_ranks):
            if initial_role == new_role:
                prepare_results.append({"rank": rank, "success": True, "message": "no-op", "safe_to_restore": True})
            elif worker_can_switch[rank]:
                prepare_results.append({"rank": rank, "success": True, "message": "ready", "safe_to_restore": True})
            else:
                prepare_results.append({"rank": rank, "success": False, "message": f"rank {rank} busy/incompatible", "safe_to_restore": True})
        if not all(r["success"] for r in prepare_results):
            # Abort before any worker tears down KV/transfer engines
            return {
                "use_two_phase_prepare": True,
                "overall_success": False,
                "tokenizer_role": tokenizer_role,
                "worker_roles": worker_roles,
                "split_brain": len(set(worker_roles)) > 1 or tokenizer_role != worker_roles[0],
            }

    # Phase 2 (or legacy single-phase): execute role switch
    exec_results = []
    for rank in range(num_dp_ranks):
        if worker_can_switch[rank]:
            worker_roles[rank] = new_role
            exec_results.append({"rank": rank, "success": True})
        else:
            exec_results.append({"rank": rank, "success": False})

    overall_success = all(r["success"] for r in exec_results)
    if overall_success:
        tokenizer_role = new_role

    return {
        "use_two_phase_prepare": use_two_phase_prepare,
        "overall_success": overall_success,
        "tokenizer_role": tokenizer_role,
        "worker_roles": worker_roles,
        "split_brain": len(set(worker_roles)) > 1 or any(r != tokenizer_role for r in worker_roles),
    }


def simulate_vllm_dsv41_replay_row_preservation(
    prompt_len: int = 8,
    prompt_logprobs: int | None = None,
    prompt_logprob_token_ids: list[int] | None = None,
    enable_return_routed_experts: bool = False,
    include_fix: bool = True,
) -> dict[str, object]:
    # Mirrors ModelState.add_request in vllm/models/deepseek_v41/nvidia/model_state.py (#62652, 85752a43c3)
    if include_fix:
        keeps_rows = (
            prompt_logprobs is not None
            or prompt_logprob_token_ids is not None
            or enable_return_routed_experts
        )
    else:
        keeps_rows = (
            prompt_logprobs is not None
            or enable_return_routed_experts
        )

    replayed_rows = prompt_len if keeps_rows else 1
    target_positions_valid = (
        True if prompt_logprob_token_ids is None else replayed_rows == prompt_len
    )
    return {
        "prompt_len": prompt_len,
        "prompt_logprobs": prompt_logprobs,
        "prompt_logprob_token_ids": prompt_logprob_token_ids,
        "include_fix": include_fix,
        "keeps_rows": keeps_rows,
        "replayed_rows": replayed_rows,
        "target_positions_valid": target_positions_valid,
    }


def build_summary() -> dict[str, object]:
    cfg_dp4_attn_dp4, errs_ok = validate_pd_role_switch_config(
        True, "prefill", dp_size=4, attn_dp_size=4, enable_dp_attention_local_control_broadcast=False
    )
    cfg_local_bcast, errs_local_bcast = validate_pd_role_switch_config(
        True, "prefill", dp_size=4, attn_dp_size=4, enable_dp_attention_local_control_broadcast=True
    )

    # Partial worker rejection on DP=4 (ranks 0,1,2 ready, rank 3 rejects)
    legacy_one_phase = simulate_dp_role_switch_protocol(
        4, "prefill", "decode", [True, True, True, False], use_two_phase_prepare=False
    )
    two_phase_preflight = simulate_dp_role_switch_protocol(
        4, "prefill", "decode", [True, True, True, False], use_two_phase_prepare=True
    )
    two_phase_all_ready = simulate_dp_role_switch_protocol(
        4, "prefill", "decode", [True, True, True, True], use_two_phase_prepare=True
    )

    # vLLM DeepSeek-V4.1 target-token scoring replay row preservation
    dsv41_buggy_target_scoring = simulate_vllm_dsv41_replay_row_preservation(
        prompt_len=16,
        prompt_logprobs=None,
        prompt_logprob_token_ids=[101, 202, 303],
        include_fix=False,
    )
    dsv41_fixed_target_scoring = simulate_vllm_dsv41_replay_row_preservation(
        prompt_len=16,
        prompt_logprobs=None,
        prompt_logprob_token_ids=[101, 202, 303],
        include_fix=True,
    )
    dsv41_normal_decode = simulate_vllm_dsv41_replay_row_preservation(
        prompt_len=16,
        prompt_logprobs=None,
        prompt_logprob_token_ids=None,
        include_fix=True,
    )

    invariants = {
        "sglang_dp_and_attn_dp_admitted_without_local_broadcast": (
            cfg_dp4_attn_dp4 is True
            and cfg_local_bcast is False
            and len(errs_local_bcast) == 1
        ),
        "sglang_two_phase_prepare_prevents_dp_split_brain": (
            legacy_one_phase["split_brain"] is True
            and two_phase_preflight["split_brain"] is False
            and two_phase_preflight["worker_roles"] == ["prefill"] * 4
            and two_phase_all_ready["worker_roles"] == ["decode"] * 4
        ),
        "vllm_dsv41_target_token_scoring_preserves_all_prompt_rows": (
            dsv41_buggy_target_scoring["replayed_rows"] == 1
            and dsv41_buggy_target_scoring["target_positions_valid"] is False
            and dsv41_fixed_target_scoring["replayed_rows"] == 16
            and dsv41_fixed_target_scoring["target_positions_valid"] is True
            and dsv41_normal_decode["replayed_rows"] == 1
        ),
    }

    return {
        "experiment": "sglang_pd_dp_role_switch_vllm_dsv41_replay",
        "phase": 41,
        "priority": 50,
        "sglang_pd_dp_role_switch": {
            "dp4_attn_dp4_valid": cfg_dp4_attn_dp4,
            "dp4_local_control_broadcast_valid": cfg_local_bcast,
            "legacy_one_phase_partial_failure": legacy_one_phase,
            "two_phase_prepare_partial_failure": two_phase_preflight,
            "two_phase_prepare_all_ready": two_phase_all_ready,
        },
        "vllm_dsv41_replay_rows": {
            "buggy_target_scoring": dsv41_buggy_target_scoring,
            "fixed_target_scoring": dsv41_fixed_target_scoring,
            "normal_decode_no_logprobs": dsv41_normal_decode,
        },
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    for path in SUMMARY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment": summary["experiment"],
                    "invariants": summary["invariants"],
                }
            )
        )


if __name__ == "__main__":
    main()
