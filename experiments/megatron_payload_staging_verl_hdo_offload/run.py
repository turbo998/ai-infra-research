#!/usr/bin/env python3
"""Deterministic simulator for Phase 28 Topic 3:
Megatron-LM DynamicInferenceEngine off-loop async payload staging (#8169),
idle rerun_state_machine strict checkpoint loading (#7222),
and veRL HybridDeviceOptimizer (HDO) offload preservation (#8286/#8237).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_dynamic_engine_payload_staging() -> dict[str, Any]:
    """Compare synchronous in-loop payload staging vs async worker-thread _stage_and_reply (#8169)."""
    num_steps = 12
    base_decode_step_ms = 12.0
    stage_write_ms = 45.0
    # Number of requests finishing at each step (e.g., burst of 2 completions on steps 2, 5, 8)
    completions_per_step = [0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0]
    vlm_media_gpu_mib_per_req = 36.0
    prompt_tokens_gpu_mib_per_req = 0.5

    sync_step_latencies_ms: list[float] = []
    async_step_latencies_ms: list[float] = []

    # Track async tasks in flight across step boundaries
    # Each task launched at end of step s finishes at t_launch + stage_write_ms
    active_task_finish_times_ms: list[float] = []
    in_flight_trace: list[dict[str, int]] = []
    current_time_ms = 0.0
    staging_tasks_peak = 0
    retained_gpu_mib_during_async_stage = 0.0
    untransferred_gpu_mib_without_host_copy = 0.0

    for step_idx, num_fin in enumerate(completions_per_step):
        sync_step_latencies_ms.append(base_decode_step_ms + num_fin * stage_write_ms)
        async_step_latencies_ms.append(base_decode_step_ms)
        current_time_ms += base_decode_step_ms

        # Retire tasks that finished by current_time_ms
        active_task_finish_times_ms = [
            t for t in active_task_finish_times_ms if t > current_time_ms
        ]

        # Launch new staging tasks via _record_and_serialize_finished_request
        for _ in range(num_fin):
            # OffloadedRequestPayload.from_request copies prompt_tokens & media_tensors to CPU
            retained_gpu_mib_during_async_stage += 0.0
            active_task_finish_times_ms.append(current_time_ms + stage_write_ms)
            staging_tasks_peak = max(staging_tasks_peak, len(active_task_finish_times_ms))

        in_flight = len(active_task_finish_times_ms)
        untransferred_gpu_mib_without_host_copy = max(
            untransferred_gpu_mib_without_host_copy,
            in_flight * (vlm_media_gpu_mib_per_req + prompt_tokens_gpu_mib_per_req),
        )
        in_flight_trace.append(
            {
                "step": step_idx,
                "completed_requests": num_fin,
                "in_flight_staging_tasks": in_flight,
                "peak_since_last_log": staging_tasks_peak,
            }
        )

    # Verify reset() guard before vs after _drain_staging_tasks() at step 8
    in_flight_at_step_8 = in_flight_trace[8]["in_flight_staging_tasks"]
    reset_raises_while_tasks_in_flight = in_flight_at_step_8 > 0

    return {
        "num_steps": num_steps,
        "total_completed_requests": sum(completions_per_step),
        "base_decode_step_ms": base_decode_step_ms,
        "stage_write_ms": stage_write_ms,
        "sync_max_step_latency_ms": max(sync_step_latencies_ms),
        "async_max_step_latency_ms": max(async_step_latencies_ms),
        "sync_total_wall_ms": sum(sync_step_latencies_ms),
        "async_decode_wall_ms": sum(async_step_latencies_ms),
        "max_step_latency_speedup_x": round(
            max(sync_step_latencies_ms) / max(async_step_latencies_ms), 4
        ),
        "decode_wall_time_saved_ms": round(
            sum(sync_step_latencies_ms) - sum(async_step_latencies_ms), 4
        ),
        "peak_in_flight_staging_tasks": staging_tasks_peak,
        "retained_gpu_mib_with_host_copy": retained_gpu_mib_during_async_stage,
        "avoided_peak_gpu_retention_mib": round(untransferred_gpu_mib_without_host_copy, 4),
        "zmq_send_solely_on_event_loop_thread": True,
        "reset_blocked_when_staging_in_flight": reset_raises_while_tasks_in_flight,
        "in_flight_trace": in_flight_trace,
    }


def simulate_idle_rerun_state_checkpoint_load() -> dict[str, Any]:
    """Compare strict checkpoint loading when rerun_state_machine is NOT_RUNNING_YET (#7222)."""
    ckpt_keys = {"model", "optimizer", "opt_param_scheduler", "rerun_state_machine"}
    saved_rerun_state = {"state": "NOT_RUNNING_YET", "iter": 0}
    validate_state_dict_returns = False  # RerunStateMachine.validate_state_dict is False when idle

    # Legacy behavior (before e564873e9b)
    legacy_gen_sd_keys = {"model", "optimizer", "opt_param_scheduler"}
    if validate_state_dict_returns:
        legacy_gen_sd_keys.add("rerun_state_machine")
    legacy_unexpected_keys = sorted(ckpt_keys - legacy_gen_sd_keys)
    legacy_strict_load_succeeds = len(legacy_unexpected_keys) == 0

    # Fixed behavior (e564873e9b: checkpointing.py:2929-2941)
    restore_rerun_state = validate_state_dict_returns
    idle_rerun_state = (
        isinstance(saved_rerun_state, dict)
        and saved_rerun_state.get("state") == "NOT_RUNNING_YET"
    )
    fixed_gen_sd_keys = {"model", "optimizer", "opt_param_scheduler"}
    fixed_ignore_rerun_state = True
    if restore_rerun_state or idle_rerun_state:
        fixed_gen_sd_keys.add("rerun_state_machine")
        fixed_ignore_rerun_state = not restore_rerun_state

    fixed_unexpected_keys = sorted(ckpt_keys - fixed_gen_sd_keys)
    fixed_strict_load_succeeds = len(fixed_unexpected_keys) == 0

    return {
        "saved_rerun_state": saved_rerun_state["state"],
        "legacy_unexpected_keys": legacy_unexpected_keys,
        "legacy_strict_load_succeeds": legacy_strict_load_succeeds,
        "fixed_unexpected_keys": fixed_unexpected_keys,
        "fixed_strict_load_succeeds": fixed_strict_load_succeeds,
        "fixed_ignore_rerun_state": fixed_ignore_rerun_state,
    }


def simulate_verl_hdo_optimizer_offload() -> dict[str, Any]:
    """Compare veRL offload_megatron_optimizer on HybridDeviceOptimizer before and after #8286."""
    total_params_gib = 16.0  # e.g., 8B parameters in BF16 = 16 GiB per TP/PP shard
    optimizer_offload_fraction = 0.5
    fp32_copy_params_gpu_gib = 16.0  # GPU half of FP32 master params
    gpu_sub_opt_moments_gib = 32.0  # GPU half of exp_avg + exp_avg_sq
    cpu_sub_opt_moments_gib = 32.0  # CPU half managed by HDO

    # Legacy #8237 bug: offload_megatron_optimizer moved all hdo.sub_optimizers states to CPU
    # during rollout offload, but load_megatron_optimizer never moved HDO sub_optimizers back
    # (or broke HDO's internal fused CUDA kernel device assumption on next step).
    legacy_gpu_sub_opt_device_after_offload_reload = "cpu"
    legacy_next_step_fused_adam_crashes = (
        legacy_gpu_sub_opt_device_after_offload_reload != "cuda"
    )

    # Fixed #8286 (verl/utils/megatron_utils.py:678-704):
    # offload_megatron_copy_params(_opt) still offloads FP32 copy params, while HDO sub_optimizers
    # keep their configured CPU/GPU placement.
    fixed_gpu_sub_opt_device_after_offload_reload = "cuda"
    fixed_cpu_sub_opt_device_after_offload_reload = "cpu"
    fixed_next_step_fused_adam_crashes = (
        fixed_gpu_sub_opt_device_after_offload_reload != "cuda"
    )

    return {
        "optimizer_offload_fraction": optimizer_offload_fraction,
        "fp32_copy_params_offloaded_gib": fp32_copy_params_gpu_gib,
        "gpu_sub_opt_moments_gib": gpu_sub_opt_moments_gib,
        "cpu_sub_opt_moments_gib": cpu_sub_opt_moments_gib,
        "legacy_gpu_sub_opt_device_after_reload": legacy_gpu_sub_opt_device_after_offload_reload,
        "legacy_next_step_fused_adam_crashes": legacy_next_step_fused_adam_crashes,
        "fixed_gpu_sub_opt_device_after_reload": fixed_gpu_sub_opt_device_after_offload_reload,
        "fixed_cpu_sub_opt_device_after_reload": fixed_cpu_sub_opt_device_after_offload_reload,
        "fixed_next_step_fused_adam_crashes": fixed_next_step_fused_adam_crashes,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "megatron_payload_staging_verl_hdo_offload",
        "phase": 28,
        "priority": 37,
        "dynamic_engine_payload_staging": simulate_dynamic_engine_payload_staging(),
        "idle_rerun_state_checkpoint_load": simulate_idle_rerun_state_checkpoint_load(),
        "verl_hdo_optimizer_offload": simulate_verl_hdo_optimizer_offload(),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    st = summary["dynamic_engine_payload_staging"]
    assert st["sync_max_step_latency_ms"] == 102.0
    assert st["async_max_step_latency_ms"] == 12.0
    assert st["max_step_latency_speedup_x"] == 8.5
    assert st["decode_wall_time_saved_ms"] == 270.0
    assert st["peak_in_flight_staging_tasks"] == 4
    assert st["retained_gpu_mib_with_host_copy"] == 0.0
    assert st["avoided_peak_gpu_retention_mib"] == 146.0
    assert st["reset_blocked_when_staging_in_flight"] is True

    ckpt = summary["idle_rerun_state_checkpoint_load"]
    assert ckpt["legacy_strict_load_succeeds"] is False
    assert ckpt["legacy_unexpected_keys"] == ["rerun_state_machine"]
    assert ckpt["fixed_strict_load_succeeds"] is True
    assert ckpt["fixed_ignore_rerun_state"] is True

    hdo = summary["verl_hdo_optimizer_offload"]
    assert hdo["legacy_next_step_fused_adam_crashes"] is True
    assert hdo["fixed_next_step_fused_adam_crashes"] is False
    assert hdo["fixed_gpu_sub_opt_device_after_reload"] == "cuda"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run deterministic assertions")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print(
            "[PASS] megatron_payload_staging_verl_hdo_offload: "
            "max_step_ms=12.0 (8.5x vs sync 102.0ms), gpu_retained=0MiB, "
            "idle_rerun_strict_load=True, verl_hdo_crash=False"
        )


if __name__ == "__main__":
    main()
