from __future__ import annotations
import json
from pathlib import Path


def evaluate_rlvr_verifier_sandbox_pipeline() -> dict:
    # Workload: 64x H100 GRPO/RLVR post-training step generating 16,384 code/math rollouts
    # (1,024 prompts x G=16 group size), each requiring unit-test execution + SymPy/Lean4 verification.
    tiers = [
        {
            "tier_id": "tier1_sync_end_of_rollout_docker_cli_spawn",
            "name": "Synchronous End-of-Rollout Batch Docker CLI Spawn (Barrier Wait After All 16K Rollouts)",
            "gpu_verifier_wait_stall_sec": 48.5,
            "step_end_to_end_sec": 92.0,
            "effective_gpu_utilization_pct": 41.8,
            "action": "Send to a human",
            "status": "fail",
            "note": "Waits until all 16,384 rollouts finish, then spawns heavy Docker containers; 64x H100 sit 100% idle for 48.5s per step!"
        },
        {
            "tier_id": "tier2_unbounded_host_subprocesses_pid_inode_exhaustion",
            "name": "Unbounded Local Host Subprocess Forking on GPU Nodes (PID / OverlayFS Inode Exhaustion)",
            "gpu_verifier_wait_stall_sec": 31.2,
            "step_end_to_end_sec": 74.5,
            "effective_gpu_utilization_pct": 51.6,
            "action": "Send to a human",
            "status": "fail",
            "note": "Forking thousands of Python/pytest workers directly on GPU hosts starves vLLM CPU event loops, hits cgroup PIDs_max, and leaks zombie processes!"
        },
        {
            "tier_id": "tier3_decoupled_cpu_ray_pool_batch_barrier",
            "name": "Decoupled Remote CPU Sandbox Pool (Pre-Warmed Workers) with Batch-Level Synchronization",
            "gpu_verifier_wait_stall_sec": 11.4,
            "step_end_to_end_sec": 54.8,
            "effective_gpu_utilization_pct": 70.1,
            "action": "Confirm",
            "status": "pass",
            "note": "Offloads execution to remote CPU nodes, but still waits for the slowest rollout tail before triggering reward verification."
        },
        {
            "tier_id": "tier4_streaming_per_sample_nsjail_sandbox_plus_group_ready_grpo_trigger",
            "name": "Streaming Sample-Ready Verification (nsjail/UserNS Sub-ms Pool) + Group-Ready GRPO Advantage Trigger",
            "gpu_verifier_wait_stall_sec": 1.6,
            "step_end_to_end_sec": 26.9,
            "effective_gpu_utilization_pct": 89.4,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Dispatches each rollout to pre-forked nsjail/UserNS CPU workers the instant EOS is hit, overlapping 96.7% of verification with ongoing GPU generation!"
        },
    ]
    return {
        "experiment_id": "llm_rlvr_code_math_verifier_sandbox_async_reward_pipeline_selection",
        "round": 300,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "cluster_gpus": "64x H100-80GB",
            "rollouts_per_step": 16384,
            "prompts_per_step": 1024,
            "grpo_group_size_g": 16
        },
        "tiers": tiers,
        "summary": {
            "verifier_stall_reduction_x": round(48.5 / 1.6, 2),
            "end_to_end_step_speedup_x": round(92.0 / 26.9, 2),
            "recommended_gpu_util_pct": 89.4,
            "recommended_tier": "tier4_streaming_per_sample_nsjail_sandbox_plus_group_ready_grpo_trigger"
        }
    }


if __name__ == "__main__":
    data = evaluate_rlvr_verifier_sandbox_pipeline()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
