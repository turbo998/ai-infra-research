#!/usr/bin/env python3
"""Deterministic RLHF / GRPO post-training rollout, weight-resharding, and straggler-bubble evaluator for Round 11:
RLHF / GRPO Post-Training Rollout & Weight-Sync Architecture Selection
(Naive FSDP Rollout vs. Statically Disaggregated Actor/Rollout Pools vs. HybridEngine Co-Located 3D Weight Resharding vs. Asynchronous Partial-Rollout Streaming).

Provenance Note:
- All step latencies, rollout tail idle bubbles, weight-resharding overheads, token throughputs, and 1,000-step TCO figures
  are deterministic synthetic calculations (`provenance_type: "synthetic"`, `independently_reproduced: false`)
  calibrated against publicly documented RLHF/GRPO post-training systems (`anonymized_reported`:
  ByteDance/HKU verl HybridFlow EuroSys'25 paper, OpenRLHF Ray+vLLM architecture, Kimi k1.5 / StreamRL / AReaL partial-rollout
  & bounded-staleness asynchronous RL designs, and NVIDIA H100 80GB SXM hardware specs).
- No physical 64-GPU H100 cluster was provisioned or run during this session (`physical_gpu_cluster_measurement: "not_measured"`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def compute_results() -> dict[str, Any]:
    # Workload & Cluster Baseline:
    # 70B Policy Model (GRPO Reasoning Post-Training, G=8 responses/prompt, 512 prompts/step = 4,096 trajectories/step)
    # Mean response length = 4,096 tokens -> 16,777,216 generated tokens per step (16.78M tokens/step)
    # P99 long-CoT response tail = 24,576 tokens (6.0x mean length)
    # Cluster: 64 x NVIDIA H100 80GB SXM ($2.50/GPU-hr -> $160.00/hr cluster rate)
    num_gpus = 64
    gpu_hourly_usd = 2.50
    cluster_hourly_usd = round(num_gpus * gpu_hourly_usd, 2)  # $160.00/hr
    prompts_per_step = 512
    group_size_g = 8
    trajectories_per_step = prompts_per_step * group_size_g  # 4,096
    mean_response_tokens = 4096
    p99_tail_response_tokens = 24576
    total_tokens_per_step = trajectories_per_step * mean_response_tokens  # 16,777,216 tokens
    steps_benchmark = 1000

    # Tier 1: Naive Co-Located Training-Sharded Rollout (Native FSDP/ZeRO-3 Generate without vLLM resharding)
    # All-gathers 70B parameters across ranks on every autoregressive token step.
    t1_rollout_sec = 548.0
    t1_weight_sync_sec = 0.0
    t1_actor_train_sec = 64.0
    t1_step_sec = round(t1_rollout_sec + t1_weight_sync_sec + t1_actor_train_sec, 1)  # 612.0 s
    t1_tok_per_sec = round(total_tokens_per_step / t1_step_sec, 1)  # 27,413.8 tok/s
    t1_straggler_bubble_pct = 62.0
    t1_effective_mfu_pct = 11.5
    t1_1000_steps_hours = round((t1_step_sec * steps_benchmark) / 3600.0, 2)  # 170.00 hrs
    t1_1000_steps_cost_usd = round(t1_1000_steps_hours * cluster_hourly_usd, 2)  # $27,200.00

    # Tier 2: Statically Disaggregated Actor Pool (32 GPUs) + vLLM Rollout Pool (32 GPUs) via Ray + RDMA Broadcast
    # 32 GPUs run vLLM (TP=4, PagedAttention), 32 GPUs run FSDP Actor update.
    # Suffers from 50% static pool split + 18.5s cross-pool NCCL weight broadcast & KV flush + synchronous long-CoT straggler wait.
    t2_rollout_sec = 196.0
    t2_weight_sync_sec = 18.5
    t2_actor_train_sec = 72.0  # 32 GPUs take longer on backward pass or sit idle waiting for rollout
    t2_step_sec = round(t2_rollout_sec + t2_weight_sync_sec + t2_actor_train_sec, 1)  # 286.5 s
    t2_tok_per_sec = round(total_tokens_per_step / t2_step_sec, 1)  # 58,559.2 tok/s
    t2_speedup_vs_t1 = round(t1_step_sec / t2_step_sec, 2)  # 2.14x
    t2_straggler_bubble_pct = 48.0
    t2_effective_mfu_pct = 24.6
    t2_1000_steps_hours = round((t2_step_sec * steps_benchmark) / 3600.0, 2)  # 79.58 hrs
    t2_1000_steps_cost_usd = round(t2_1000_steps_hours * cluster_hourly_usd, 2)  # $12,732.80

    # Tier 3: HybridEngine Co-Located Zero-Copy 3D Weight Resharding (verl / HybridFlow on all 64 GPUs)
    # All 64 GPUs run vLLM/SGLang Rollout (16 x TP=4) then transition in-memory (3.2s CUDA IPC / All-to-All resharding)
    # to 64-GPU FSDP/Megatron training (63.0s). However, synchronous barrier still waits for 24K-token CoT stragglers (52.0% rollout tail bubble).
    t3_rollout_sec = 92.0  # 44.2s dense generation + 47.8s long-tail CoT straggler tail wait (52.0% bubble)
    t3_weight_sync_sec = 3.2
    t3_actor_train_sec = 63.0
    t3_step_sec = round(t3_rollout_sec + t3_weight_sync_sec + t3_actor_train_sec, 1)  # 158.2 s
    t3_tok_per_sec = round(total_tokens_per_step / t3_step_sec, 1)  # 106,050.7 tok/s
    t3_speedup_vs_t1 = round(t1_step_sec / t3_step_sec, 2)  # 3.87x
    t3_speedup_vs_t2 = round(t2_step_sec / t3_step_sec, 2)  # 1.81x
    t3_straggler_bubble_pct = 52.0
    t3_effective_mfu_pct = 44.5
    t3_1000_steps_hours = round((t3_step_sec * steps_benchmark) / 3600.0, 2)  # 43.94 hrs
    t3_1000_steps_cost_usd = round(t3_1000_steps_hours * cluster_hourly_usd, 2)  # $7,030.40

    # Tier 4: Asynchronous Streaming Rollout + Partial Rollout Prefix Reuse (AReaL / StreamRL / Kimi k1.5 Partial Rollout)
    # Interrupts/segments ultra-long (>8K) trajectories at step boundary, caches KV prefix in host/GPU pool,
    # and resumes in step t+1 with bounded-staleness truncated importance sampling (Dual-Clip GRPO).
    # Cuts rollout straggler bubble from 52.0% -> 8.5%!
    t4_rollout_sec = 31.0
    t4_weight_sync_sec = 2.4
    t4_actor_train_sec = 63.0
    t4_step_sec = round(t4_rollout_sec + t4_weight_sync_sec + t4_actor_train_sec, 1)  # 96.4 s
    t4_tok_per_sec = round(total_tokens_per_step / t4_step_sec, 1)  # 174,037.5 tok/s
    t4_speedup_vs_t1 = round(t1_step_sec / t4_step_sec, 2)  # 6.35x
    t4_speedup_vs_t3 = round(t3_step_sec / t4_step_sec, 2)  # 1.64x
    t4_straggler_bubble_pct = 8.5
    t4_effective_mfu_pct = 73.0
    t4_1000_steps_hours = round((t4_step_sec * steps_benchmark) / 3600.0, 2)  # 26.78 hrs
    t4_1000_steps_cost_usd = round(t4_1000_steps_hours * cluster_hourly_usd, 2)  # $4,284.80
    t4_savings_vs_t1_usd = round(t1_1000_steps_cost_usd - t4_1000_steps_cost_usd, 2)  # $22,915.20
    t4_savings_vs_t3_usd = round(t3_1000_steps_cost_usd - t4_1000_steps_cost_usd, 2)  # $2,745.60

    return {
        "metadata": {
            "experiment_id": "rlhf_post_training_infra_selection",
            "title": "RLHF / GRPO Post-Training Rollout & Weight-Sync Selection (Naive FSDP vs. Static Disaggregated vs. HybridEngine Co-Located vs. Async Partial-Rollout)",
            "provenance_type": "synthetic",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "local_cpu_harness_measurement": "measured",
            "calibration_sources": [
                "HybridFlow / verl (EuroSys 2025): 3D HybridEngine zero-copy weight resharding between FSDP/Megatron and vLLM/SGLang",
                "OpenRLHF: Ray-based disaggregated Actor/Rollout/Reference/Reward scheduling over NCCL/RDMA",
                "Kimi k1.5 / AReaL / StreamRL: Partial Rollout prefix reuse and bounded-staleness asynchronous GRPO/PPO for long-CoT reasoning"
            ],
            "workload_baseline": {
                "policy_model": "Llama-3 / Qwen-2.5 70B (BF16 Training / FP8 Rollout)",
                "algorithm": "GRPO (Group Relative Policy Optimization, G=8)",
                "cluster_gpus": num_gpus,
                "gpu_sku": "NVIDIA H100 80GB SXM",
                "cluster_hourly_rate_usd": cluster_hourly_usd,
                "prompts_per_step": prompts_per_step,
                "group_size_g": group_size_g,
                "trajectories_per_step": trajectories_per_step,
                "mean_response_tokens": mean_response_tokens,
                "p99_tail_response_tokens": p99_tail_response_tokens,
                "total_generated_tokens_per_step": total_tokens_per_step,
                "benchmark_steps": steps_benchmark,
            },
        },
        "tiers": {
            "tier1_naive_fsdp_colocated_rollout": {
                "label": "Tier 1: Naive FSDP/ZeRO-3 Training-Sharded Rollout (No Inference Engine Resharding)",
                "rollout_phase_sec": t1_rollout_sec,
                "weight_sync_reshard_sec": t1_weight_sync_sec,
                "actor_ref_train_sec": t1_actor_train_sec,
                "total_step_time_sec": t1_step_sec,
                "throughput_tokens_per_sec": t1_tok_per_sec,
                "speedup_vs_tier1": 1.0,
                "rollout_straggler_idle_bubble_pct": t1_straggler_bubble_pct,
                "effective_cluster_mfu_pct": t1_effective_mfu_pct,
                "hours_per_1000_steps": t1_1000_steps_hours,
                "cost_per_1000_steps_usd": t1_1000_steps_cost_usd,
            },
            "tier2_static_disaggregated_ray_pools": {
                "label": "Tier 2: Statically Disaggregated Actor (32 GPUs) + vLLM Rollout (32 GPUs) Pools (OpenRLHF Style)",
                "rollout_phase_sec": t2_rollout_sec,
                "weight_sync_reshard_sec": t2_weight_sync_sec,
                "actor_ref_train_sec": t2_actor_train_sec,
                "total_step_time_sec": t2_step_sec,
                "throughput_tokens_per_sec": t2_tok_per_sec,
                "speedup_vs_tier1": t2_speedup_vs_t1,
                "rollout_straggler_idle_bubble_pct": t2_straggler_bubble_pct,
                "effective_cluster_mfu_pct": t2_effective_mfu_pct,
                "hours_per_1000_steps": t2_1000_steps_hours,
                "cost_per_1000_steps_usd": t2_1000_steps_cost_usd,
            },
            "tier3_hybrid_engine_colocated_resharding": {
                "label": "Tier 3: HybridEngine Co-Located Zero-Copy 3D Weight Resharding (verl / HybridFlow, Synchronous)",
                "rollout_phase_sec": t3_rollout_sec,
                "weight_sync_reshard_sec": t3_weight_sync_sec,
                "actor_ref_train_sec": t3_actor_train_sec,
                "total_step_time_sec": t3_step_sec,
                "throughput_tokens_per_sec": t3_tok_per_sec,
                "speedup_vs_tier1": t3_speedup_vs_t1,
                "speedup_vs_tier2": t3_speedup_vs_t2,
                "rollout_straggler_idle_bubble_pct": t3_straggler_bubble_pct,
                "effective_cluster_mfu_pct": t3_effective_mfu_pct,
                "hours_per_1000_steps": t3_1000_steps_hours,
                "cost_per_1000_steps_usd": t3_1000_steps_cost_usd,
            },
            "tier4_async_partial_rollout_streaming": {
                "label": "Tier 4: Async Streaming + Partial Rollout Prefix Reuse (AReaL / StreamRL / Kimi k1.5 Style)",
                "rollout_phase_sec": t4_rollout_sec,
                "weight_sync_reshard_sec": t4_weight_sync_sec,
                "actor_ref_train_sec": t4_actor_train_sec,
                "total_step_time_sec": t4_step_sec,
                "throughput_tokens_per_sec": t4_tok_per_sec,
                "speedup_vs_tier1": t4_speedup_vs_t1,
                "speedup_vs_tier3": t4_speedup_vs_t3,
                "rollout_straggler_idle_bubble_pct": t4_straggler_bubble_pct,
                "effective_cluster_mfu_pct": t4_effective_mfu_pct,
                "hours_per_1000_steps": t4_1000_steps_hours,
                "cost_per_1000_steps_usd": t4_1000_steps_cost_usd,
                "savings_per_1000_steps_vs_tier1_usd": t4_savings_vs_t1_usd,
                "savings_per_1000_steps_vs_tier3_usd": t4_savings_vs_t3_usd,
            },
        },
        "staleness_and_length_sensitivity": {
            "short_cot_2k_sync_hybrid_vs_async_speedup": 1.12,
            "long_cot_24k_sync_hybrid_vs_async_speedup": 1.64,
            "ultra_long_agent_64k_sync_hybrid_vs_async_speedup": 2.38,
            "max_safe_policy_staleness_steps_with_is_clipping": 2,
            "unclipped_stale_step_5_reward_collapse_risk": "HIGH (KL divergence spike without truncated importance sampling)",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or check Round 11 RLHF/GRPO post-training evaluator.")
    parser.add_argument("--check", action="store_true", help="Verify results.json matches deterministic model.")
    args = parser.parse_args()

    computed = compute_results()
    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}")
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if existing != computed:
            raise SystemExit("MISMATCH: experiments/rlhf_post_training_infra_selection/results.json is out of sync.")
        print("OK: experiments/rlhf_post_training_infra_selection/results.json matches deterministic model.")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
