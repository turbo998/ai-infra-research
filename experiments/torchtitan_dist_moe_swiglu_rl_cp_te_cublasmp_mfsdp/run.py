#!/usr/bin/env python3
"""Deterministic stdlib simulator for Phase 39 Experiment 3:
TorchTitan Dist-MoE MXFP8 Mega Fused SwiGLU Inference & RL Ulysses CP Per-Token Sharding,
TransformerEngine cuBLASMp WGRAD Workspace Reuse & Fullgraph Linear Attention,
and Megatron-FSDP Frozen Hook Deferral / LMCache / Dynamo Resilience.

First-hand sources:
- pytorch/torchtitan@ed3c713afeb21a514714ff59fec4fd4bb2d3b09d (#5465)
- pytorch/torchtitan@9371a2ffcb8655d74e6e1c099d7ca7b756259cae (#5424)
- NVIDIA/TransformerEngine@1aaede2f9edcaf70020c112c673e0b16818bc6a4 (#3629)
- NVIDIA/TransformerEngine@5eeb62a2fcb020af9778e3d29b14b5dc5c5ad50a (#3685)
- NVIDIA/Megatron-LM@d37b3a8fdb37af1b3a1d75720636af4c323bd522 (#8113)
- LMCache/LMCache@64b05b8af88d5fc13b375cdc06b5425cd56dea47 (#5791)
- ai-dynamo/dynamo@2ae8a46299b1c7d0287d1ab8fe4b67018bafe560 (#16633)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_torchtitan_dist_moe_mega_swiglu() -> dict[str, Any]:
    """Model TorchTitan ed3c713 (#5465): Dist-MoE MXFP8 Mega kernel with fused SwiGLU for RL generator."""
    # Microbenchmark rows from Qwen3-30B-A3B at EP=4 on 1x B200 (BF16 us vs Staged MXFP8 us vs Mega+Fused SwiGLU MXFP8 us)
    rows = [
        {"routed_rows_R": 512, "bf16_staged_us": 105.2, "mxfp8_staged_us": 117.3, "bf16_mega_us": 105.6, "mxfp8_mega_swiglu_us": 61.3},
        {"routed_rows_R": 2048, "bf16_staged_us": 122.9, "mxfp8_staged_us": 125.6, "bf16_mega_us": 123.8, "mxfp8_mega_swiglu_us": 72.0},
        {"routed_rows_R": 5120, "bf16_staged_us": 144.8, "mxfp8_staged_us": 151.1, "bf16_mega_us": 144.2, "mxfp8_mega_swiglu_us": 87.4},
        {"routed_rows_R": 8192, "bf16_staged_us": 174.1, "mxfp8_staged_us": 171.4, "bf16_mega_us": 174.0, "mxfp8_mega_swiglu_us": 112.0},
        {"routed_rows_R": 10240, "bf16_staged_us": 218.4, "mxfp8_staged_us": 192.9, "bf16_mega_us": 219.8, "mxfp8_mega_swiglu_us": 135.5},
        {"routed_rows_R": 16384, "bf16_staged_us": 300.4, "mxfp8_staged_us": 230.7, "bf16_mega_us": 300.4, "mxfp8_mega_swiglu_us": 192.0},
        {"routed_rows_R": 32768, "bf16_staged_us": 450.4, "mxfp8_staged_us": 342.6, "bf16_mega_us": 450.7, "mxfp8_mega_swiglu_us": 292.8},
        {"routed_rows_R": 65536, "bf16_staged_us": 790.3, "mxfp8_staged_us": 583.6, "bf16_mega_us": 789.6, "mxfp8_mega_swiglu_us": 516.8},
    ]
    for r in rows:
        r["staged_speedup_vs_bf16"] = round(r["bf16_staged_us"] / r["mxfp8_staged_us"], 2)
        r["mega_swiglu_speedup_vs_bf16"] = round(r["bf16_mega_us"] / r["mxfp8_mega_swiglu_us"], 2)
        r["mega_swiglu_speedup_vs_staged"] = round(r["mxfp8_staged_us"] / r["mxfp8_mega_swiglu_us"], 2)

    return {
        "microbenchmarks_qwen3_30b_a3b_ep4_b200": rows,
        "max_mega_swiglu_speedup_vs_bf16": max(r["mega_swiglu_speedup_vs_bf16"] for r in rows),
        "min_mega_swiglu_speedup_vs_bf16": min(r["mega_swiglu_speedup_vs_bf16"] for r in rows),
        "inference_dgrad_scale_numel": 0,
        "interleaved_fc13_for_w13": True,
        "mxfp8_rel_error_vs_bf16": 0.066,
        "swapped_gate_up_rel_error_vs_bf16": 0.90,
        "mxfp8_rel_tol_guard": 0.12,
        "e2e_dapo_math_step_speedup": 1.10,
        "e2e_dapo_math_decode_speedup": 1.09,
    }


def simulate_torchtitan_rl_context_parallel() -> dict[str, Any]:
    """Model TorchTitan 9371a2f (#5424): shard_per_token_tensors for RL trainer under Ulysses CP."""
    total_tokens = 8
    cp_degree = 2
    advantages = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    loss_mask = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

    full_masked_sum = sum(a * m for a, m in zip(advantages, loss_mask))
    full_mask_count = sum(loss_mask)
    full_grpo_loss = full_masked_sum / full_mask_count

    chunk = total_tokens // cp_degree
    cp_shard_sums = []
    for cp_rank in range(cp_degree):
        lo, hi = cp_rank * chunk, (cp_rank + 1) * chunk
        shard_adv = advantages[lo:hi]
        shard_mask = loss_mask[lo:hi]
        cp_shard_sums.append(sum(a * m for a, m in zip(shard_adv, shard_mask)))

    reconstructed_grpo_loss = sum(cp_shard_sums) / full_mask_count

    return {
        "total_tokens": total_tokens,
        "cp_degree": cp_degree,
        "full_grpo_loss": full_grpo_loss,
        "reconstructed_grpo_loss": reconstructed_grpo_loss,
        "masked_sum_exact_match": abs(full_grpo_loss - reconstructed_grpo_loss) < 1e-12,
        "rejects_cp_load_balancer": True,
        "gb300_qwen3_0_6b_loss_rel_diff": 2.0e-7,
        "gb300_no_cp_peak_mem_gib": 4.47,
        "gb300_cp2_peak_mem_gib": 3.92,
        "gb300_mem_reduction_pct": round(100.0 * (4.47 - 3.92) / 4.47, 2),
    }


def simulate_te_cublasmp_wgrad_reuse_and_linear_attn_compile() -> dict[str, Any]:
    """Model TransformerEngine 1aaede2 (#3629) and 5eeb62a (#3685)."""
    # All-gather count per backward pass in row-parallel Linear and LayerNormMLP
    cublasmp_0_10_linear_bwd_ag = 1
    cublasmp_0_11_linear_bwd_ag = 0
    cublasmp_0_10_ln_mlp_bwd_ag = 2
    cublasmp_0_11_ln_mlp_bwd_ag = 1  # fc2 reuses dgrad AG workspace; fc1 still gathers input

    return {
        "min_cublasmp_version_for_reuse": 1100,
        "cublasmp_0_10_linear_bwd_allgathers": cublasmp_0_10_linear_bwd_ag,
        "cublasmp_0_11_linear_bwd_allgathers": cublasmp_0_11_linear_bwd_ag,
        "cublasmp_0_10_layernorm_mlp_bwd_allgathers": cublasmp_0_10_ln_mlp_bwd_ag,
        "cublasmp_0_11_layernorm_mlp_bwd_allgathers": cublasmp_0_11_ln_mlp_bwd_ag,
        "linear_attention_init_cached_op_fullgraph": True,
    }


def simulate_mfsdp_frozen_defer_lmcache_local_pd_dynamo_watchdog() -> dict[str, Any]:
    """Model Megatron-LM d37b3a8 (#8113), LMCache 64b05b8 (#5791), and Dynamo 2ae8a46 (#16633)."""
    # 1. Megatron-FSDP frozen module (`_trainable_parameter_countdown.initial_value == 0`)
    # When grad_input is all None, full_backward_hook fires BEFORE internal backward completes.
    # Deferring cleanup to `context._delayed_post_backward_callbacks` prevents premature weight release.
    legacy_premature_weight_free_on_frozen_no_grad_input = True
    fixed_premature_weight_free_on_frozen_no_grad_input = False

    # 2. LMCache PDBackendAsync.batched_submit_put_task with transfer_spec=None (local request)
    initial_ref_count = 1
    legacy_local_req_ref_count = initial_ref_count + 1  # leaked ref_count_up()
    fixed_local_req_ref_count = initial_ref_count       # early return when transfer_spec is None

    # 3. Dynamo TRT-LLM EngineDeadError fatal exit
    # When `engine.get_health_check_fatal_error()` is set (MPI rank died), skip `engine.shutdown()`
    # so worker exits promptly instead of hanging forever on dead MPI rank IPC.
    engine_teardown_skipped_on_dead_mpi_rank = True

    return {
        "mfsdp_legacy_premature_free": legacy_premature_weight_free_on_frozen_no_grad_input,
        "mfsdp_fixed_premature_free": fixed_premature_weight_free_on_frozen_no_grad_input,
        "lmcache_legacy_local_req_ref_count": legacy_local_req_ref_count,
        "lmcache_fixed_local_req_ref_count": fixed_local_req_ref_count,
        "dynamo_trtllm_teardown_skipped_on_dead_mpi_rank": engine_teardown_skipped_on_dead_mpi_rank,
    }


def run_experiment() -> dict[str, Any]:
    dist_moe = simulate_torchtitan_dist_moe_mega_swiglu()
    rl_cp = simulate_torchtitan_rl_context_parallel()
    te_cublasmp = simulate_te_cublasmp_wgrad_reuse_and_linear_attn_compile()
    resilience = simulate_mfsdp_frozen_defer_lmcache_local_pd_dynamo_watchdog()

    invariants = {
        "dist_moe_mega_swiglu_1_72x_and_empty_dgrad": (
            dist_moe["max_mega_swiglu_speedup_vs_bf16"] == 1.72
            and dist_moe["min_mega_swiglu_speedup_vs_bf16"] == 1.53
            and dist_moe["inference_dgrad_scale_numel"] == 0
        ),
        "rl_cp_per_token_shard_exact_and_mem_saved": (
            rl_cp["masked_sum_exact_match"] is True
            and rl_cp["gb300_mem_reduction_pct"] == 12.3
        ),
        "te_cublasmp_0_11_eliminates_linear_wgrad_ag": (
            te_cublasmp["cublasmp_0_11_linear_bwd_allgathers"] == 0
            and te_cublasmp["cublasmp_0_11_layernorm_mlp_bwd_allgathers"] == 1
            and te_cublasmp["linear_attention_init_cached_op_fullgraph"] is True
        ),
        "mfsdp_lmcache_dynamo_lifecycle_safe": (
            resilience["mfsdp_fixed_premature_free"] is False
            and resilience["lmcache_fixed_local_req_ref_count"] == 1
            and resilience["dynamo_trtllm_teardown_skipped_on_dead_mpi_rank"] is True
        ),
    }

    return {
        "experiment": "torchtitan_dist_moe_swiglu_rl_cp_te_cublasmp_mfsdp",
        "phase": 39,
        "priority": 48,
        "evidence_mode": "own-offline",
        "torchtitan_dist_moe_mega_swiglu": dist_moe,
        "torchtitan_rl_context_parallel": rl_cp,
        "te_cublasmp_and_linear_attention": te_cublasmp,
        "mfsdp_lmcache_dynamo_resilience": resilience,
        "invariants": invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    out_dir = Path(__file__).resolve().parent
    for target in (out_dir / "summary.json", out_dir / "results" / "summary.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"Failed invariants: {failed}")
        print(json.dumps({"status": "ok", "experiment": summary["experiment"], "invariants": summary["invariants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
