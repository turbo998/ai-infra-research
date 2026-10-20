#!/usr/bin/env python3
"""Deterministic offline model for Phase 34 Topic 1:
1. vLLM FlashInfer DCP prefill trtllm-gen workspace sizing & empty-shard zero/LSE masking (vllm@7b569ec895)
2. vLLM NCCL symmetric-memory collective scratch output cloning vs alias overwrite (vllm@8122776fba)
3. Megatron-LM attention & YaRN MLA softmax_scale 3-tier precedence (Megatron-LM@0156c21373)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def simulate_trtllm_dcp_prefill(
    num_qo_heads_per_rank: int = 8,
    dcp_world_size: int = 8,
    workspace_mib: int = 8,
) -> dict[str, Any]:
    workspace_bytes = workspace_mib * (1 << 20)
    safety_margin_bytes = 1 << 20  # 1 MiB carve-out in fits_workspace
    total_qo_heads = num_qo_heads_per_rank * dcp_world_size

    scenarios = [
        {"name": "B=4, max_q=1024", "batch_size": 4, "max_q_len": 1024},
        {"name": "B=8, max_q=2048", "batch_size": 8, "max_q_len": 2048},
        {"name": "B=8, max_q=3600", "batch_size": 8, "max_q_len": 3600},
        {"name": "B=16, max_q=2048", "batch_size": 16, "max_q_len": 2048},
    ]

    workspace_checks = []
    for sc in scenarios:
        b = sc["batch_size"]
        q_aligned = round_up(sc["max_q_len"], 256)
        stats_bytes = total_qo_heads * b * q_aligned * 8  # float2 = 8 bytes
        required_bytes = stats_bytes + safety_margin_bytes
        fits = required_bytes <= workspace_bytes
        workspace_checks.append(
            {
                "name": sc["name"],
                "batch_size": b,
                "max_q_len": sc["max_q_len"],
                "q_len_rounded_256": q_aligned,
                "stats_bytes": stats_bytes,
                "required_bytes": required_bytes,
                "workspace_bytes": workspace_bytes,
                "fits_trtllm_workspace": fits,
                "selected_wrapper": "TrtllmDCPContextWrapper" if fits else "BatchPrefillWithPagedKVCacheWrapper",
            }
        )

    # Empty local context shard combine simulation across 4 DCP ranks:
    # Suppose rank 0..2 have valid context (seq_len > 0) and rank 3 has seq_len == 0 (empty local shard).
    # Uninitialized empty_like buffer on rank 3 contains garbage/NaN (e.g. float('nan')) and unmasked lse = +80.0.
    rank_seq_lens = [512, 512, 256, 0]
    rank_true_out = [1.20, 0.80, 1.50, float("nan")]
    rank_true_lse = [4.0, 4.0, 3.3068528194400546, 85.0]  # garbage LSE on empty shard if unmasked

    # Legacy buggy path: torch.empty_like(q) + unmasked LSE on seq_len == 0
    max_lse_buggy = max(rank_true_lse)
    weights_buggy = [math.exp(l - max_lse_buggy) for l in rank_true_lse]
    denom_buggy = sum(weights_buggy)
    combined_buggy = sum(w * v for w, v in zip(weights_buggy, rank_true_out)) / denom_buggy
    buggy_is_nan = math.isnan(combined_buggy)

    # Fixed path (vllm@7b569ec895): out = torch.zeros_like(q) when _has_empty_ctx and mask_dcp_empty_shards_ sets lse = -inf
    rank_fixed_out = [1.20, 0.80, 1.50, 0.0]
    rank_fixed_lse = [
        lse if seq_len > 0 else float("-inf")
        for lse, seq_len in zip(rank_true_lse, rank_seq_lens)
    ]
    max_lse_fixed = max(rank_fixed_lse)
    weights_fixed = [0.0 if l == float("-inf") else math.exp(l - max_lse_fixed) for l in rank_fixed_lse]
    denom_fixed = sum(weights_fixed)
    combined_fixed = sum(w * v for w, v in zip(weights_fixed, rank_fixed_out)) / denom_fixed

    # Exact reference over non-empty shards (ranks 0..2, weights [1.0, 1.0, 0.5])
    ref_combined = (1.0 * 1.20 + 1.0 * 0.80 + 0.5 * 1.50) / 2.5

    return {
        "total_qo_heads": total_qo_heads,
        "workspace_bytes": workspace_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "workspace_checks": workspace_checks,
        "empty_shard_has_nan_without_zero_and_mask": buggy_is_nan,
        "empty_shard_fixed_combined_out": round(combined_fixed, 6),
        "empty_shard_reference_out": round(ref_combined, 6),
        "empty_shard_abs_err": abs(combined_fixed - ref_combined),
        "empty_shard_weight_on_rank3": weights_fixed[3],
    }


def simulate_symm_mem_scratch_lifetime() -> dict[str, Any]:
    # Simulate two consecutive _reduce_scatter_symm_mem(output=None) or _all_gather_symm_mem calls
    # sharing cache key "rs_out" / "ag_out" when the caller retains the first output across a residual branch.
    scratch_pool: dict[str, list[float]] = {
        "rs_out": [0.0, 0.0, 0.0, 0.0],
        "ag_out": [0.0, 0.0, 0.0, 0.0],
    }

    # Buggy behavior: return internal scratch reference directly
    rs_step1_vals = [1.5, -0.5, 2.0, 0.25]
    rs_step2_vals = [9.0, 8.0, -7.0, 6.0]

    scratch_pool["rs_out"][:] = rs_step1_vals
    caller_saved_buggy = scratch_pool["rs_out"]  # Direct alias to internal scratch
    scratch_pool["rs_out"][:] = rs_step2_vals  # Next layer/collective reuses "rs_out"
    buggy_overwrite_l1_err = sum(abs(a - b) for a, b in zip(caller_saved_buggy, rs_step1_vals))

    # Fixed behavior (vllm@8122776fba): return output.clone() when scratch_output is True
    scratch_pool["rs_out"][:] = rs_step1_vals
    caller_saved_fixed = list(scratch_pool["rs_out"])  # Cloned output buffer
    scratch_pool["rs_out"][:] = rs_step2_vals
    fixed_overwrite_l1_err = sum(abs(a - b) for a, b in zip(caller_saved_fixed, rs_step1_vals))

    # Explicit output tensor passed to _reduce_scatter_symm_mem(output=user_out) does not allocate or clone scratch
    explicit_out_cloned = False
    implicit_scratch_cloned = True

    return {
        "expected_step1_output": rs_step1_vals,
        "buggy_aliased_output_after_step2": caller_saved_buggy,
        "fixed_cloned_output_after_step2": caller_saved_fixed,
        "buggy_overwrite_l1_error": round(buggy_overwrite_l1_err, 6),
        "fixed_overwrite_l1_error": round(fixed_overwrite_l1_err, 6),
        "explicit_output_clones": explicit_out_cloned,
        "implicit_scratch_clones": implicit_scratch_cloned,
    }


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def simulate_megatron_softmax_scale(
    q_head_dim: int = 192,
    rotary_scaling_factor: float = 40.0,
    mscale_all_dim: float = 1.0,
    custom_softmax_scale: float = 0.095,
) -> dict[str, Any]:
    default_inv_sqrt_d = 1.0 / math.sqrt(q_head_dim)
    mscale = yarn_get_mscale(rotary_scaling_factor, mscale_all_dim)

    # Standard Attention flash_decode / flash_decode_and_prefill:
    # Legacy ignored config.softmax_scale and used q.shape[-1] ** -0.5 when self.softmax_scale was None
    std_legacy_scale = default_inv_sqrt_d
    std_fixed_scale = custom_softmax_scale  # get_softmax_scale checks self.softmax_scale -> self.config.softmax_scale -> d^-0.5

    # MLASelfAttention.__init__:
    # Legacy always computed mscale * mscale / sqrt(q_head_dim), ignoring config.softmax_scale
    mla_legacy_scale = (mscale * mscale) * default_inv_sqrt_d
    mla_fixed_scale = (mscale * mscale) * custom_softmax_scale

    # Relative logit scale drift when custom_softmax_scale = 0.095 vs 1/sqrt(192) = 0.07216878
    relative_scale_ratio = mla_fixed_scale / mla_legacy_scale

    return {
        "q_head_dim": q_head_dim,
        "default_inv_sqrt_d": round(default_inv_sqrt_d, 8),
        "yarn_mscale": round(mscale, 8),
        "yarn_mscale_sq": round(mscale * mscale, 8),
        "custom_config_softmax_scale": custom_softmax_scale,
        "std_attention_legacy_scale": round(std_legacy_scale, 8),
        "std_attention_fixed_scale": round(std_fixed_scale, 8),
        "mla_legacy_scale": round(mla_legacy_scale, 8),
        "mla_fixed_scale": round(mla_fixed_scale, 8),
        "fixed_vs_legacy_scale_ratio": round(relative_scale_ratio, 6),
    }


def build_summary() -> dict[str, Any]:
    dcp = simulate_trtllm_dcp_prefill()
    symm = simulate_symm_mem_scratch_lifetime()
    scale = simulate_megatron_softmax_scale()
    return {
        "experiment": "vllm_trtllm_dcp_symm_scratch_megatron_scale",
        "phase": "Phase 34 (priority: 43)",
        "sources": [
            "vllm-project/vllm@7b569ec8953cf8099638a6d2ee1e915b75867c3a",
            "vllm-project/vllm@8122776fba2dcfae2973a0659db2da6ac688306b",
            "NVIDIA/Megatron-LM@0156c213738eaf718f7fba2a77a3b79fa4f4ef97",
        ],
        "trtllm_dcp_prefill": dcp,
        "symm_mem_scratch": symm,
        "megatron_softmax_scale": scale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        dcp = summary["trtllm_dcp_prefill"]
        checks = dcp["workspace_checks"]
        assert checks[0]["fits_trtllm_workspace"] is True and checks[0]["required_bytes"] == 3145728
        assert checks[2]["fits_trtllm_workspace"] is False and checks[2]["required_bytes"] == 16777216
        assert dcp["empty_shard_has_nan_without_zero_and_mask"] is True
        assert dcp["empty_shard_abs_err"] == 0.0
        assert dcp["empty_shard_weight_on_rank3"] == 0.0

        symm = summary["symm_mem_scratch"]
        assert symm["buggy_overwrite_l1_error"] == 30.75
        assert symm["fixed_overwrite_l1_error"] == 0.0

        sc = summary["megatron_softmax_scale"]
        assert sc["std_attention_fixed_scale"] == 0.095
        assert abs(sc["fixed_vs_legacy_scale_ratio"] - (0.095 / (1.0 / math.sqrt(192)))) < 1e-6

    print(json.dumps({"status": "ok", "experiment": summary["experiment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
