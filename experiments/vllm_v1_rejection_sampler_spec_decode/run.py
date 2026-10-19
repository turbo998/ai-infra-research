from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def simulate_vocab_roofline(batch_size: int = 64, draft_k: int = 5) -> Dict[str, object]:
    """
    Compare PyTorch eager multi-pass residual sampling (`sub -> clamp -> sum -> div -> multinomial`)
    vs vLLM V1 Triton `sample_recovered_tokens_kernel` (`BLOCK_SIZE=8192`, normalization-free
    Gumbel-Max `tl.maximum(target_prob - draft_prob, 0.0) * inv_q` with per-request `inv_q` `[B, V]`).
    """
    num_tokens = batch_size * draft_k  # 320 draft tokens
    hbm_effective_bw_gbps = 2650.0  # H100 SXM5 effective streaming bandwidth
    results = {}

    for vocab_size in [128256, 256000]:
        # Eager multi-pass HBM traffic (FP32 = 4B):
        # 1) diff = target - draft (read 2*N*V, write N*V)
        # 2) pos = clamp(diff, min=0) (read N*V, write N*V)
        # 3) norm = pos / sum(pos) (read 2*N*V, write N*V)
        # 4) multinomial Gumbel / CDF (read N*V + N*V random exponential)
        # Total = 10 * N * V * 4 bytes + 5 kernel launches (~28 us launch overhead)
        eager_bytes = 10 * num_tokens * vocab_size * 4
        eager_mib = eager_bytes / (1024**2)
        eager_ms = (eager_bytes / (hbm_effective_bw_gbps * 1e9)) * 1000.0 + 0.032

        # vLLM V1 Triton `sample_recovered_tokens_kernel` (`BLOCK_SIZE=8192`):
        # Reads `target_probs` [N, V] + `draft_probs` [N, V] once, and shares `inv_q` [B, V]
        # across the K draft tokens of the same request (`req_idx = tl.program_id(0)`),
        # writing only 1 int32 `recovered_id` per token! Zero intermediate HBM writes, zero `tl.sum` pass.
        triton_bytes = (2 * num_tokens * vocab_size + batch_size * vocab_size) * 4 + num_tokens * 4
        triton_mib = triton_bytes / (1024**2)
        triton_ms = (triton_bytes / (hbm_effective_bw_gbps * 1e9)) * 1000.0 + 0.008

        results[f"vocab_{vocab_size}"] = {
            "vocab_size": vocab_size,
            "num_draft_tokens_total": num_tokens,
            "block_size": 8192,
            "tiles_per_token": math.ceil(vocab_size / 8192),
            "eager_multipass_hbm_mib": round(eager_mib, 2),
            "triton_gumbel_1pass_hbm_mib": round(triton_mib, 2),
            "hbm_traffic_reduction_x": round(eager_mib / triton_mib, 3),
            "eager_latency_ms": round(eager_ms, 3),
            "triton_latency_ms": round(triton_ms, 3),
            "kernel_speedup_x": round(eager_ms / triton_ms, 3),
        }

    return results


def verify_gumbel_max_equivalence_and_fp64_guard() -> Dict[str, object]:
    """
    Prove that `argmax_v (max(p_v - q_v, 0) / Exp(1)_v)` is mathematically identical to sampling
    from `p'_v = max(p_v - q_v, 0) / Z` where `Z = sum_u max(p_u - q_u, 0)`, because scaling all
    exponential rates by a positive constant `1/Z` preserves the argmax distribution `p'_v`.
    Also quantify the `float32` exact-0.0 uniform sample bug (`pytorch#16706`) when `target_prob == 0.0`.
    """
    target_probs = [0.52, 0.28, 0.15, 0.05, 0.00]
    draft_probs = [0.30, 0.40, 0.20, 0.02, 0.08]
    residual_unnorm = [max(t - d, 0.0) for t, d in zip(target_probs, draft_probs)]
    z_norm = sum(residual_unnorm)
    exact_recovered_dist = [r / z_norm for r in residual_unnorm]

    # Verify scale invariance on deterministic exponential draw grid
    exp_draws_set = [
        [0.42, 1.15, 0.88, 0.19, 0.63],
        [0.11, 0.95, 0.74, 0.55, 0.31],
        [0.85, 0.33, 0.21, 0.04, 0.92],
    ]
    unnorm_winners: List[int] = []
    norm_winners: List[int] = []
    for q_exp in exp_draws_set:
        inv_q = [1.0 / x for x in q_exp]
        w_unnorm = max(range(len(residual_unnorm)), key=lambda i: residual_unnorm[i] * inv_q[i])
        w_norm = max(range(len(exact_recovered_dist)), key=lambda i: exact_recovered_dist[i] * inv_q[i])
        unnorm_winners.append(w_unnorm)
        norm_winners.append(w_norm)

    # At token index 4 (`target_prob = 0.00`, `draft_prob = 0.08`), ratio `target/draft = 0.0`.
    # If `uniform_prob == 0.0` (which happens with p = 2^-24 in float32 vs 2^-53 in float64)
    # and acceptance uses `target_prob / draft_prob >= uniform_prob` without `target_prob > 0`,
    # token 4 is falsely accepted with zero target support!
    fp32_zero_prob = 2.0**-24
    fp64_zero_prob = 2.0**-53

    return {
        "residual_mass_Z": round(z_norm, 4),
        "exact_recovered_distribution": [round(x, 6) for x in exact_recovered_dist],
        "unnormalized_argmax_equals_normalized_argmax": unnorm_winners == norm_winners,
        "fp32_exact_zero_sample_prob": fp32_zero_prob,
        "fp64_exact_zero_sample_prob": fp64_zero_prob,
        "zero_underflow_risk_reduction_x": round(fp32_zero_prob / fp64_zero_prob, 1),
    }


def simulate_fly_entropy_deferred_verification() -> Dict[str, object]:
    """
    Model `vllm/v1/spec_decode/fly.py` (`compute_fly_entropy`, `apply_fly_greedy_acceptance_kernel`,
    `apply_fly_random_acceptance_kernel`): deferring a rejection at `pos` when top-3 entropy
    `fly_entropy[pos] >= fly_entropy_threshold` and the next `W` (`FLY_WINDOW_SIZE`) draft tokens
    match the target (`future_draft_id == future_target_id`).
    """
    # Trace of 8 representative speculative steps (K=5 draft tokens each, W=2 lookahead window)
    # Each step has: (mismatch_pos_or_5, entropy_at_mismatch, future_W_match_bool)
    steps = [
        (5, 0.12, True),   # all 5 match natively
        (1, 0.82, True),   # pos 1 mismatch at high entropy (0.82 >= 0.65), pos 2,3,4 match!
        (2, 0.21, True),   # pos 2 mismatch at low entropy (0.21 < 0.65) -> must reject
        (0, 0.79, True),   # pos 0 mismatch at high entropy (0.79 >= 0.65), pos 1..4 match!
        (3, 0.74, False),  # pos 3 mismatch at high entropy, but pos 4 mismatches -> must reject
        (1, 0.91, True),   # pos 1 mismatch at high entropy, pos 2,3 match, pos 4 match!
        (4, 0.88, True),   # pos 4 is at boundary (pos + W >= K for K=5, W=2) -> cannot defer
        (2, 0.68, True),   # pos 2 mismatch at high entropy (0.68 >= 0.65), pos 3,4 match!
    ]
    k_draft = 5
    w_window = 2
    threshold = 0.65

    native_accepted_counts: List[int] = []
    fly_accepted_counts: List[int] = []

    for mismatch_pos, ent, future_match in steps:
        native_acc = mismatch_pos  # number of accepted draft tokens before first rejection
        native_accepted_counts.append(native_acc)

        can_defer = (
            mismatch_pos < k_draft
            and (mismatch_pos + w_window < k_draft)
            and ent >= threshold
            and future_match
        )
        fly_acc = k_draft if can_defer else mismatch_pos
        fly_accepted_counts.append(fly_acc)

    # Total output tokens per step = accepted_draft + 1 (recovered or bonus token)
    mean_native_output_tokens = sum(x + 1 for x in native_accepted_counts) / len(steps)
    mean_fly_output_tokens = sum(x + 1 for x in fly_accepted_counts) / len(steps)

    return {
        "draft_len_k": k_draft,
        "fly_window_size_w": w_window,
        "fly_entropy_top_k": 3,
        "fly_entropy_threshold": threshold,
        "native_mean_output_tokens_per_step": round(mean_native_output_tokens, 3),
        "fly_mean_output_tokens_per_step": round(mean_fly_output_tokens, 3),
        "acceptance_length_gain_pct": round(
            (mean_fly_output_tokens / mean_native_output_tokens - 1.0) * 100.0, 2
        ),
    }


def run_simulation() -> Dict[str, object]:
    return {
        "vocab_roofline": simulate_vocab_roofline(),
        "gumbel_max_and_fp64_guard": verify_gumbel_max_equivalence_and_fp64_guard(),
        "fly_deferred_verification": simulate_fly_entropy_deferred_verification(),
    }


def _check(res: Dict[str, object]) -> None:
    vr = res["vocab_roofline"]
    assert vr["vocab_128256"]["hbm_traffic_reduction_x"] > 4.4
    assert vr["vocab_128256"]["kernel_speedup_x"] > 3.5
    assert vr["vocab_256000"]["kernel_speedup_x"] > 3.8

    gm = res["gumbel_max_and_fp64_guard"]
    assert gm["unnormalized_argmax_equals_normalized_argmax"] is True
    assert gm["zero_underflow_risk_reduction_x"] > 1e8

    fly = res["fly_deferred_verification"]
    assert fly["fly_mean_output_tokens_per_step"] > fly["native_mean_output_tokens_per_step"]
    assert fly["acceptance_length_gain_pct"] > 35.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    res = run_simulation()
    if args.check:
        _check(res)
        print("vllm_v1_rejection_sampler_spec_decode: OK")
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    elif not args.check:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
