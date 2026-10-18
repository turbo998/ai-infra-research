#!/usr/bin/env python3
"""Own-offline analytical model for EAGLE-3 / MTP speculative tree verification and Roofline collapse."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "input.json"
RESULTS_JSON_PATH = ROOT / "results.json"
RESULTS_CSV_PATH = ROOT / "results.csv"


def _expected_accepted_tokens(
    alpha: float, steps: int, topk: int, num_verify_tokens: int, is_tree: bool
) -> float:
    """Return total committed tokens per verify step (1 bonus + E[accepted drafts])."""
    if steps == 0 or num_verify_tokens <= 1:
        return 1.0
    if not is_tree or topk == 1:
        accepted = sum(alpha**i for i in range(1, steps + 1))
        return 1.0 + accepted

    # Tree speculative decoding: topk branches increase per-depth survival probability
    # while bounded by pruned num_verify_tokens budget
    branch_boost = min(0.955, 1.0 - ((1.0 - alpha) ** (1.0 + 0.24 * math.log2(topk))))
    prune_efficiency = min(1.0, (num_verify_tokens / (steps * topk)) ** 0.18)
    eff_alpha = branch_boost * prune_efficiency
    accepted = sum(eff_alpha**i for i in range(1, steps + 1))
    return 1.0 + accepted


def _step_latency_ms(
    batch_size: int,
    cfg_entry: dict[str, Any],
    hw: dict[str, Any],
    model: dict[str, Any],
) -> tuple[float, float, float]:
    """Return (step_latency_ms, effective_query_tokens, arithmetic_intensity_ratio)."""
    steps = cfg_entry["steps"]
    n_verify = cfg_entry["num_verify_tokens"]
    effective_q_tokens = batch_size * n_verify
    ridge_tokens = hw["ridge_point_tokens_per_step"]
    ai_ratio = effective_q_tokens / ridge_tokens

    # Target model forward latency on (batch_size * n_verify) query tokens:
    # Roofline transition: memory-bound below ridge_tokens, compute-bound above ridge_tokens!
    weight_time_ms = (
        model["target_active_weights_gb_per_gpu"] / hw["effective_hbm_bw_gbps_per_gpu"]
    ) * 1000.0
    # Once effective_q_tokens crosses ridge_tokens, GEMM latency scales linearly with query tokens
    # plus non-causal tree-mask attention & Tensor Core wave quantization overhead
    gemm_multiplier = max(1.0, ai_ratio * 1.45) + 0.12 * min(1.0, ai_ratio)
    if n_verify > 1:
        gemm_multiplier *= 1.0 + 0.10 * math.log2(n_verify)
    target_gemm_ms = weight_time_ms * gemm_multiplier

    # KV cache read + tree mask overhead (FULL_MASK: seq_lens_sum * n_verify + n_verify^2 * bs)
    kv_gb = (
        batch_size * model["avg_context_len"] * model["kv_bytes_per_token_per_gpu"]
    ) / 1e9
    # Non-causal tree verify cannot share a single query vector cleanly across all KV heads when n_verify is large
    kv_verify_amplification = 1.0 + 0.048 * (n_verify - 1)
    tree_mask_gb = (
        (
            batch_size * model["avg_context_len"] * n_verify
            + batch_size * (n_verify**2)
        )
        * 2.0
        / 1e9
    )
    attn_ms = (
        (kv_gb * kv_verify_amplification + tree_mask_gb * (2.4 if n_verify > 1 else 0.0))
        / hw["effective_hbm_bw_gbps_per_gpu"]
    ) * 1000.0

    target_total_ms = model["target_base_latency_ms"] + target_gemm_ms + attn_ms

    if steps == 0:
        return target_total_ms, float(effective_q_tokens), ai_ratio

    draft_q_tokens = batch_size * cfg_entry["topk"]
    draft_ai = draft_q_tokens / ridge_tokens
    draft_weight_ms = (
        model["draft_weights_gb_per_gpu"] / hw["effective_hbm_bw_gbps_per_gpu"]
    ) * 1000.0
    draft_per_step_ms = model["draft_step_base_latency_ms"] + draft_weight_ms * (
        1.0 + 1.25 * draft_ai
    )
    draft_total_ms = (
        steps * draft_per_step_ms
        + model["tree_build_and_verify_kernel_ms"] * (1.0 + 0.012 * batch_size)
    )

    return target_total_ms + draft_total_ms, float(effective_q_tokens), ai_ratio


def evaluate_eagle3_speculative_tree(config: dict[str, Any]) -> dict[str, Any]:
    hw = config["hardware"]
    model = config["model"]
    candidates = config["candidate_configs"]

    rows: list[dict[str, Any]] = []
    adaptive_policy_table: list[dict[str, Any]] = []

    for profile_name, alpha in config["acceptance_profiles"].items():
        for bs in config["batch_sizes"]:
            baseline_entry = candidates[0]
            base_lat_ms, _, _ = _step_latency_ms(bs, baseline_entry, hw, model)
            base_tps = (bs * 1.0) / (base_lat_ms / 1000.0)

            best_cfg_id = "nospec_baseline"
            best_speedup = 1.0
            best_tps = base_tps
            best_verify_tokens = 1

            for cand in candidates:
                tau = _expected_accepted_tokens(
                    alpha,
                    cand["steps"],
                    cand["topk"],
                    cand["num_verify_tokens"],
                    cand["is_tree"],
                )
                step_ms, eff_q_tokens, ai_ratio = _step_latency_ms(
                    bs, cand, hw, model
                )
                tpot_ms = step_ms / tau
                cluster_tps = (bs * tau) / (step_ms / 1000.0)
                speedup = cluster_tps / base_tps

                if speedup > best_speedup:
                    best_speedup = speedup
                    best_cfg_id = cand["id"]
                    best_tps = cluster_tps
                    best_verify_tokens = cand["num_verify_tokens"]

                rows.append(
                    {
                        "profile": profile_name,
                        "alpha": alpha,
                        "batch_size": bs,
                        "config_id": cand["id"],
                        "steps": cand["steps"],
                        "topk": cand["topk"],
                        "num_verify_tokens": cand["num_verify_tokens"],
                        "effective_query_tokens": int(eff_q_tokens),
                        "ridge_saturation_ratio": round(ai_ratio, 3),
                        "expected_tokens_per_step": round(tau, 3),
                        "step_latency_ms": round(step_ms, 3),
                        "effective_tpot_ms": round(tpot_ms, 3),
                        "cluster_tokens_per_sec": round(cluster_tps, 1),
                        "speedup_vs_nospec": round(speedup, 3),
                    }
                )

            adaptive_policy_table.append(
                {
                    "profile": profile_name,
                    "alpha": alpha,
                    "batch_size": bs,
                    "optimal_config_id": best_cfg_id,
                    "optimal_verify_tokens": best_verify_tokens,
                    "optimal_speedup_vs_nospec": round(best_speedup, 3),
                    "optimal_tokens_per_sec": round(best_tps, 1),
                }
            )

    # Extract key findings for code_math (alpha=0.86) and general_chat (alpha=0.74)
    v32_bs1 = next(
        r
        for r in rows
        if r["profile"] == "code_math"
        and r["batch_size"] == 1
        and r["config_id"] == "tree_s5_k6_v32"
    )
    v32_bs32 = next(
        r
        for r in rows
        if r["profile"] == "code_math"
        and r["batch_size"] == 32
        and r["config_id"] == "tree_s5_k6_v32"
    )
    v64_bs16 = next(
        r
        for r in rows
        if r["profile"] == "general_chat"
        and r["batch_size"] == 16
        and r["config_id"] == "tree_s6_k8_v64"
    )
    adaptive_bs32 = next(
        p
        for p in adaptive_policy_table
        if p["profile"] == "code_math" and p["batch_size"] == 32
    )

    summary = {
        "ridge_point_tokens_per_step": hw["ridge_point_tokens_per_step"],
        "code_math_bs1_tree32_tau": v32_bs1["expected_tokens_per_step"],
        "code_math_bs1_tree32_speedup": v32_bs1["speedup_vs_nospec"],
        "code_math_bs32_tree32_effective_query_tokens": v32_bs32[
            "effective_query_tokens"
        ],
        "code_math_bs32_tree32_ridge_ratio": v32_bs32["ridge_saturation_ratio"],
        "code_math_bs32_tree32_collapsed_speedup": v32_bs32["speedup_vs_nospec"],
        "general_chat_bs16_tree64_collapsed_speedup": v64_bs16["speedup_vs_nospec"],
        "adaptive_bs32_optimal_config": adaptive_bs32["optimal_config_id"],
        "adaptive_bs32_recovered_speedup": adaptive_bs32["optimal_speedup_vs_nospec"],
    }

    return {
        "experiment_id": "eagle3_speculative_tree",
        "evidence_class": "own-offline",
        "summary": summary,
        "adaptive_policy_table": adaptive_policy_table,
        "rows": rows,
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    results = evaluate_eagle3_speculative_tree(config)
    rendered = json.dumps(results, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not RESULTS_JSON_PATH.exists():
            raise SystemExit("Missing results.json")
        existing = RESULTS_JSON_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        print("eagle3_speculative_tree check passed.")
        return 0

    RESULTS_JSON_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    _write_csv(results["rows"], RESULTS_CSV_PATH)
    print("eagle3_speculative_tree results written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
