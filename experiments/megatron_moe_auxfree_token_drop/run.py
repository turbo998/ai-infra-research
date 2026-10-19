#!/usr/bin/env python3
"""Deterministic simulator for Megatron-Core MoE Aux-Loss-Free Routing (`expert_bias`),
Dual Coordinate-Descent Quantile Balancing (`qb_dual_update`), and Capacity-Factor Token Dropping.

Models the exact algorithms verified in `NVIDIA/Megatron-LM`
(`commit e8a302ac5bfeb98c6701020cc6cba977d05ab1d4`):
- `megatron/core/transformer/moe/router.py` (`TopKRouter`, `expert_bias` in `fp32`, `local_tokens_per_expert` in `int64`, `qb_beta`)
- `megatron/core/transformer/moe/moe_utils.py`:
  - `get_capacity(num_tokens, num_experts, capacity_factor)` (lines 258-277)
  - `apply_router_token_dropping(routing_probs, routing_map, router_topk, capacity_factor, drop_policy, pad_to_capacity)` (lines 1065-1129)
  - `get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate)` (lines 1245-1282)
  - `qb_dual_update(scores, k, beta, update_beta=True)` (lines 220-256)
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
from typing import Any


RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "results.json"


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity: int | None = None) -> int:
    """Exact replica of `megatron/core/transformer/moe/moe_utils.py:258-277`."""
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


def get_updated_expert_bias(
    tokens_per_expert: list[int],
    expert_bias: list[float],
    expert_bias_update_rate: float,
) -> list[float]:
    """Exact replica of `megatron/core/transformer/moe/moe_utils.py:1245-1282`.

    ```python
    num_experts = tokens_per_expert.shape[-1]
    total_tokens = tokens_per_expert.sum(dim=-1, keepdim=True)
    update_direction = torch.sign(total_tokens - tokens_per_expert * num_experts)
    updated_expert_bias = expert_bias + update_direction * expert_bias_update_rate
    ```
    Note the integer-exact comparison without float rounding:
    `tokens_per_expert < total / num_experts` iff `tokens_per_expert * num_experts < total`.
    """
    num_experts = len(tokens_per_expert)
    total_tokens = sum(tokens_per_expert)
    updated: list[float] = []
    for count, bias in zip(tokens_per_expert, expert_bias):
        diff = total_tokens - count * num_experts
        sign = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        updated.append(bias + sign * expert_bias_update_rate)
    return updated


def qb_dual_update(
    scores: list[list[float]],
    k: int,
    beta: list[float],
) -> tuple[list[list[int]], list[float]]:
    """Exact replica of `qb_dual_update` (`megatron/core/transformer/moe/moe_utils.py:220-256`).

    ```python
    topk_result = (scores - beta).topk(k + 1, dim=1)
    indices = topk_result.indices[:, :-1]
    col_target = num_tokens * k // num_experts
    alpha = topk_result.values[:, -1:]
    beta_local = (scores - alpha).topk(col_target + 1, dim=0).values[-1].contiguous()
    ```
    """
    num_tokens = len(scores)
    num_experts = len(scores[0])
    assert (num_tokens * k) % num_experts == 0

    col_target = (num_tokens * k) // num_experts
    indices: list[list[int]] = []
    alphas: list[float] = []

    for m in range(num_tokens):
        adjusted = [(scores[m][e] - beta[e], e) for e in range(num_experts)]
        adjusted.sort(key=lambda x: x[0], reverse=True)
        top_k_plus_1 = adjusted[: k + 1]
        indices.append([e for _, e in top_k_plus_1[:-1]])
        alphas.append(top_k_plus_1[-1][0])

    beta_local: list[float] = []
    for e in range(num_experts):
        col_vals = sorted((scores[m][e] - alphas[m] for m in range(num_tokens)), reverse=True)
        beta_local.append(col_vals[col_target])

    return indices, beta_local


def apply_router_token_dropping_sim(
    routing_probs: list[list[float]],
    routing_map: list[list[bool]],
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",
    pad_to_capacity: bool = False,
) -> dict[str, Any]:
    """Exact replica of `apply_router_token_dropping` (`megatron/core/transformer/moe/moe_utils.py:1065-1129`)."""
    num_tokens = len(routing_probs)
    num_experts = len(routing_probs[0])
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    capacity_mask = [[False] * num_experts for _ in range(num_tokens)]
    if expert_capacity > num_tokens:
        for t in range(num_tokens):
            for e in range(num_experts):
                capacity_mask[t][e] = True
    else:
        for e in range(num_experts):
            if drop_policy == "probs":
                # topk along dim=0 of routing_probs[:, e]
                ranked = sorted(range(num_tokens), key=lambda t: routing_probs[t][e], reverse=True)
            elif drop_policy == "position":
                # topk along dim=0 of routing_map[:, e].int() (preserves earlier token positions)
                ranked = sorted(range(num_tokens), key=lambda t: (1 if routing_map[t][e] else 0, -t), reverse=True)
            else:
                raise ValueError(f"Invalid drop_policy: {drop_policy}")
            for t in ranked[:expert_capacity]:
                capacity_mask[t][e] = True

    total_routed_before = sum(1 for t in range(num_tokens) for e in range(num_experts) if routing_map[t][e])
    final_map = [[False] * num_experts for _ in range(num_tokens)]
    dropped_assignments = 0
    dropped_prob_mass = 0.0
    total_prob_mass = sum(sum(row) for row in routing_probs)

    for t in range(num_tokens):
        for e in range(num_experts):
            if pad_to_capacity:
                final_map[t][e] = capacity_mask[t][e]
            else:
                final_map[t][e] = routing_map[t][e] and capacity_mask[t][e]
            if routing_map[t][e] and not (routing_map[t][e] and capacity_mask[t][e]):
                dropped_assignments += 1
                dropped_prob_mass += routing_probs[t][e]

    padded_dummy_slots = 0
    if pad_to_capacity:
        for e in range(num_experts):
            actual_kept = sum(1 for t in range(num_tokens) if routing_map[t][e] and capacity_mask[t][e])
            padded_dummy_slots += max(0, expert_capacity - actual_kept)

    return {
        "capacity_factor": capacity_factor,
        "drop_policy": drop_policy,
        "pad_to_capacity": pad_to_capacity,
        "expert_capacity": expert_capacity,
        "token_drop_rate_pct": round((dropped_assignments / max(1, total_routed_before)) * 100.0, 2),
        "dropped_prob_mass_pct": round((dropped_prob_mass / max(1e-9, total_prob_mass)) * 100.0, 2),
        "padding_waste_pct": round(
            (padded_dummy_slots / max(1, num_experts * expert_capacity)) * 100.0, 2
        )
        if pad_to_capacity
        else 0.0,
    }


def generate_zipfian_router_scores(
    num_tokens: int, num_experts: int, skew_alpha: float = 0.85, seed: int = 42
) -> list[list[float]]:
    """Generate realistic Zipf-skewed router affinity scores `[num_tokens, num_experts]`."""
    rng = random.Random(seed)
    expert_popularity = [1.25 / ((e + 1) ** skew_alpha) for e in range(num_experts)]
    scores: list[list[float]] = []
    for _ in range(num_tokens):
        row = [expert_popularity[e] + rng.gauss(0.0, 0.32) for e in range(num_experts)]
        scores.append(row)
    return scores


def route_with_bias(
    scores: list[list[float]],
    topk: int,
    expert_bias: list[float],
) -> tuple[list[list[float]], list[list[bool]], list[int]]:
    """Route tokens by selecting top-k on `scores + expert_bias`, but computing gating weights from unbiased `scores`."""
    num_tokens = len(scores)
    num_experts = len(scores[0])
    routing_probs = [[0.0] * num_experts for _ in range(num_tokens)]
    routing_map = [[False] * num_experts for _ in range(num_tokens)]
    tokens_per_expert = [0] * num_experts

    for t in range(num_tokens):
        biased = [(scores[t][e] + expert_bias[e], e) for e in range(num_experts)]
        biased.sort(key=lambda x: x[0], reverse=True)
        selected = [e for _, e in biased[:topk]]
        # Sigmoid / normalized unbiased scores on selected experts (`router.py:1105-1114`)
        raw_weights = [1.0 / (1.0 + math.exp(-scores[t][e])) for e in selected]
        norm = sum(raw_weights)
        for e, w in zip(selected, raw_weights):
            routing_probs[t][e] = w / norm
            routing_map[t][e] = True
            tokens_per_expert[e] += 1

    return routing_probs, routing_map, tokens_per_expert


def run_moe_routing_experiment() -> dict[str, Any]:
    num_tokens = 1024
    num_experts = 32
    topk = 4
    ideal_per_expert = (num_tokens * topk) / num_experts  # 128 tokens/expert

    base_scores = generate_zipfian_router_scores(num_tokens, num_experts, skew_alpha=0.85, seed=42)

    # 1. Unbalanced routing (no aux loss, no bias)
    probs_unbal, map_unbal, counts_unbal = route_with_bias(base_scores, topk, [0.0] * num_experts)
    max_violation_unbal = max(counts_unbal) / ideal_per_expert

    # 2. Aux-Loss-Free `expert_bias` sign-update convergence (`get_updated_expert_bias`, rate=0.02)
    bias = [0.0] * num_experts
    bias_trajectory: list[dict[str, Any]] = []
    for step in range(45):
        step_scores = generate_zipfian_router_scores(num_tokens, num_experts, skew_alpha=0.85, seed=100 + step)
        probs_step, map_step, counts_step = route_with_bias(step_scores, topk, bias)
        max_vio = max(counts_step) / ideal_per_expert
        min_vio = min(counts_step) / ideal_per_expert
        if step in {0, 5, 10, 20, 30, 44}:
            bias_trajectory.append(
                {
                    "step": step,
                    "max_load_factor": round(max_vio, 3),
                    "min_load_factor": round(min_vio, 3),
                    "max_expert_tokens": max(counts_step),
                    "min_expert_tokens": min(counts_step),
                }
            )
        bias = get_updated_expert_bias(counts_step, bias, expert_bias_update_rate=0.02)

    probs_biased, map_biased, counts_biased = route_with_bias(base_scores, topk, bias)
    max_violation_biased = max(counts_biased) / ideal_per_expert

    # 3. Dual Coordinate-Descent Quantile Balancing (`qb_dual_update`)
    qb_beta = [0.0] * num_experts
    for _ in range(3):
        qb_indices, qb_beta = qb_dual_update(base_scores, topk, qb_beta)
    counts_qb = [0] * num_experts
    for row_idx in qb_indices:
        for e in row_idx:
            counts_qb[e] += 1
    max_violation_qb = max(counts_qb) / ideal_per_expert

    # 4. Capacity Factor & Drop Policy Sweep on Unbalanced vs Aux-Loss-Free Biased Routing
    capacity_factors = [1.0, 1.25, 1.5, 2.0]
    drop_sweep: list[dict[str, Any]] = []
    for cf in capacity_factors:
        unbal_probs_drop = apply_router_token_dropping_sim(
            probs_unbal, map_unbal, topk, cf, drop_policy="probs", pad_to_capacity=True
        )
        unbal_pos_drop = apply_router_token_dropping_sim(
            probs_unbal, map_unbal, topk, cf, drop_policy="position", pad_to_capacity=True
        )
        biased_probs_drop = apply_router_token_dropping_sim(
            probs_biased, map_biased, topk, cf, drop_policy="probs", pad_to_capacity=False
        )
        drop_sweep.append(
            {
                "capacity_factor": cf,
                "expert_capacity": unbal_probs_drop["expert_capacity"],
                "unbalanced_probs_drop_rate_pct": unbal_probs_drop["token_drop_rate_pct"],
                "unbalanced_probs_dropped_mass_pct": unbal_probs_drop["dropped_prob_mass_pct"],
                "unbalanced_position_drop_rate_pct": unbal_pos_drop["token_drop_rate_pct"],
                "unbalanced_position_dropped_mass_pct": unbal_pos_drop["dropped_prob_mass_pct"],
                "unbalanced_pad_to_capacity_waste_pct": unbal_probs_drop["padding_waste_pct"],
                "biased_probs_drop_rate_pct": biased_probs_drop["token_drop_rate_pct"],
                "biased_probs_dropped_mass_pct": biased_probs_drop["dropped_prob_mass_pct"],
            }
        )

    cf10 = next(r for r in drop_sweep if r["capacity_factor"] == 1.0)
    cf125 = next(r for r in drop_sweep if r["capacity_factor"] == 1.25)

    return {
        "experiment_id": "megatron_moe_auxfree_token_drop",
        "router_config": {
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "topk": topk,
            "ideal_tokens_per_expert": int(ideal_per_expert),
        },
        "expert_bias_convergence": bias_trajectory,
        "capacity_factor_sweep": drop_sweep,
        "summary": {
            "unbalanced_max_load_factor": round(max_violation_unbal, 3),
            "expert_bias_max_load_factor": round(max_violation_biased, 3),
            "quantile_balancing_max_load_factor": round(max_violation_qb, 3),
            "cf10_unbalanced_drop_rate_pct": cf10["unbalanced_probs_drop_rate_pct"],
            "cf10_biased_drop_rate_pct": cf10["biased_probs_drop_rate_pct"],
            "cf125_unbalanced_drop_rate_pct": cf125["unbalanced_probs_drop_rate_pct"],
            "cf125_biased_drop_rate_pct": cf125["biased_probs_drop_rate_pct"],
            "cf125_probs_dropped_mass_pct": cf125["unbalanced_probs_dropped_mass_pct"],
            "cf125_position_dropped_mass_pct": cf125["unbalanced_position_dropped_mass_pct"],
            "cf125_pad_waste_pct": cf125["unbalanced_pad_to_capacity_waste_pct"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json matches.")
    args = parser.parse_args()

    results = run_moe_routing_experiment()
    formatted = json.dumps(results, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            print(f"Missing {RESULTS_PATH}", file=sys.stderr)
            return 1
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != formatted:
            print(f"Mismatch in {RESULTS_PATH}; re-run without --check to update.", file=sys.stderr)
            return 1
        print(f"OK: {RESULTS_PATH} is up to date.")
        return 0

    RESULTS_PATH.write_text(formatted, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
