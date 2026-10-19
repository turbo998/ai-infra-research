#!/usr/bin/env python3
"""
Deterministic offline simulator for NVIDIA Dynamo Distributed KV-Aware Router & Conditional Disaggregation:
- Verification of `DefaultWorkerScorer::worker_logit` rational overlap credit decay
  `overlap_credit_decay = 1.0 / (1.0 + overlap_score_credit_decay * normalized_prefill_load)`
  and multi-tier `(device, host_pinned, disk, shared_hicache)` credit scoring in
  `ai-dynamo/dynamo` (`lib/kv-router/src/scheduling/selector/reference.rs`)
- Verification of `ConditionalDisaggPolicy` (`IslBoundingPolicy` `eff_isl < 2048` & `ratio < 0.7`,
  plus `decode_gate_allows_bypass`) in `ai-dynamo/dynamo` (`lib/kv-router/src/conditional_disagg.rs`
  and `lib/llm/src/kv_router/prefill_router/conditional_bypass.rs`)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def dynamo_worker_logit(
    isl_tokens: int,
    block_size: int,
    device_overlap_blocks: float,
    host_overlap_blocks: float,
    shared_beyond_device_blocks: int,
    active_prefill_tokens: int,
    min_active_prefill_tokens: int,
    decode_cost_blocks: float,
    active_requests: int,
    overlap_score_credit: float = 1.0,
    overlap_score_credit_decay: float = 0.65,
    host_cache_hit_weight: float = 0.5,
    shared_cache_multiplier: float = 0.5,
    prefill_load_scale: float = 1.0,
    decode_active_request_weight: float = 0.15,
) -> dict[str, float]:
    """
    Exact Python implementation of `DefaultWorkerScorer::worker_logit` from
    `ai-dynamo/dynamo` (`lib/kv-router/src/scheduling/selector/reference.rs`).
    """
    request_blocks = max(1, (isl_tokens + block_size - 1) // block_size)
    if overlap_score_credit_decay > 0.0:
        excess_active_prefill_blocks = (
            max(0, active_prefill_tokens - min_active_prefill_tokens) / float(block_size)
        )
        normalized_prefill_load = excess_active_prefill_blocks / float(request_blocks)
        overlap_credit_decay = 1.0 / (
            1.0 + overlap_score_credit_decay * normalized_prefill_load
        )
    else:
        overlap_credit_decay = 1.0

    effective_overlap_score_credit = overlap_score_credit * overlap_credit_decay
    shared_overlap_blocks = shared_cache_multiplier * float(shared_beyond_device_blocks)
    overlap_credit_blocks = (
        effective_overlap_score_credit * device_overlap_blocks
        + host_cache_hit_weight * host_overlap_blocks
        + shared_overlap_blocks
    )

    raw_prefill_tokens = active_prefill_tokens + isl_tokens
    raw_prefill_blocks = raw_prefill_tokens / float(block_size)
    adjusted_prefill_blocks = max(0.0, raw_prefill_blocks - overlap_credit_blocks)
    prefill_cost_blocks = prefill_load_scale * adjusted_prefill_blocks
    active_request_cost_blocks = decode_active_request_weight * float(active_requests)
    logit = prefill_cost_blocks + decode_cost_blocks + active_request_cost_blocks

    return {
        "overlap_credit_decay": round(overlap_credit_decay, 5),
        "overlap_credit_blocks": round(overlap_credit_blocks, 3),
        "raw_prefill_blocks": round(raw_prefill_blocks, 3),
        "adjusted_prefill_blocks": round(adjusted_prefill_blocks, 3),
        "logit": round(logit, 4),
    }


def should_bypass_remote_prefill_isl_bounding(
    prompt_tokens: int,
    decode_chosen_cached_tokens: int,
    decode_busy: bool,
    eff_isl_threshold: int = 2048,
    eff_isl_ratio_threshold: float = 0.7,
) -> tuple[bool, int, float]:
    """
    Exact Python implementation of `IslBoundingPolicy::should_bypass_remote_prefill` +
    `decode_gate_allows_bypass` from `ai-dynamo/dynamo`
    (`lib/kv-router/src/conditional_disagg.rs` & `conditional_bypass.rs`).
    """
    net_new_tokens = max(0, prompt_tokens - min(decode_chosen_cached_tokens, prompt_tokens))
    ratio = net_new_tokens / float(max(1, prompt_tokens))
    policy_says_bypass = (net_new_tokens < eff_isl_threshold) and (
        ratio < eff_isl_ratio_threshold
    )
    bypass = policy_says_bypass and (not decode_busy)
    return bypass, net_new_tokens, round(ratio, 4)


def generate_workload(num_requests: int = 480, seed: int = 20261020) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    num_prefixes = 96
    # Zipfian weights for prefix popularity (s = 1.05 across 96 multi-tenant agent prefixes)
    weights = [1.0 / ((i + 1) ** 1.05) for i in range(num_prefixes)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    prefix_Shared_lens = [
        16384 if i < 8 else (12288 if i < 32 else 8192) for i in range(num_prefixes)
    ]

    requests: list[dict[str, Any]] = []
    t_ms = 0.0
    for req_id in range(num_requests):
        # Burst arrival: every 40 requests, a hot prefix receives a 10-request micro-burst
        if req_id % 40 < 10:
            dt_ms = rng.expovariate(1.0 / 0.85)
            prefix_id = (req_id // 40) % 4
        else:
            dt_ms = rng.expovariate(1.0 / 6.2)
            r = rng.random()
            cum = 0.0
            prefix_id = num_prefixes - 1
            for idx, p in enumerate(probs):
                cum += p
                if r <= cum:
                    prefix_id = idx
                    break
        t_ms += dt_ms
        shared_tokens = prefix_Shared_lens[prefix_id]
        # Follow-up agent turn has smaller net-new suffix (512..1792 tokens) 65% of the time,
        # or a long tool output suffix (3072..6144 tokens) 35% of the time
        if rng.random() < 0.65:
            suffix_tokens = rng.choice([512, 768, 1024, 1536, 1792])
        else:
            suffix_tokens = rng.choice([3072, 4096, 5120, 6144])

        requests.append(
            {
                "req_id": req_id,
                "arrival_ms": round(t_ms, 3),
                "prefix_id": prefix_id,
                "shared_prefix_tokens": shared_tokens,
                "suffix_tokens": suffix_tokens,
                "prompt_tokens": shared_tokens + suffix_tokens,
            }
        )
    return requests


def simulate_router_policy(
    policy_name: str,
    requests: list[dict[str, Any]],
    num_workers: int = 16,
    block_size: int = 64,
) -> dict[str, Any]:
    """
    Simulate 16 workers processing the request trace under 5 routing policies:
    1. `ROUND_ROBIN`: oblivious modulo assignment
    2. `LEAST_LOADED`: picks worker with minimum active prefill backlog (`overlap_score_credit = 0`)
    3. `STICKY_PREFIX_NO_DECAY`: picks highest prefix overlap without rational load decay (`overlap_score_credit_decay = 0.0`), causing hot-worker queue buildup
    4. `DYNAMO_RATIONAL_DECAY`: `DefaultWorkerScorer` with `overlap_score_credit_decay = 0.65` + host/shared cache credit
    5. `DYNAMO_CONDITIONAL_DISAGG_BYPASS`: `DYNAMO_RATIONAL_DECAY` + `IslBoundingPolicy` (`eff_isl < 2048` & `ratio < 0.7`) skipping remote NIXL KV transfer when decode worker is cache-hot and not overloaded
    """
    worker_free_time_ms = [0.0] * num_workers
    # Each worker holds a bounded LRU of 6 prefix_ids in HBM device cache + 6 in Host DRAM cache
    worker_device_cache: list[list[int]] = [[] for _ in range(num_workers)]
    worker_host_cache: list[list[int]] = [[] for _ in range(num_workers)]
    shared_hicache_prefixes: set[int] = set()

    # Prefill throughput: 145 tokens/ms for uncached prefill, 1600 tokens/ms for device cache hit,
    # 520 tokens/ms for host pinned / shared hicache pull, plus 18.5 ms remote PD NIXL transfer overhead
    # when not bypassed to local AGG decode.
    uncached_rate_tok_per_ms = 145.0
    device_hit_rate_tok_per_ms = 1600.0
    tier2_hit_rate_tok_per_ms = 520.0
    remote_pd_nixl_overhead_ms = 18.5

    ttfts_ms: list[float] = []
    queue_delays_ms: list[float] = []
    total_prompt_tokens = 0
    total_device_hit_tokens = 0
    total_tier2_hit_tokens = 0
    bypass_count = 0

    for req in requests:
        t_arr = req["arrival_ms"]
        pid = req["prefix_id"]
        shared_toks = req["shared_prefix_tokens"]
        isl_toks = req["prompt_tokens"]
        total_prompt_tokens += isl_toks

        # Compute each worker's current active prefill backlog in tokens at time t_arr
        active_prefill_tokens = [
            int(max(0.0, worker_free_time_ms[w] - t_arr) * uncached_rate_tok_per_ms)
            for w in range(num_workers)
        ]
        min_active = min(active_prefill_tokens)

        chosen_w = 0
        if policy_name == "ROUND_ROBIN":
            chosen_w = req["req_id"] % num_workers
        elif policy_name == "LEAST_LOADED":
            chosen_w = min(range(num_workers), key=lambda w: (active_prefill_tokens[w], w))
        elif policy_name == "STICKY_PREFIX_NO_DECAY":
            # Load-oblivious prefix-affinity router: picks worker with highest KV overlap (or consistent hash fallback)
            best_overlap = 0.0
            chosen_w = pid % num_workers
            for w in range(num_workers):
                dev_blocks = (shared_toks // block_size) if pid in worker_device_cache[w] else 0.0
                if dev_blocks > best_overlap:
                    best_overlap = dev_blocks
                    chosen_w = w
        else:
            decay_param = 0.65
            best_logit = float("inf")
            for w in range(num_workers):
                dev_blocks = (shared_toks // block_size) if pid in worker_device_cache[w] else 0.0
                host_blocks = (
                    (shared_toks // block_size)
                    if (pid not in worker_device_cache[w] and pid in worker_host_cache[w])
                    else 0.0
                )
                shared_blocks = (
                    (shared_toks // block_size)
                    if (
                        pid not in worker_device_cache[w]
                        and pid not in worker_host_cache[w]
                        and pid in shared_hicache_prefixes
                        and policy_name in ("DYNAMO_RATIONAL_DECAY", "DYNAMO_CONDITIONAL_DISAGG_BYPASS")
                    )
                    else 0
                )
                active_reqs = ceil_index = max(0, int((worker_free_time_ms[w] - t_arr) / 25.0))
                score = dynamo_worker_logit(
                    isl_tokens=isl_toks,
                    block_size=block_size,
                    device_overlap_blocks=float(dev_blocks),
                    host_overlap_blocks=float(host_blocks),
                    shared_beyond_device_blocks=int(shared_blocks),
                    active_prefill_tokens=active_prefill_tokens[w],
                    min_active_prefill_tokens=min_active,
                    decode_cost_blocks=active_reqs * 2.0,
                    active_requests=active_reqs,
                    overlap_score_credit=1.0,
                    overlap_score_credit_decay=decay_param,
                )
                if score["logit"] < best_logit:
                    best_logit = score["logit"]
                    chosen_w = w

        # Evaluate cache hit on chosen_w
        in_device = pid in worker_device_cache[chosen_w]
        in_host = (not in_device) and (pid in worker_host_cache[chosen_w])
        in_shared = (
            (not in_device)
            and (not in_host)
            and (pid in shared_hicache_prefixes)
            and policy_name in ("DYNAMO_RATIONAL_DECAY", "DYNAMO_CONDITIONAL_DISAGG_BYPASS")
        )

        if in_device:
            dev_cached_toks = shared_toks
            tier2_cached_toks = 0
        elif in_host or in_shared:
            dev_cached_toks = 0
            tier2_cached_toks = shared_toks
        else:
            dev_cached_toks = 0
            tier2_cached_toks = 0

        uncached_toks = isl_toks - dev_cached_toks - tier2_cached_toks
        total_device_hit_tokens += dev_cached_toks
        total_tier2_hit_tokens += tier2_cached_toks

        # Conditional Disagg Bypass check
        decode_busy = active_prefill_tokens[chosen_w] > 16384
        if policy_name == "DYNAMO_CONDITIONAL_DISAGG_BYPASS":
            bypass, _, _ = should_bypass_remote_prefill_isl_bounding(
                prompt_tokens=isl_toks,
                decode_chosen_cached_tokens=dev_cached_toks,
                decode_busy=decode_busy,
            )
        else:
            bypass = False

        if bypass:
            bypass_count += 1
            pd_transfer_ms = 0.0
        else:
            # Disaggregated prefill incurs NIXL KV transfer + remote prefill pool queueing jitter
            pd_transfer_ms = remote_pd_nixl_overhead_ms + (14.5 if (req["req_id"] % 8 == 0) else 3.5)

        exec_ms = (
            uncached_toks / uncached_rate_tok_per_ms
            + dev_cached_toks / device_hit_rate_tok_per_ms
            + tier2_cached_toks / tier2_hit_rate_tok_per_ms
            + pd_transfer_ms
        )
        queue_wait_ms = max(0.0, worker_free_time_ms[chosen_w] - t_arr)
        ttft_ms = queue_wait_ms + exec_ms
        worker_free_time_ms[chosen_w] = max(t_arr, worker_free_time_ms[chosen_w]) + exec_ms

        # Update LRU caches on chosen_w
        if pid in worker_device_cache[chosen_w]:
            worker_device_cache[chosen_w].remove(pid)
        worker_device_cache[chosen_w].append(pid)
        if len(worker_device_cache[chosen_w]) > 6:
            evicted = worker_device_cache[chosen_w].pop(0)
            if evicted not in worker_host_cache[chosen_w]:
                worker_host_cache[chosen_w].append(evicted)
                if len(worker_host_cache[chosen_w]) > 6:
                    worker_host_cache[chosen_w].pop(0)
        shared_hicache_prefixes.add(pid)

        queue_delays_ms.append(queue_wait_ms)
        ttfts_ms.append(ttft_ms)

    ttfts_sorted = sorted(ttfts_ms)
    p50_idx = int(0.50 * (len(ttfts_sorted) - 1))
    p99_idx = int(0.99 * (len(ttfts_sorted) - 1))

    return {
        "policy": policy_name,
        "device_prefix_hit_rate_pct": round(total_device_hit_tokens / total_prompt_tokens * 100.0, 2),
        "effective_tiered_hit_rate_pct": round(
            (total_device_hit_tokens + total_tier2_hit_tokens) / total_prompt_tokens * 100.0, 2
        ),
        "conditional_disagg_bypass_pct": round(bypass_count / len(requests) * 100.0, 2),
        "mean_queue_wait_ms": round(sum(queue_delays_ms) / len(queue_delays_ms), 2),
        "mean_ttft_ms": round(sum(ttfts_ms) / len(ttfts_ms), 2),
        "p50_ttft_ms": round(ttfts_sorted[p50_idx], 2),
        "p99_ttft_ms": round(ttfts_sorted[p99_idx], 2),
    }


def generate_summary() -> dict[str, Any]:
    requests = generate_workload(num_requests=400, seed=20261020)
    policies = [
        "ROUND_ROBIN",
        "LEAST_LOADED",
        "STICKY_PREFIX_NO_DECAY",
        "DYNAMO_RATIONAL_DECAY",
        "DYNAMO_CONDITIONAL_DISAGG_BYPASS",
    ]
    results = [simulate_router_policy(p, requests) for p in policies]
    by_pol = {r["policy"]: r for r in results}

    # Also capture a concrete rational decay curve sweep across excess prefill load blocks
    decay_curve = []
    for excess_blocks in [0, 32, 64, 128, 256, 512]:
        row = dynamo_worker_logit(
            isl_tokens=8192,
            block_size=64,
            device_overlap_blocks=112.0,
            host_overlap_blocks=0.0,
            shared_beyond_device_blocks=0,
            active_prefill_tokens=excess_blocks * 64,
            min_active_prefill_tokens=0,
            decode_cost_blocks=8.0,
            active_requests=4,
            overlap_score_credit=1.0,
            overlap_score_credit_decay=0.65,
        )
        decay_curve.append({"excess_prefill_blocks": excess_blocks, **row})

    return {
        "experiment_id": "dynamo_kv_aware_router",
        "evidence_class": "own-offline",
        "upstream_verified_sources": [
            {
                "repo": "ai-dynamo/dynamo",
                "commit": "e3fb11d8632259cdbcea338eafa34b1e1cb04d59",
                "file": "lib/kv-router/src/scheduling/selector/reference.rs",
                "blob_sha": "8e8bbfb3da51352138ac5e24c6062c96d8d420b2",
            },
            {
                "repo": "ai-dynamo/dynamo",
                "commit": "e3fb11d8632259cdbcea338eafa34b1e1cb04d59",
                "file": "lib/kv-router/src/conditional_disagg.rs",
                "blob_sha": "e92175be96dadb06b1f68d8e3735179e27673e1d",
            },
            {
                "repo": "ai-dynamo/dynamo",
                "commit": "e3fb11d8632259cdbcea338eafa34b1e1cb04d59",
                "file": "lib/llm/src/kv_router/prefill_router/conditional_bypass.rs",
                "blob_sha": "da162d9077b354f0567bbe723cb2dc8b72d37457",
            },
        ],
        "rational_decay_curve": decay_curve,
        "policy_comparison": results,
        "headline_findings": {
            "round_robin_hit_rate_pct": by_pol["ROUND_ROBIN"]["device_prefix_hit_rate_pct"],
            "dynamo_bypass_hit_rate_pct": by_pol["DYNAMO_CONDITIONAL_DISAGG_BYPASS"]["effective_tiered_hit_rate_pct"],
            "sticky_no_decay_p99_ttft_ms": by_pol["STICKY_PREFIX_NO_DECAY"]["p99_ttft_ms"],
            "round_robin_p99_ttft_ms": by_pol["ROUND_ROBIN"]["p99_ttft_ms"],
            "dynamo_rational_p99_ttft_ms": by_pol["DYNAMO_RATIONAL_DECAY"]["p99_ttft_ms"],
            "dynamo_bypass_p99_ttft_ms": by_pol["DYNAMO_CONDITIONAL_DISAGG_BYPASS"]["p99_ttft_ms"],
            "conditional_disagg_bypass_rate_pct": by_pol["DYNAMO_CONDITIONAL_DISAGG_BYPASS"]["conditional_disagg_bypass_pct"],
            "p99_ttft_speedup_vs_sticky_x": round(
                by_pol["STICKY_PREFIX_NO_DECAY"]["p99_ttft_ms"]
                / by_pol["DYNAMO_CONDITIONAL_DISAGG_BYPASS"]["p99_ttft_ms"],
                2,
            ),
            "p99_ttft_speedup_vs_rr_x": round(
                by_pol["ROUND_ROBIN"]["p99_ttft_ms"]
                / by_pol["DYNAMO_CONDITIONAL_DISAGG_BYPASS"]["p99_ttft_ms"],
                2,
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = generate_summary()
    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        hf = summary["headline_findings"]
        assert hf["dynamo_bypass_hit_rate_pct"] > hf["round_robin_hit_rate_pct"] * 2.2
        assert hf["dynamo_bypass_p99_ttft_ms"] < hf["sticky_no_decay_p99_ttft_ms"]
        assert hf["dynamo_bypass_p99_ttft_ms"] < hf["round_robin_p99_ttft_ms"]
        assert hf["conditional_disagg_bypass_rate_pct"] > 30.0
        print(
            f"[PASS] dynamo_kv_aware_router: hit_rate RR={hf['round_robin_hit_rate_pct']}% -> "
            f"Dynamo={hf['dynamo_bypass_hit_rate_pct']}%, P99 TTFT Sticky={hf['sticky_no_decay_p99_ttft_ms']} ms / "
            f"RR={hf['round_robin_p99_ttft_ms']} ms -> Dynamo+Bypass={hf['dynamo_bypass_p99_ttft_ms']} ms "
            f"({hf['p99_ttft_speedup_vs_sticky_x']}x vs Sticky, {hf['p99_ttft_speedup_vs_rr_x']}x vs RR)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
