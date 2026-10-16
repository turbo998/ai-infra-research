"""Evaluate vLLM V1 unified token-budget scheduling, KV block preallocation/eviction invariants, and P/D disaggregation SLO bounds."""

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from ..capacity_roofline.budget import integer
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.capacity_roofline.budget import integer


HERE = Path(__file__).resolve().parent
DEFAULT_PREALLOCATE_TOKENS = 64
LIMITATIONS = [
    "own-offline synthetic arithmetic; no GPU kernel execution, RDMA NIC transfer, or live vLLM server profiling",
    "models static invariants in vllm-project/vllm@25f560a62c4f955672e2c6080b17ab3a48f96201 vllm/v1/core",
    "step latency model uses analytical linear weight-read + per-token compute bounds without CUDA launch jitter",
    "P/D KV transfer bound assumes contiguous layer-pipelined or bulk RDMA transfer at specified unidirectional bandwidth",
]


def _ascii_label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or not value.replace("_", "").isalnum():
        raise ValueError(f"{name}: expected non-empty ASCII alphanumeric/underscore label")
    return value


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def evaluate_inference_slo_case(
    *,
    case_id: str,
    block_size: int,
    num_gpu_blocks: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    num_preallocate_tokens: int = DEFAULT_PREALLOCATE_TOKENS,
    enable_prefix_caching: bool = True,
    running_decode_requests: int,
    running_occupied_blocks: int,
    lowest_priority_running_req_blocks: int,
    waiting_prompt_tokens: int,
    waiting_cached_prefix_blocks: int,
    evictable_cached_prefix_blocks: int,
    kv_bytes_per_token: int,
    base_decode_step_ms: float,
    marginal_prefill_token_ms: float,
    pd_rdma_bandwidth_bytes_per_sec: int,
    tpot_slo_ms: float,
    ttft_slo_ms: float,
) -> dict:
    """Evaluate vLLM V1 unified scheduler step, KV block allocation/preemption, and colocated vs P/D SLOs."""
    label = _ascii_label(case_id, "case_id")
    blk_size = integer(block_size, "block_size")
    pool_blocks = integer(num_gpu_blocks, "num_gpu_blocks")
    max_len = integer(max_model_len, "max_model_len")
    token_budget = integer(max_num_batched_tokens, "max_num_batched_tokens")
    prealloc_tokens = int(num_preallocate_tokens)
    if type(num_preallocate_tokens) is not int or prealloc_tokens < 0:
        raise ValueError("num_preallocate_tokens: expected non-negative int")
    if type(enable_prefix_caching) is not bool:
        raise ValueError("enable_prefix_caching: expected bool")

    running_reqs = int(running_decode_requests)
    occupied_blocks = int(running_occupied_blocks)
    tail_req_blocks = int(lowest_priority_running_req_blocks)
    prompt_tokens = integer(waiting_prompt_tokens, "waiting_prompt_tokens")
    cached_blocks = int(waiting_cached_prefix_blocks)
    evictable_hits = int(evictable_cached_prefix_blocks)
    kv_bpt = integer(kv_bytes_per_token, "kv_bytes_per_token")
    rdma_bw = integer(pd_rdma_bandwidth_bytes_per_sec, "pd_rdma_bandwidth_bytes_per_sec")

    if (
        running_reqs < 0
        or occupied_blocks < 0
        or occupied_blocks > pool_blocks
        or tail_req_blocks < 0
        or cached_blocks < 0
        or evictable_hits < 0
        or evictable_hits > cached_blocks
    ):
        raise ValueError("invalid block or request counts")
    if running_reqs > token_budget:
        raise ValueError("running_decode_requests cannot exceed max_num_batched_tokens in this fixture")

    for name, val in (
        ("base_decode_step_ms", base_decode_step_ms),
        ("marginal_prefill_token_ms", marginal_prefill_token_ms),
        ("tpot_slo_ms", tpot_slo_ms),
        ("ttft_slo_ms", ttft_slo_ms),
    ):
        if not isinstance(val, (int, float)) or isinstance(val, bool) or not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name}: expected finite positive number")

    max_num_blocks_per_req = _cdiv(max_len, blk_size)
    num_preallocate_blocks = _cdiv(prealloc_tokens, blk_size) if prealloc_tokens > 0 else 0
    free_blocks_initial = pool_blocks - occupied_blocks

    # 1. Prefix cache full-hit recompute invariant (scheduler.py:265-275)
    raw_cached_tokens = (cached_blocks * blk_size) if enable_prefix_caching else 0
    if raw_cached_tokens > prompt_tokens:
        raise ValueError("cached tokens cannot exceed waiting_prompt_tokens")

    full_prefix_hit_recompute_triggered = False
    effective_computed_blocks = cached_blocks if enable_prefix_caching else 0
    effective_computed_tokens = raw_cached_tokens
    effective_evictable_hits = evictable_hits if enable_prefix_caching else 0

    if enable_prefix_caching and (prompt_tokens - raw_cached_tokens == 0):
        # scheduler.py:273-275 forces recomputation of the last full block when num_new_tokens == 0
        full_prefix_hit_recompute_triggered = True
        effective_computed_tokens -= blk_size
        effective_computed_blocks -= 1
        if effective_evictable_hits > effective_computed_blocks:
            effective_evictable_hits = effective_computed_blocks

    tokens_to_prefill_total = prompt_tokens - effective_computed_tokens
    remaining_budget_for_waiting = token_budget - running_reqs
    scheduled_prefill_tokens_step1 = min(tokens_to_prefill_total, remaining_budget_for_waiting)

    # 2. KV cache slot allocation & evictable hit guard (kv_cache_manager.py:168-206)
    num_required_blocks_step1 = _cdiv(
        effective_computed_tokens + scheduled_prefill_tokens_step1, blk_size
    )
    num_new_blocks_needed = num_required_blocks_step1 - effective_computed_blocks
    usable_free_blocks_for_new = free_blocks_initial - effective_evictable_hits
    can_allocate_waiting_step1 = num_new_blocks_needed <= usable_free_blocks_for_new

    if can_allocate_waiting_step1:
        free_after_touch = usable_free_blocks_for_new
        if num_new_blocks_needed <= 0:
            allocated_new_blocks_step1 = 0
        else:
            allocated_new_blocks_step1 = min(
                num_new_blocks_needed + num_preallocate_blocks,
                free_after_touch,
                max_num_blocks_per_req - effective_computed_blocks,
            )
        preallocated_extra_blocks = max(0, allocated_new_blocks_step1 - max(0, num_new_blocks_needed))
    else:
        allocated_new_blocks_step1 = 0
        preallocated_extra_blocks = 0

    # Check if a running request needing 1 new block would trigger preemption (scheduler.py:165-179)
    running_step_needs_one_block_preempts = free_blocks_initial == 0
    free_blocks_after_one_preemption = (
        free_blocks_initial + tail_req_blocks if running_step_needs_one_block_preemption_check(free_blocks_initial) else free_blocks_initial
    )

    # 3. Colocated Chunked Prefill vs P/D Disaggregation SLO metrics
    if remaining_budget_for_waiting <= 0:
        num_prefill_chunks = math.ceil(tokens_to_prefill_total / max(1, token_budget))
    else:
        num_prefill_chunks = math.ceil(tokens_to_prefill_total / remaining_budget_for_waiting)

    colocated_mixed_step_ms = float(base_decode_step_ms) + (
        scheduled_prefill_tokens_step1 * float(marginal_prefill_token_ms)
    )
    colocated_tpot_during_prefill_ms = colocated_mixed_step_ms
    colocated_ttft_ms = (num_prefill_chunks - 1) * (
        float(base_decode_step_ms) + remaining_budget_for_waiting * float(marginal_prefill_token_ms)
    ) + (
        float(base_decode_step_ms)
        + (tokens_to_prefill_total - (num_prefill_chunks - 1) * remaining_budget_for_waiting)
        * float(marginal_prefill_token_ms)
    )

    # Disaggregated Prefill/Decode: prefill node executes unchunked tokens_to_prefill_total, transfers KV over RDMA
    pd_kv_transfer_bytes = prompt_tokens * kv_bpt
    pd_kv_transfer_ms = (pd_kv_transfer_bytes / rdma_bw) * 1000.0
    pd_prefill_compute_ms = float(base_decode_step_ms) + (
        tokens_to_prefill_total * float(marginal_prefill_token_ms)
    )
    pd_ttft_ms = pd_prefill_compute_ms + pd_kv_transfer_ms
    pd_decode_node_tpot_ms = float(base_decode_step_ms)

    return {
        "case_id": label,
        "max_num_blocks_per_req": max_num_blocks_per_req,
        "num_preallocate_blocks": num_preallocate_blocks,
        "free_blocks_initial": free_blocks_initial,
        "full_prefix_hit_recompute_triggered": full_prefix_hit_recompute_triggered,
        "effective_computed_tokens": effective_computed_tokens,
        "tokens_to_prefill_total": tokens_to_prefill_total,
        "remaining_budget_for_waiting": remaining_budget_for_waiting,
        "scheduled_prefill_tokens_step1": scheduled_prefill_tokens_step1,
        "num_new_blocks_needed_step1": num_new_blocks_needed,
        "usable_free_blocks_after_evictable_guard": usable_free_blocks_for_new,
        "can_allocate_waiting_step1": can_allocate_waiting_step1,
        "allocated_new_blocks_step1": allocated_new_blocks_step1,
        "preallocated_extra_blocks": preallocated_extra_blocks,
        "running_Needs_block_triggers_preemption": running_step_needs_one_block_preempts,
        "free_blocks_after_one_preemption": free_blocks_after_one_preemption,
        "num_prefill_chunks": num_prefill_chunks,
        "colocated_tpot_during_prefill_ms": round(colocated_tpot_during_prefill_ms, 4),
        "colocated_ttft_ms": round(colocated_ttft_ms, 4),
        "colocated_meets_tpot_slo": colocated_tpot_during_prefill_ms <= float(tpot_slo_ms),
        "colocated_meets_ttft_slo": colocated_ttft_ms <= float(ttft_slo_ms),
        "pd_kv_transfer_bytes": pd_kv_transfer_bytes,
        "pd_kv_transfer_ms": round(pd_kv_transfer_ms, 4),
        "pd_ttft_ms": round(pd_ttft_ms, 4),
        "pd_decode_node_tpot_ms": round(pd_decode_node_tpot_ms, 4),
        "pd_meets_tpot_slo": pd_decode_node_tpot_ms <= float(tpot_slo_ms),
        "pd_meets_ttft_slo": pd_ttft_ms <= float(ttft_slo_ms),
    }


def running_step_needs_one_block_preemption_check(free_blocks_initial: int) -> bool:
    return free_blocks_initial == 0


def calculate(fixture: dict) -> dict:
    """Evaluate all synthetic vLLM V1 scheduling and SLO tradeoff cases."""
    if (
        not isinstance(fixture, dict)
        or type(fixture.get("schema_version")) is not int
        or fixture["schema_version"] != 1
        or fixture.get("synthetic") is not True
        or not isinstance(fixture.get("cases"), list)
        or not fixture["cases"]
        or any(not isinstance(case, dict) for case in fixture["cases"])
    ):
        raise ValueError("fixture: expected version-one nonempty synthetic cases")
    return {
        "schema_version": 1,
        "evidence_status": "own-offline",
        "synthetic": True,
        "cases": [{"inputs": case, "result": evaluate_inference_slo_case(**case)} for case in fixture["cases"]],
        "limitations": LIMITATIONS,
    }


def main(argv: list[str] | None = None) -> int:
    """Generate results.json or verify exact match with --check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        content = json.dumps(calculate(fixture), indent=2, sort_keys=True, allow_nan=False) + "\n"
        output = HERE / "results.json"
        if args.check:
            if not output.is_file() or output.read_bytes() != content.encode("utf-8"):
                print("Inference SLO scheduler check failed: missing or different results.json", file=sys.stderr)
                return 1
        else:
            with output.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print("Inference SLO scheduler failed: invalid fixture or output", file=sys.stderr)
        return 1
    print(
        "own-offline inference slo scheduler verified"
        if args.check
        else "own-offline inference slo scheduler regenerated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
