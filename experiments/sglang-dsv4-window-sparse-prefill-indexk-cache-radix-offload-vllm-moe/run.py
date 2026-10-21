#!/usr/bin/env python3
"""Deterministic stdlib reproduction of Phase 54 Topic 2:
SGLang DeepSeek-V4.1 Request-Window History Deduplication, Cached Index-K Prefill Plan,
Restored Sparse Prefill, Ordered Radix-Cache Offloads, and vLLM Monolithic MoE Simulated Routing.

First-hand upstream commits:
- sgl-project/sglang@f7d4f06242b76615a2cc95e87426b7123a90c7f6 (#45579)
- sgl-project/sglang@bc412213c17cecfa826716e23d5ae00748bbe528 (#45577)
- sgl-project/sglang@97945831f9ec29755099284243ab7d91336d314f (#45573)
- sgl-project/sglang@620718c8ea67d8f1ef827f6be78a77b3a7830b41 (#45575)
- sgl-project/sglang@2c7a9e3b50828720951a5f7e71f42563d2df1b47 (#45559)
- vllm-project/vllm@b0e1f8052abb3de7c2fa8dc6234733104e829804 (#52780)
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


def simulate_dsv41_window_history_and_indexk_cache() -> dict[str, Any]:
    """Model SGLang DSV4.1 WindowLayout.history_src (#45579), prefill_k_slots caching (#45577),
    and restored request-window sparse prefill (#45573)."""
    num_layers = 60
    window = 128
    capacity = 256
    zero_row = 0
    reqs = [2, 5, 9]
    starts = [64, 300, 120]
    floors = [0, 280, 100]

    # Compute history_src once in window_layout() (#45579)
    history_src: list[int] = []
    valid_count = 0
    for req_id, start_pos, floor_pos in zip(reqs, starts, floors):
        for w in range(window):
            h_pos = start_pos - window + w
            is_valid = (h_pos >= 0) and (h_pos >= floor_pos)
            if is_valid:
                valid_count += 1
                src_row = req_id * capacity + (h_pos % capacity)
            else:
                src_row = zero_row
            history_src.append(src_row)

    # Before #45579: computed per layer (60 times). After #45579: computed once in window_layout().
    history_src_ops_before = num_layers
    history_src_ops_after = 1

    # Model prefill_k_slots and low_ratio_index_plans caching (#45577)
    # Suppose 48 low-ratio layers share compression ratios {4: 24 layers, 8: 24 layers}
    seq_lens_cpu = [512, 1024, 768]
    req_pool_indices = [1, 4, 7]
    req_to_token_stride = 2048

    def compute_prefill_k_slots(ratio: int) -> list[int]:
        lens_per_req = [s // ratio for s in seq_lens_cpu]
        starts_acc = list(itertools.accumulate(lens_per_req, initial=0))[:-1]
        slots: list[int] = []
        for r_idx, length in enumerate(lens_per_req):
            pool_row = req_pool_indices[r_idx]
            for pos in range(length):
                slots.append(pool_row * req_to_token_stride + pos)
        assert len(slots) == sum(lens_per_req)
        assert starts_acc[1] == lens_per_req[0]
        return slots

    slots_r4 = compute_prefill_k_slots(4)
    slots_r8 = compute_prefill_k_slots(8)
    low_ratio_layers = 48
    distinct_low_ratios = 2  # ratio=4 and ratio=8
    h2d_gathers_before = low_ratio_layers * 2  # async_h2d(lens) + async_h2d(starts) per layer
    h2d_gathers_after = distinct_low_ratios * 2
    plan_builds_before = low_ratio_layers
    plan_builds_after = distinct_low_ratios

    # Model _use_sparse_prefill with request_window active (#45573)
    large_indexer_threshold = 256
    num_qo_tokens = 2048
    request_window_active = True
    use_sparse_prefill_before = (num_qo_tokens > large_indexer_threshold) and (not request_window_active)
    use_sparse_prefill_after = num_qo_tokens > large_indexer_threshold

    return {
        "num_layers": num_layers,
        "history_rows_total": len(history_src),
        "history_valid_rows": valid_count,
        "history_src_ops_before": history_src_ops_before,
        "history_src_ops_after": history_src_ops_after,
        "history_src_reduction_factor": history_src_ops_before // history_src_ops_after,
        "slots_r4_count": len(slots_r4),
        "slots_r8_count": len(slots_r8),
        "h2d_gathers_before": h2d_gathers_before,
        "h2d_gathers_after": h2d_gathers_after,
        "plan_builds_before": plan_builds_before,
        "plan_builds_after": plan_builds_after,
        "use_sparse_prefill_before": use_sparse_prefill_before,
        "use_sparse_prefill_after": use_sparse_prefill_after,
        "invariants_passed": (
            history_src_ops_after == 1
            and len(slots_r4) == 576
            and len(slots_r8) == 288
            and h2d_gathers_before // h2d_gathers_after == 24
            and not use_sparse_prefill_before
            and use_sparse_prefill_after
        ),
    }


def simulate_radix_cache_offload_ordering() -> dict[str, Any]:
    """Model UnifiedCacheLinker.record_offload_ready() across scheduler_stream and forward_stream (#45559)."""
    # Timeline (us):
    # scheduler_stream finishes metadata prep at t=15 us (runs ahead due to prefill read-done)
    # forward_stream writes KV pages from t=10 us to t=85 us
    scheduler_stream_done_us = 15.0
    forward_stream_write_done_us = 85.0

    # Before #45559: offload recorded event on current (scheduler) stream only
    old_events_waited_us = [scheduler_stream_done_us]
    old_dma_read_start_us = max(old_events_waited_us)
    old_reads_unwritten_kv = old_dma_read_start_us < forward_stream_write_done_us

    # After #45559: record_offload_ready() records on current_stream AND forward_stream (if distinct)
    new_events_waited_us = [scheduler_stream_done_us, forward_stream_write_done_us]
    new_dma_read_start_us = max(new_events_waited_us)
    new_reads_unwritten_kv = new_dma_read_start_us < forward_stream_write_done_us

    return {
        "scheduler_stream_done_us": scheduler_stream_done_us,
        "forward_stream_write_done_us": forward_stream_write_done_us,
        "old_dma_read_start_us": old_dma_read_start_us,
        "new_dma_read_start_us": new_dma_read_start_us,
        "old_reads_unwritten_kv": old_reads_unwritten_kv,
        "new_reads_unwritten_kv": new_reads_unwritten_kv,
        "invariants_passed": old_reads_unwritten_kv and (not new_reads_unwritten_kv),
    }


def simulate_moe_routing_simulation_guards() -> dict[str, Any]:
    """Model SGLang DSV4.1 vision_topk packed-ids guard (#45575) and vLLM monolithic MoE
    simulated routing separation (#52780)."""
    # 1. SGLang vision_topk (#45575): FlashInfer MXFP4 packs weight/flag bits into topk_ids;
    # routing simulation overwrites topk_ids with plain expert indices [0..E-1].
    def select_sglang_vision_router_path(
        is_flashinfer_mxfp4: bool,
        routing_simulation: bool,
        use_mega_moe: bool = False,
    ) -> str:
        old_packed = is_flashinfer_mxfp4 and (not use_mega_moe)
        new_packed = is_flashinfer_mxfp4 and (not use_mega_moe) and (not routing_simulation)
        return "old_packed_new_unpacked" if (old_packed and not new_packed) else "unchanged"

    vision_sim_transition = select_sglang_vision_router_path(
        is_flashinfer_mxfp4=True, routing_simulation=True
    )

    # 2. vLLM monolithic MoE backends (#52780):
    # Monolithic kernels (TRTLLM FP8 / MXINT4 / NVFP4) run routing inside the fused C++ kernel
    # and cannot invoke Python DistributionBasedRouting.
    def resolve_vllm_moe_backend(
        base_routing_method: str,
        simulated_routing_active: bool,
        expert_substitution: bool = False,
    ) -> dict[str, Any]:
        # RoutingSimulatorRouter wraps base router:
        # supports_monolithic_kernels = False when simulated_routing_active is True
        supports_monolithic = not simulated_routing_active
        kernel_routing_method = base_routing_method
        require_decomposed = expert_substitution or (not supports_monolithic)
        selected_backend = "decomposed_modular_moe" if require_decomposed else "monolithic_trtllm_moe"
        return {
            "kernel_routing_method": kernel_routing_method,
            "supports_monolithic_kernels": supports_monolithic,
            "require_decomposed_backend": require_decomposed,
            "selected_backend": selected_backend,
        }

    prod_case = resolve_vllm_moe_backend("DeepSeekV3", simulated_routing_active=False)
    sim_case = resolve_vllm_moe_backend("DeepSeekV3", simulated_routing_active=True)

    return {
        "sglang_vision_sim_transition": vision_sim_transition,
        "vllm_production_moe": prod_case,
        "vllm_simulated_moe": sim_case,
        "invariants_passed": (
            vision_sim_transition == "old_packed_new_unpacked"
            and prod_case["selected_backend"] == "monolithic_trtllm_moe"
            and sim_case["selected_backend"] == "decomposed_modular_moe"
            and not sim_case["supports_monolithic_kernels"]
        ),
    }


def run_experiment() -> dict[str, Any]:
    dsv4 = simulate_dsv41_window_history_and_indexk_cache()
    radix = simulate_radix_cache_offload_ordering()
    moe_sim = simulate_moe_routing_simulation_guards()
    all_passed = (
        dsv4["invariants_passed"]
        and radix["invariants_passed"]
        and moe_sim["invariants_passed"]
    )
    return {
        "experiment": "sglang_dsv4_window_sparse_prefill_indexk_cache_radix_offload_vllm_moe",
        "upstream_commits": [
            "sgl-project/sglang@f7d4f06242b76615a2cc95e87426b7123a90c7f6",
            "sgl-project/sglang@bc412213c17cecfa826716e23d5ae00748bbe528",
            "sgl-project/sglang@97945831f9ec29755099284243ab7d91336d314f",
            "sgl-project/sglang@620718c8ea67d8f1ef827f6be78a77b3a7830b41",
            "sgl-project/sglang@2c7a9e3b50828720951a5f7e71f42563d2df1b47",
            "vllm-project/vllm@b0e1f8052abb3de7c2fa8dc6234733104e829804",
        ],
        "dsv41_window_and_indexk": dsv4,
        "radix_cache_offload_ordering": radix,
        "moe_routing_simulation_guards": moe_sim,
        "all_invariants_passed": all_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify all invariants and write summary.json")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not summary["all_invariants_passed"]:
        raise AssertionError("Experiment invariants failed")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
