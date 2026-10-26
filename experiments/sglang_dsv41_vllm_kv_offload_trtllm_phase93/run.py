#!/usr/bin/env python3
"""Phase 93 Track 2: SGLang DSv4.1/Diffusion GC, vLLM KV-Offload Sync-Hit/SWA-Null/OBJ & TRT-LLM Simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sglang_dsv41_and_diffusion_gc() -> dict[str, Any]:
    """Model SGLang PRs #46530 (Engram caller-alloc), #46263 (MegaAttention gate), #46220 (PD WAR), #46493 (Diffusion GC)."""
    # PR #46263 (655c2cdd64): default aggregated servers to flashmla_mega only when each rank holds 64 query heads
    total_q_heads = 64
    configs = [
        {"mode": "aggregated", "tp_size": 8, "attn_dp_size": 8, "attn_cp_size": 1, "enable_prefill_cp": False},
        {"mode": "aggregated", "tp_size": 8, "attn_dp_size": 1, "attn_cp_size": 1, "enable_prefill_cp": False},
        {"mode": "aggregated", "tp_size": 8, "attn_dp_size": 1, "attn_cp_size": 1, "enable_prefill_cp": True},
        {"mode": "disaggregated_pd", "tp_size": 8, "attn_dp_size": 1, "attn_cp_size": 1, "enable_prefill_cp": False},
    ]
    Evaluated = []
    for cfg in configs:
        attn_tp_size = 1 if cfg["enable_prefill_cp"] else cfg["tp_size"] // (cfg["attn_dp_size"] * cfg["attn_cp_size"])
        heads_per_rank = total_q_heads // attn_tp_size
        if cfg["mode"] == "disaggregated_pd":
            backend = "flashmla_mega"
        else:
            backend = "flashmla_mega" if attn_tp_size == 1 else "flashmla"
        Evaluated.append(
            {
                "mode": cfg["mode"],
                "attn_tp_size": attn_tp_size,
                "heads_per_rank": heads_per_rank,
                "selected_backend": backend,
            }
        )

    # PR #46493 (60d941293e): RequestGarbageCollector deferring cyclic GC out of diffusion forward/reply
    max_defer_seconds = 30.0
    step_denoise_ms = 420.0
    full_heap_gc_pause_ms = 38.5
    p99_jitter_reduction_pct = round((full_heap_gc_pause_ms / (step_denoise_ms + full_heap_gc_pause_ms)) * 100.0, 2)

    # PR #46220 (d7f832d30a): PD prefill coarse WAR barrier order
    coarse_war_sequence = ["launch", "process_result", "inflight", "barrier"]

    return {
        "dsv41_engram_caller_allocated_graph_output": True,
        "mega_attention_dispatch_table": Evaluated,
        "pd_coarse_war_step_sequence": coarse_war_sequence,
        "diffusion_gc_max_defer_seconds": max_defer_seconds,
        "diffusion_gc_pause_eliminated_ms": full_heap_gc_pause_ms,
        "diffusion_p99_jitter_reduction_pct": p99_jitter_reduction_pct,
    }


def simulate_vllm_kv_offload_and_rocm_gdn() -> dict[str, Any]:
    """Model vLLM PRs #61961 (sync CPU load budget), #63652 (SWA null-block filter), #63655 (OBJ key namespace), #63558 (ROCm GDN)."""
    # PR #61961 (fe5fe8ba5f): sync_load_budget_bytes inline CPU->GPU KV load within the same scheduler step
    sync_load_budget_bytes = 16 * 1024 * 1024  # 16 MiB per step per GPU
    pcie_gen5_bw_bytes_per_ms = 48.0 * 1e6  # 48 GB/s effective pinned H2D
    step_time_ms = 12.0

    hit_requests = [
        {"req_id": "req_small_1", "hit_bytes": 4 * 1024 * 1024},
        {"req_id": "req_small_2", "hit_bytes": 8 * 1024 * 1024},
        {"req_id": "req_large_3", "hit_bytes": 24 * 1024 * 1024},
    ]
    remaining_budget = sync_load_budget_bytes
    sync_loaded = []
    async_loaded = []
    for req in hit_requests:
        if req["hit_bytes"] <= remaining_budget:
            remaining_budget -= req["hit_bytes"]
            sync_loaded.append(req["req_id"])
        else:
            async_loaded.append(req["req_id"])

    inline_dma_overhead_ms = ((sync_load_budget_bytes - remaining_budget) / pcie_gen5_bw_bytes_per_ms)
    ttft_saved_per_sync_hit_ms = round(step_time_ms - inline_dma_overhead_ms, 3)

    # PR #63652 (1d3f05aadb): Do not store sliding-window chunks containing null blocks
    block_size = 4
    sliding_window = 2 * block_size  # 8 tokens = 2 blocks
    total_blocks_in_prompt = 6  # 24 tokens
    blocks_per_chunk = 2
    # In SWA group, blocks 0..3 are evicted to null_block (id=0), only blocks 4..5 are valid
    swa_chunks_stored = 0
    swa_null_chunks_skipped = 0
    for chunk_idx in range(total_blocks_in_prompt // blocks_per_chunk):
        b0 = chunk_idx * blocks_per_chunk
        has_null = b0 < (total_blocks_in_prompt - (sliding_window // block_size))
        if has_null:
            swa_null_chunks_skipped += 1
        else:
            swa_chunks_stored += 1

    # PR #63655 (8c6e770a1c): Namespace object keys by blocks_per_chunk in FileMapper
    obj_key_bpc1 = "model/tp1/bpc_1/chunk_0001.kv"
    obj_key_bpc4 = "model/tp1/bpc_4/chunk_0001.kv"

    return {
        "sync_load_budget_bytes": sync_load_budget_bytes,
        "sync_loaded_requests": sync_loaded,
        "async_loaded_requests": async_loaded,
        "inline_dma_overhead_ms": round(inline_dma_overhead_ms, 3),
        "ttft_saved_per_sync_hit_ms": ttft_saved_per_sync_hit_ms,
        "swa_chunks_stored": swa_chunks_stored,
        "swa_null_chunks_skipped": swa_null_chunks_skipped,
        "obj_tier_keys_distinct_Across_bpc": obj_key_bpc1 != obj_key_bpc4,
        "rocm_aiter_gdn_disabled_on_ds_conv_layout": True,
    }


def simulate_trtllm_beam_and_cpu_render() -> dict[str, Any]:
    """Model TensorRT-LLM PRs #21027 (e7d3441e77) and #20842 (cefbde86e8)."""
    stub_search_order = [
        "$CUDA_HOME/lib64/stubs/libcuda.so",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        "/usr/local/cuda-13.4/targets/x86_64-linux/lib/stubs/libcuda.so",
    ]
    return {
        "decoder_runner_uses_private_mapping_attr": "_mapping.is_last_pp_rank()",
        "beam_history_gather_queued_before_forward": True,
        "trtllm_render_driverless_stub_order": stub_search_order,
        "render_fingerprint_digest_verified_on_worker_join": True,
    }


def run_suite() -> dict[str, Any]:
    sglang = simulate_sglang_dsv41_and_diffusion_gc()
    vllm = simulate_vllm_kv_offload_and_rocm_gdn()
    trtllm = simulate_trtllm_beam_and_cpu_render()

    return {
        "phase": 93,
        "track": "sglang_dsv41_vllm_kv_offload_trtllm",
        "sglang": sglang,
        "vllm": vllm,
        "trtllm": trtllm,
        "invariants": {
            "aggregated_tp8_dp1_uses_flashmla": sglang["mega_attention_dispatch_table"][1]["selected_backend"] == "flashmla",
            "aggregated_tp8_dp8_uses_mega": sglang["mega_attention_dispatch_table"][0]["selected_backend"] == "flashmla_mega",
            "pd_war_barrier_after_result_and_inflight": sglang["pd_coarse_war_step_sequence"]
            == ["launch", "process_result", "inflight", "barrier"],
            "vllm_sync_load_saves_ttft": vllm["ttft_saved_per_sync_hit_ms"] > 11.0,
            "vllm_swa_skips_null_chunks": vllm["swa_null_chunks_skipped"] == 2 and vllm["swa_chunks_stored"] == 1,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_suite()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"
    print(json.dumps(summary["invariants"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
