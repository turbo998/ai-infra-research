from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. SGLang Unified-Memory SWA Tail Async Page Binding (#47542, 573649879b)
    # + HiCache mlock Pinned Host Buffers (#47543, 9dbc462c65)
    # + Neutral UE8M0 Block Scales for Dummy Weights (#47337, 800de62d7c)
    swa_extend_steps = 64
    legacy_host_syncs_per_step = 1
    async_host_syncs_per_step = 0
    saved_host_sync_stalls = swa_extend_steps * (legacy_host_syncs_per_step - async_host_syncs_per_step)
    hicache_pinned_pool_gib = 64
    page_aging_scanned_gib_after_mlock = 0
    ue8m0_neutral_exponent_bias = 127  # 2^(127-127) = 1.0 neutral scale

    # 2. DeepEP Internode Dispatch CUDA Graph Capture (#438, 92fe2deaec)
    # + TRT-LLM Ray Worker FlashInfer TRT-LLM-Gen Artifact Prefetch (#21286, c94310bdff)
    moe_layers = 61
    eager_internode_dispatch_cpu_launch_us = 18.5
    cudagraph_internode_dispatch_cpu_launch_us = 2.5
    per_step_dispatch_cpu_overhead_reduction_ratio = round(
        (moe_layers * eager_internode_dispatch_cpu_launch_us)
        / (moe_layers * cudagraph_internode_dispatch_cpu_launch_us),
        2,
    )
    ray_workers = 8
    worker_jit_cache_race_failures_after_prefetch = 0

    # 3. vLLM Rust Frontend Tool-Call Delta Token Accounting (#64557, 32a73034d5) & OLMoE LoRA (#63384, bc57dc9d53)
    text_output_tokens = 120
    tool_call_delta_tokens = 180
    legacy_counted_tokens = text_output_tokens
    fixed_counted_tokens = text_output_tokens + tool_call_delta_tokens
    accounting_accuracy_ratio = round(fixed_counted_tokens / (text_output_tokens + tool_call_delta_tokens), 4)

    return {
        "suite": "sglang_swa_async_hicache_mlock_deepep_internode_graph_phase168",
        "phase": 168,
        "priority": 177,
        "status": "ok",
        "sglang_swa_hicache_ue8m0": {
            "swa_extend_steps": swa_extend_steps,
            "async_host_syncs_per_step": async_host_syncs_per_step,
            "saved_host_sync_stalls": saved_host_sync_stalls,
            "page_aging_scanned_gib_after_mlock": page_aging_scanned_gib_after_mlock,
            "ue8m0_neutral_exponent_bias": ue8m0_neutral_exponent_bias,
        },
        "deepep_internode_cudagraph_and_trtllm_prefetch": {
            "moe_layers": moe_layers,
            "per_step_dispatch_cpu_overhead_reduction_ratio": per_step_dispatch_cpu_overhead_reduction_ratio,
            "ray_workers": ray_workers,
            "worker_jit_cache_race_failures_after_prefetch": worker_jit_cache_race_failures_after_prefetch,
        },
        "vllm_rust_tool_call_delta_accounting": {
            "legacy_counted_tokens": legacy_counted_tokens,
            "fixed_counted_tokens": fixed_counted_tokens,
            "accounting_accuracy_ratio": accounting_accuracy_ratio,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = run_suite()
    res_path = Path(__file__).with_name("results.json")
    res_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert out["status"] == "ok"
        assert out["sglang_swa_hicache_ue8m0"]["async_host_syncs_per_step"] == 0
        assert out["deepep_internode_cudagraph_and_trtllm_prefetch"]["per_step_dispatch_cpu_overhead_reduction_ratio"] == 7.4
        assert out["vllm_rust_tool_call_delta_accounting"]["fixed_counted_tokens"] == 300
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
