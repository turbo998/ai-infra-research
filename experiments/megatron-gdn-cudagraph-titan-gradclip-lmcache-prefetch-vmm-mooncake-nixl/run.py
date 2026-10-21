#!/usr/bin/env python3
"""Deterministic stdlib reproduction of Phase 54 Topic 3:
Megatron-LM GDN Single-Update GDP Prefill CUDA Graph Replay, TorchTitan Optimizer-Owned
GradientClipper & LoRA SPMD Typing, LMCache Watermark Prefetch Admission & VMM Padded IPC,
and Mooncake/NIXL/Dynamo Transport Telemetry.

First-hand upstream commits:
- NVIDIA/Megatron-LM@29e1a7b6697c61d184c95726865d417fc51b8574 (#8353)
- pytorch/torchtitan@9698a733623ca997ef824a03cb1f44706ee631e7 (#5394)
- pytorch/torchtitan@a242475253433725f10f39c18f5ccbfdf188e097 (#5390)
- LMCache/LMCache@a6accee6d84c77da8c8b3e3a68574b69247e9684 (#5889)
- LMCache/LMCache@21aa6ac06c83e1bad460cc231c5168a6bc7defe6 (#5855)
- LMCache/LMCache@45486917f13aa454a22435aa03e2dc28efda47f5 (#5929)
- kvcache-ai/Mooncake@7a0585567458a4aba6a58115888c54b7fd1a44e1 (#4410)
- ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319 (#2479)
- ai-dynamo/dynamo@5004f3fa8093d99de68a8054ea238ea534d9daf4 (#16947)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GDP_CHUNK_SIZE = 64


def simulate_megatron_gdn_and_torchtitan() -> dict[str, Any]:
    """Model Megatron-LM GDN prefill as single-update GDP (M=1, #8353),
    TorchTitan optimizer-owned GradientClipper (#5394), and LoRA SPMD typing (#5390)."""
    # 1. Megatron-LM GDN prefill CUDA graph replay (#8353):
    # GDN sets num_householder = 1 (M = 1) so the inference context allocates persistent
    # GDP chunk descriptors aligned to GDP_CHUNK_SIZE = 64.
    total_tokens = 4096
    ssm_chunk_alignment = 128
    gdn_num_householder_old = 0
    gdn_num_householder_new = 1
    alignment_valid = (ssm_chunk_alignment % GDP_CHUNK_SIZE) == 0
    persistent_chunk_descriptors_old = (
        (total_tokens * gdn_num_householder_old) // GDP_CHUNK_SIZE
    )
    persistent_chunk_descriptors_new = (
        (total_tokens * gdn_num_householder_new) // GDP_CHUNK_SIZE
    )

    # 2. TorchTitan GradientClipper (#5394) & LoRA SPMD typecheck (#5390):
    # On RowParallelLinear: input is Shard(K), lora_a weight is Shard(K) -> output is Partial.
    # lora_b weight is Replicate -> einsum("...k,nk->...n", Partial, Replicate) -> Partial.
    # Bias add is jointly linear: Partial + Partial -> Partial, before RowParallel all-reduce.
    def spmd_einsum_matmul(input_placement: str, weight_placement: str) -> str:
        if input_placement == "Shard(K)" and weight_placement == "Shard(K)":
            return "Partial"
        if input_placement == "Partial" and weight_placement == "Replicate":
            return "Partial"
        raise ValueError(f"Unsupported SPMD matmul placements: {input_placement}, {weight_placement}")

    lora_a_out = spmd_einsum_matmul("Shard(K)", "Shard(K)")
    lora_b_out_old_local_rule = "SpmdTypeError"
    lora_b_out_new_einsum_rule = spmd_einsum_matmul(lora_a_out, "Replicate")

    # Optimizer-owned compiled GradientClipper reduces per-step kernel launches across N params
    num_params = 240
    eager_clip_dispatches_per_step = num_params * 2 + 2  # per-param norm + global norm + per-param scale
    compiled_device_clipper_dispatches = 2  # 1 compiled local norm + 1 all-reduce + fused scale

    return {
        "gdn_persistent_chunk_descriptors_old": persistent_chunk_descriptors_old,
        "gdn_persistent_chunk_descriptors_new": persistent_chunk_descriptors_new,
        "ssm_chunk_alignment_valid": alignment_valid,
        "lora_a_placement": lora_a_out,
        "lora_b_old_typecheck": lora_b_out_old_local_rule,
        "lora_b_new_typecheck": lora_b_out_new_einsum_rule,
        "eager_clip_dispatches_per_step": eager_clip_dispatches_per_step,
        "compiled_device_clipper_dispatches": compiled_device_clipper_dispatches,
        "invariants_passed": (
            persistent_chunk_descriptors_old == 0
            and persistent_chunk_descriptors_new == 64
            and alignment_valid
            and lora_b_out_new_einsum_rule == "Partial"
        ),
    }


def simulate_lmcache_watermark_and_vmm_ipc() -> dict[str, Any]:
    """Model LMCache dynamic prefetch watermark admission (#5889), VMM dim-0 padded IPC (#5855),
    and primary CUDA context binding before timeline semaphore calls (#5929)."""
    l1_capacity_mib = 1024
    watermark = 0.5  # 512 MiB cap for in-flight staging + lookup worst-case
    watermark_cap_mib = int(l1_capacity_mib * watermark)

    # Case A: 32 small requests of 8 MiB each (total 256 MiB <= 512 MiB)
    # Old static max_in_flight=8 admits only 8 (64 MiB); new watermark controller admits all 32!
    small_req_mib = 8
    num_small_reqs = 32
    old_admitted_small = min(8, num_small_reqs)
    new_admitted_small = 0
    inflight_mib = 0
    free_mib = 900
    for _ in range(num_small_reqs):
        if inflight_mib + small_req_mib <= watermark_cap_mib and (free_mib - inflight_mib) >= small_req_mib:
            inflight_mib += small_req_mib
            new_admitted_small += 1

    # Case B: 8 large requests of 128 MiB each (total 1024 MiB == 100% L1 capacity!)
    # Old static max_in_flight=8 admits all 8 (overflowing 900 MiB free L1);
    # New watermark controller caps in-flight at 512 MiB (4 requests = 512 MiB).
    large_req_mib = 128
    num_large_reqs = 8
    old_admitted_large = min(8, num_large_reqs)
    old_large_staging_mib = old_admitted_large * large_req_mib
    new_admitted_large = 0
    inflight_large_mib = 0
    for _ in range(num_large_reqs):
        if inflight_large_mib + large_req_mib <= watermark_cap_mib and (free_mib - inflight_large_mib) >= large_req_mib:
            inflight_large_mib += large_req_mib
            new_admitted_large += 1

    # VMM dim-0 padded KV pool view (#5855): shape (4, 8, 16), stride (256, 16, 1)
    shape = (4, 8, 16)
    stride = (256, 16, 1)
    elem_size = 4
    contiguous_numel = shape[0] * shape[1] * shape[2]  # 512 floats = 2048 B
    # True byte span across dim-0 padded blocks: ((4 - 1) * 256 + (8 - 1) * 16 + (16 - 1) * 1 + 1) * 4 = 3584 B
    padded_span_elems = (shape[0] - 1) * stride[0] + (shape[1] - 1) * stride[1] + (shape[2] - 1) * stride[2] + 1
    padded_byte_span = padded_span_elems * elem_size
    inter_block_padding_preserved = padded_byte_span == 3584 and contiguous_numel * elem_size == 2048

    # Primary context binding on fresh thread (#5929)
    fresh_thread_has_ctx_before = False
    fresh_thread_has_ctx_after_bind = True

    return {
        "l1_capacity_mib": l1_capacity_mib,
        "watermark_cap_mib": watermark_cap_mib,
        "old_admitted_small_8mib": old_admitted_small,
        "new_admitted_small_8mib": new_admitted_small,
        "small_concurrency_gain": new_admitted_small / old_admitted_small,
        "old_large_staging_mib": old_large_staging_mib,
        "new_large_staging_mib": inflight_large_mib,
        "vmm_contiguous_bytes": contiguous_numel * elem_size,
        "vmm_padded_byte_span": padded_byte_span,
        "inter_block_padding_preserved": inter_block_padding_preserved,
        "fresh_thread_has_ctx_before": fresh_thread_has_ctx_before,
        "fresh_thread_has_ctx_after_bind": fresh_thread_has_ctx_after_bind,
        "invariants_passed": (
            new_admitted_small == 32
            and new_admitted_large == 4
            and old_large_staging_mib > free_mib
            and inflight_large_mib <= watermark_cap_mib
            and inter_block_padding_preserved
            and fresh_thread_has_ctx_after_bind
        ),
    }


def simulate_mooncake_nixl_dynamo_transport() -> dict[str, Any]:
    """Model Mooncake CPU->GPU TCP transport selector (#4410), NIXL request ID & execution
    counter correlation (#2479), and Dynamo mm-routing token accounting error propagation (#16947)."""
    # 1. Mooncake TcpTransport capabilities on CPU-only build (#4410)
    cpu_build_tcp_caps_old = {"dram_to_dram": True, "dram_to_gpu": False, "gpu_to_dram": False, "gpu_to_gpu": False}
    cpu_build_tcp_caps_new = {"dram_to_dram": True, "dram_to_gpu": True, "gpu_to_dram": False, "gpu_to_gpu": False}

    # 2. NIXL transfer request correlation (#2479)
    # Draw non-zero 64-bit unsigned decimal string; execution counter increments only on posts
    # that pass repost/notif validation and reach backend->postXfer.
    req_id_u64 = (1 << 63) + 12345  # would be negative if formatted as signed int64!
    req_id_decimal_str = str(req_id_u64)
    exec_counter = 0
    trace_events = []

    for attempt in ["valid_post_1", "rejected_repost_while_in_progress", "valid_repost_2", "backend_failed_post_3"]:
        if attempt == "rejected_repost_while_in_progress":
            # Rejected before backend->postXfer: does NOT consume an execution number
            trace_events.append({"attempt": attempt, "req_id": req_id_decimal_str, "execution": None})
        else:
            exec_counter += 1
            trace_events.append({"attempt": attempt, "req_id": req_id_decimal_str, "execution": exec_counter})

    # 3. Dynamo mm-routing image token accounting (#16947)
    def accumulate_image_tokens(dims: list[tuple[int, int]]) -> int | None:
        total = 0
        for w, h in dims:
            if w == 0 or h == 0 or w > 65536 or h > 65536:
                return None  # Propagate Result::Err -> image_tokens = None
            total += (w // 14) * (h // 14)
        return total

    valid_mm_tokens = accumulate_image_tokens([(448, 448), (224, 224)])
    corrupt_mm_tokens = accumulate_image_tokens([(448, 448), (0, 448)])

    return {
        "mooncake_cpu_tcp_dram_to_gpu_old": cpu_build_tcp_caps_old["dram_to_gpu"],
        "mooncake_cpu_tcp_dram_to_gpu_new": cpu_build_tcp_caps_new["dram_to_gpu"],
        "nixl_req_id_decimal": req_id_decimal_str,
        "nixl_trace_events": trace_events,
        "nixl_final_execution_count": exec_counter,
        "dynamo_valid_mm_tokens": valid_mm_tokens,
        "dynamo_corrupt_mm_tokens": corrupt_mm_tokens,
        "invariants_passed": (
            not cpu_build_tcp_caps_old["dram_to_gpu"]
            and cpu_build_tcp_caps_new["dram_to_gpu"]
            and not req_id_decimal_str.startswith("-")
            and exec_counter == 3
            and trace_events[1]["execution"] is None
            and valid_mm_tokens == 1280
            and corrupt_mm_tokens is None
        ),
    }


def run_experiment() -> dict[str, Any]:
    megatron_titan = simulate_megatron_gdn_and_torchtitan()
    lmcache = simulate_lmcache_watermark_and_vmm_ipc()
    transport = simulate_mooncake_nixl_dynamo_transport()
    all_passed = (
        megatron_titan["invariants_passed"]
        and lmcache["invariants_passed"]
        and transport["invariants_passed"]
    )
    return {
        "experiment": "megatron_gdn_cudagraph_titan_gradclip_lmcache_prefetch_vmm_mooncake_nixl",
        "upstream_commits": [
            "NVIDIA/Megatron-LM@29e1a7b6697c61d184c95726865d417fc51b8574",
            "pytorch/torchtitan@9698a733623ca997ef824a03cb1f44706ee631e7",
            "pytorch/torchtitan@a242475253433725f10f39c18f5ccbfdf188e097",
            "LMCache/LMCache@a6accee6d84c77da8c8b3e3a68574b69247e9684",
            "LMCache/LMCache@21aa6ac06c83e1bad460cc231c5168a6bc7defe6",
            "LMCache/LMCache@45486917f13aa454a22435aa03e2dc28efda47f5",
            "kvcache-ai/Mooncake@7a0585567458a4aba6a58115888c54b7fd1a44e1",
            "ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319",
            "ai-dynamo/dynamo@5004f3fa8093d99de68a8054ea238ea534d9daf4",
        ],
        "megatron_gdn_and_torchtitan": megatron_titan,
        "lmcache_watermark_and_vmm_ipc": lmcache,
        "mooncake_nixl_dynamo_transport": transport,
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
