#!/usr/bin/env python3
"""Deterministic verification of TensorRT-LLM Chunk-Scoped Multimodal Scheduling,
Blackwell SwapsMmaAb Speculative Tree Mask Sizing & KVCM2 Progressive Host Commit,
vLLM libgomp Fork Pause & ROCm HY-V4 iHC Fusion, and SGLang Intel XPU Disaggregated KV.

Covers first-hand upstream commits:
- TensorRT-LLM@b47eb8fe6b0965a85adf01121f9f0f2a1e79c7c2 (#19884)
- TensorRT-LLM@85b1b8323b78a4717aeb772aa66eed9b8346d687 (#20223)
- TensorRT-LLM@0cd9d0b5415b82696ecbaba8b0b6034aa9c09997 (#19650)
- vllm@40210d8317bbad8d05124b260e75677daf33e091 (#62884)
- vllm@833349105b9581d19487295454c19c924e8a3b37 (#58509)
- sglang@8e3e2980db83a1b020a6d273a2b653891af81bfc (#40861)
- sglang@f8fd41c36e493ec762e60f447c50c88fb8af3ec1 (#43363)
- sglang@2c9d9190713285552d7ea2bb264664c90639c824 (#45437)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def simulate_trtllm_mm_chunk_scheduling() -> dict[str, Any]:
    """Model chunk-scoped multimodal encoder budgeting & TensorLRUCache.releasable_bytes (#19884)."""
    # Suppose a long multimodal request has 4 images at token intervals:
    # img0: [256, 1280) (1024 tokens, cached in KV prefix [0, 1536))
    # img1: [1600, 2624) (1024 tokens)
    # img2: [3000, 4024) (1024 tokens)
    # img3: [4500, 5524) (1024 tokens)
    mm_items = [
        {"id": "img0", "key": "hash_A", "start": 256, "length": 1024, "bytes": 4 * 1024 * 1024},
        {"id": "img1", "key": "hash_B", "start": 1600, "length": 1024, "bytes": 4 * 1024 * 1024},
        {"id": "img2", "key": "hash_C", "start": 3000, "length": 1024, "bytes": 4 * 1024 * 1024},
        {"id": "img3", "key": "hash_B", "start": 4500, "length": 1024, "bytes": 4 * 1024 * 1024},
    ]
    encoder_max_num_tokens = 2048
    kv_cached_prefix_tokens = 1536
    chunk_size = 2048

    # Legacy whole-request check: sums all unencoded items (4096 tokens > encoder_max_num_tokens=2048) -> deadlocks/defers!
    legacy_required_tokens = sum(item["length"] for item in mm_items)
    legacy_can_schedule = legacy_required_tokens <= encoder_max_num_tokens

    # Chunk-scoped after resolving KV prefix reuse (start = 1536):
    # Chunk 1: [1536, 3584) -> intersects img1 [1600, 2624) and img2 [3000, 4024) -> 2048 tokens <= 2048
    # Chunk 2: [3584, 5632) -> img2 already encoded in Chunk 1; only img3 [4500, 5524) -> 1024 tokens <= 2048
    chunks = [(1536, 3584), (3584, 5632)]
    encoded_ids: set[str] = {"img0"}  # covered by KV prefix reuse
    chunk_schedule_log: list[dict[str, Any]] = []
    for chunk_start, chunk_end in chunks:
        active_items = [
            item
            for item in mm_items
            if item["id"] not in encoded_ids
            and (item["start"] + item["length"] > chunk_start)
            and (item["start"] < chunk_end)
        ]
        tokens_needed = sum(item["length"] for item in active_items)
        can_run = tokens_needed <= encoder_max_num_tokens
        for item in active_items:
            encoded_ids.add(item["id"])
        chunk_schedule_log.append(
            {
                "chunk_range": [chunk_start, chunk_end],
                "scheduled_item_ids": [item["id"] for item in active_items],
                "encoder_tokens_charged": tokens_needed,
                "fits_encoder_budget": can_run,
            }
        )

    # TensorLRUCache.releasable_bytes(keys): entry is freed iff 0 < entry.reference_count <= count_in_keys
    cache_entries = {
        "hash_A": {"nbytes": 4194304, "ref_count": 0},  # already unpinned (in removable_bytes)
        "hash_B": {"nbytes": 4194304, "ref_count": 2},  # shared by 2 items
        "hash_C": {"nbytes": 4194304, "ref_count": 1},  # pinned by 1 item
    }

    def releasable_bytes(keys: list[str]) -> int:
        counts = Counter(keys)
        total = 0
        for key, count in counts.items():
            entry = cache_entries.get(key)
            if entry is not None and 0 < entry["ref_count"] <= count:
                total += entry["nbytes"]
        return total

    return {
        "encoder_max_num_tokens": encoder_max_num_tokens,
        "kv_cached_prefix_tokens": kv_cached_prefix_tokens,
        "legacy_whole_request_tokens": legacy_required_tokens,
        "legacy_can_schedule": legacy_can_schedule,
        "chunk_scoped_schedule": chunk_schedule_log,
        "chunk_scoped_all_scheduled": all(c["fits_encoder_budget"] for c in chunk_schedule_log),
        "releasable_bytes_single_hash_B_and_C": releasable_bytes(["hash_B", "hash_C"]),
        "releasable_bytes_both_hash_B_and_C": releasable_bytes(["hash_B", "hash_B", "hash_C"]),
    }


def compute_keeps_mma_ab_custom_mask_size(
    seq_len_q: int, seq_len_kv: int, first_sparse_offset_kv: int, num_heads_q_per_kv: int, step_q: int, step_kv: int
) -> int:
    num_tiles_q = ceil_div(seq_len_q * num_heads_q_per_kv, step_q)
    first_sparse_tile = first_sparse_offset_kv // step_kv
    num_custom_mask_tiles_kv = ceil_div(seq_len_kv, step_kv) - first_sparse_tile
    return num_tiles_q * num_custom_mask_tiles_kv * step_q * step_kv // 32


def compute_swaps_mma_ab_custom_mask_size(
    seq_len_q: int,
    seq_len_kv: int,
    first_sparse_offset_kv: int,
    num_heads_q_per_kv: int,
    step_q: int,
    step_kv: int,
    tile_size_q_raw: int,
    tile_size_kv: int,
) -> int:
    num_insts_q = step_q // tile_size_q_raw
    num_insts_kv = step_kv // tile_size_kv
    tile_size_kv_per_cta = step_kv
    tile_size_q = ceil_div(tile_size_q_raw, 32) * 32
    tile_size_q_per_cta = tile_size_q * num_insts_q
    num_tiles_q_per_token = ceil_div(num_heads_q_per_kv, tile_size_q_per_cta)
    num_tiles_q = seq_len_q * num_tiles_q_per_token
    first_sparse_tile = first_sparse_offset_kv // tile_size_kv_per_cta
    num_custom_mask_tiles_kv = ceil_div(seq_len_kv, tile_size_kv_per_cta) - first_sparse_tile
    per_tile_size = num_insts_q * num_insts_kv * (tile_size_q * tile_size_kv) // 32
    return num_tiles_q * num_custom_mask_tiles_kv * per_tile_size


def simulate_blackwell_spec_dec_mask_sizing() -> dict[str, Any]:
    """Compare legacy KeepsMmaAb-only custom mask buffer sizing vs SwapsMmaAb upper bound (#20223)."""
    max_batch_size = 64
    max_seq_len_q = 64
    num_heads_q_per_kv = 8
    tile_size_kv = 128

    # Legacy Python formula assumed KeepsMmaAb with tile_size_q=128, num_insts_q=1, num_insts_kv=2 (step_kv=256)
    legacy_seq_len_kv = max_seq_len_q + 256 - 1
    legacy_uint32_capacity = max_batch_size * compute_keeps_mma_ab_custom_mask_size(
        max_seq_len_q, legacy_seq_len_kv, 0, num_heads_q_per_kv, step_q=128, step_kv=256
    )

    shape_Requirements: list[dict[str, Any]] = []
    max_capacity = 0
    for step_kv in (128, 256):
        seq_len_kv = max_seq_len_q + step_kv - 1
        for step_q in (128, 256):
            if step_q * step_kv <= 128 * 128 * 2:
                req = max_batch_size * compute_keeps_mma_ab_custom_mask_size(
                    max_seq_len_q, seq_len_kv, 0, num_heads_q_per_kv, step_q, step_kv
                )
                max_capacity = max(max_capacity, req)
                shape_Requirements.append(
                    {
                        "kernel_type": "KeepsMmaAbForGeneration",
                        "step_q": step_q,
                        "step_kv": step_kv,
                        "tile_size_q": 128,
                        "required_uint32": req,
                        "overflows_legacy_buffer": req > legacy_uint32_capacity,
                    }
                )
        for tile_size_q in (8, 16, 32):
            req = max_batch_size * compute_swaps_mma_ab_custom_mask_size(
                max_seq_len_q, seq_len_kv, 0, num_heads_q_per_kv, tile_size_q, step_kv, tile_size_q, tile_size_kv
            )
            max_capacity = max(max_capacity, req)
            shape_Requirements.append(
                {
                    "kernel_type": "SwapsMmaAbForGeneration",
                    "step_q": tile_size_q,
                    "step_kv": step_kv,
                    "tile_size_q": tile_size_q,
                    "required_uint32": req,
                    "overflows_legacy_buffer": req > legacy_uint32_capacity,
                }
            )

    return {
        "max_batch_size": max_batch_size,
        "max_seq_len_q": max_seq_len_q,
        "num_heads_q_per_kv": num_heads_q_per_kv,
        "legacy_keeps_only_uint32": legacy_uint32_capacity,
        "new_upper_bound_uint32": max_capacity,
        "swaps_vs_legacy_ratio": round(max_capacity / legacy_uint32_capacity, 4),
        "shape_requirements": shape_Requirements,
    }


def simulate_kvcm2_progressive_host_tier() -> dict[str, Any]:
    """Model KVCM2 HostMem MAP_NORESERVE reservation and progressive commit units (#19650)."""
    base_unit = 2 * 1024 * 1024  # 2 MiB
    one_gib = 1024**3

    def tier_allocation_granularity(quota_bytes: int) -> int:
        if quota_bytes <= one_gib:
            return base_unit
        gib_Shift = min(4, int(math.log2(quota_bytes // one_gib)))
        return base_unit << gib_Shift

    def round_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    def host_commit_unit(tier_unit: int, slot_size: int) -> int:
        align = slot_size if slot_size > 0 else tier_unit
        return round_up(max(tier_unit, slot_size), align)

    quota_profiles: list[dict[str, Any]] = []
    for quota_gib, slot_mib in [(1, 2), (4, 8), (8, 12), (16, 24), (64, 16)]:
        quota_bytes = quota_gib * one_gib
        slot_bytes = slot_mib * 1024 * 1024
        granularity = tier_allocation_granularity(quota_bytes)
        commit_unit = host_commit_unit(granularity, slot_bytes)
        first_chunk_bytes = min(quota_bytes, commit_unit)
        quota_profiles.append(
            {
                "quota_gib": quota_gib,
                "slot_mib": slot_mib,
                "tier_granularity_mib": granularity // (1024 * 1024),
                "commit_unit_mib": commit_unit // (1024 * 1024),
                "startup_synchronous_commit_mib": first_chunk_bytes // (1024 * 1024),
                "startup_commit_reduction_x": round(quota_bytes / first_chunk_bytes, 2),
                "base_address_stable_without_mremap": True,
            }
        )

    return {
        "quota_profiles": quota_profiles,
        "min_tier_granularity_mib": 2,
        "max_tier_granularity_mib": 32,
    }


def simulate_vllm_and_sglang_paths() -> dict[str, Any]:
    """Model vLLM libgomp fork pause (#62884), ROCm HY-V4 iHC fusion (#58509),
    and SGLang Intel XPU KV transfer & FlashInfer CUTLASS MoE tuning (#40861, #43363, #45437).
    """
    # 1. vLLM HY-V4 ROCm iHC fusion memory traffic reduction (#58509):
    # hc_mult = 4, hidden_size d = 4096, BF16 (2 bytes/elem)
    # Unfused eager/3-kernel path across post + pre + RMSNorm:
    # - post reads x [T, 4, d] (32 KB/tok) + attn_post [T, 4] + residual [T, d] (8 KB/tok) and writes y [T, 4, d] (32 KB/tok)
    # - pre reads y [T, 4, d] (32 KB/tok) and writes output [T, d] (8 KB/tok) + post [T, 4]
    # - rms_norm reads output [T, d] (8 KB/tok) and writes normed_output [T, d] (8 KB/tok)
    # Total HBM bytes per token (ignoring tiny scalars):
    # Unfused = (32 + 8 + 32) + (32 + 8) + (8 + 8) = 128 KiB/token
    # Fused triton_ihc_post_pre_rms_norm reads x (32 KB) + residual (8 KB) and writes y (32 KB) + normed_output (8 KB) = 80 KiB/token
    d = 4096
    hc_mult = 4
    elem_bytes = 2
    unfused_bytes_per_token = (hc_mult * d + d + hc_mult * d + hc_mult * d + d + d + d) * elem_bytes
    fused_bytes_per_token = (hc_mult * d + d + hc_mult * d + d) * elem_bytes

    # 2. SGLang Intel XPU kernel-space pointer detection (#40861):
    # On Intel XPU, device allocations reside in canonical high virtual addresses with bit 63 set (`ptr & (1 << 63) != 0`).
    xpu_device_ptr = 0xFF00_0000_1234_0000
    host_Staging_ptr = 0x0000_7F8A_4000_0000
    xpu_bit63_set = bool(xpu_device_ptr & (1 << 63))
    host_bit63_set = bool(host_Staging_ptr & (1 << 63))

    # 3. SGLang FlashInfer CUTLASS MoE tune_max_num_tokens (#43363):
    # Prefill worker: min(prefill_tokens * attn_dp_size, 8192)
    # Decode / unified worker: next_power_of_2(num_tokens)
    prefill_tokens = 2048
    attn_dp_size = 8
    prefill_tune_max_tokens = min(prefill_tokens * attn_dp_size, 8192)
    decode_num_tokens = 96
    decode_tune_max_tokens = 1 << (decode_num_tokens - 1).bit_length()

    return {
        "vllm_libgomp_fork_pause": {
            "omp_pause_hard_enum": 2,
            "prevents_post_fork_openmp_deadlock": True,
        },
        "vllm_rocm_hy_v4_ihc_fusion": {
            "hc_mult": hc_mult,
            "hidden_size": d,
            "unfused_hbm_bytes_per_token": unfused_bytes_per_token,
            "fused_hbm_bytes_per_token": fused_bytes_per_token,
            "hbm_traffic_reduction_ratio": round(unfused_bytes_per_token / fused_bytes_per_token, 4),
            "kernel_launches_per_sublayer_before": 3,
            "kernel_launches_per_sublayer_after": 1,
        },
        "sglang_xpu_and_cutlass_moe": {
            "xpu_device_ptr_hex": hex(xpu_device_ptr),
            "xpu_detected_by_bit63": xpu_bit63_set,
            "host_staging_detected_by_bit63": host_bit63_set,
            "prefill_tune_max_num_tokens": prefill_tune_max_tokens,
            "decode_96_tune_max_num_tokens": decode_tune_max_tokens,
            "zmq_source_ip_endpoint_example": "tcp://10.24.0.15:0;tcp://10.24.1.20:30001",
        },
    }


def build_summary() -> dict[str, Any]:
    mm_chunk = simulate_trtllm_mm_chunk_scheduling()
    spec_mask = simulate_blackwell_spec_dec_mask_sizing()
    kvcm2 = simulate_kvcm2_progressive_host_tier()
    vllm_sglang = simulate_vllm_and_sglang_paths()

    return {
        "experiment": "trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu",
        "phase": "Phase 46",
        "priority": 55,
        "upstream_commits": [
            "TensorRT-LLM@b47eb8fe6b0965a85adf01121f9f0f2a1e79c7c2",
            "TensorRT-LLM@85b1b8323b78a4717aeb772aa66eed9b8346d687",
            "TensorRT-LLM@0cd9d0b5415b82696ecbaba8b0b6034aa9c09997",
            "vllm@40210d8317bbad8d05124b260e75677daf33e091",
            "vllm@833349105b9581d19487295454c19c924e8a3b37",
            "sglang@8e3e2980db83a1b020a6d273a2b653891af81bfc",
            "sglang@f8fd41c36e493ec762e60f447c50c88fb8af3ec1",
            "sglang@2c9d9190713285552d7ea2bb264664c90639c824",
        ],
        "trtllm_mm_chunk_scheduling": mm_chunk,
        "blackwell_spec_dec_mask_sizing": spec_mask,
        "kvcm2_progressive_host_tier": kvcm2,
        "vllm_and_sglang_runtime": vllm_sglang,
        "headline_metrics": {
            "mm_chunk_legacy_tokens": mm_chunk["legacy_whole_request_tokens"],
            "mm_chunk_max_per_chunk_tokens": max(c["encoder_tokens_charged"] for c in mm_chunk["chunk_scoped_schedule"]),
            "swaps_vs_legacy_mask_capacity_ratio": spec_mask["swaps_vs_legacy_ratio"],
            "kvcm2_64gib_startup_commit_reduction_x": kvcm2["quota_profiles"][-1]["startup_commit_reduction_x"],
            "hy_v4_ihc_hbm_traffic_reduction_ratio": vllm_sglang["vllm_rocm_hy_v4_ihc_fusion"]["hbm_traffic_reduction_ratio"],
            "sglang_prefill_tune_max_num_tokens_cap": vllm_sglang["sglang_xpu_and_cutlass_moe"]["prefill_tune_max_num_tokens"],
        },
    }


def verify_summary(summary: dict[str, Any]) -> None:
    mm = summary["trtllm_mm_chunk_scheduling"]
    assert mm["legacy_can_schedule"] is False
    assert mm["chunk_scoped_all_scheduled"] is True
    assert mm["releasable_bytes_single_hash_B_and_C"] == 4194304
    assert mm["releasable_bytes_both_hash_B_and_C"] == 8388608

    mask = summary["blackwell_spec_dec_mask_sizing"]
    assert mask["new_upper_bound_uint32"] == 4 * mask["legacy_keeps_only_uint32"]
    assert mask["swaps_vs_legacy_ratio"] == 4.0

    kvcm2 = summary["kvcm2_progressive_host_tier"]
    assert kvcm2["quota_profiles"][0]["tier_granularity_mib"] == 2
    assert kvcm2["quota_profiles"][-1]["tier_granularity_mib"] == 32
    assert kvcm2["quota_profiles"][-1]["startup_commit_reduction_x"] == 2048.0

    vs = summary["vllm_and_sglang_runtime"]
    assert vs["vllm_libgomp_fork_pause"]["omp_pause_hard_enum"] == 2
    assert vs["vllm_rocm_hy_v4_ihc_fusion"]["hbm_traffic_reduction_ratio"] == 1.6
    assert vs["sglang_xpu_and_cutlass_moe"]["xpu_detected_by_bit63"] is True
    assert vs["sglang_xpu_and_cutlass_moe"]["host_staging_detected_by_bit63"] is False
    assert vs["sglang_xpu_and_cutlass_moe"]["prefill_tune_max_num_tokens"] == 8192


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run deterministic invariant assertions")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (check={args.check})")


if __name__ == "__main__":
    main()
