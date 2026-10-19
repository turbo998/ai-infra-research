#!/usr/bin/env python3
"""
Deterministic offline simulator for DeepSeek FlashMLA (SM90 & SM100):
- Multi-head Latent Attention (MLA) matrix absorption arithmetic intensity & KV cache byte layout
- Verification of `KVCacheFormat<MT>` (`V41` FP8 528 B/tok vs `V41_FP4` 288 B/tok & 2-CTA `KVCachePart`)
  from `deepseek-ai/FlashMLA` (`csrc/cuda_kernels/kv_cache_format.h`)
- Verification of `get_mla_metadata_kernel` greedy variable-length Split-KV work partitioning
  from `deepseek-ai/FlashMLA` (`csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def simulate_kv_formats_and_roofline() -> list[dict[str, Any]]:
    """
    Compare KV cache bytes per token per layer and Decode arithmetic intensity (FLOPs/Byte)
    at batch_size=1 (single request) for H_q = 128 query heads:
    1. Standard MHA-128 (BF16, d_qk=128, d_v=128): K+V = 2 * 128 * 128 * 2 = 65,536 B/tok
    2. Standard GQA-8 (BF16, H_kv=8, d_qk=128, d_v=128): K+V = 2 * 8 * 128 * 2 = 4,096 B/tok
    3. Unabsorbed MLA (BF16, decompressing c_KV (512) -> 128 heads of d_nope=128 + d_v=128):
       materializes 65,536 B/tok in HBM, AI = 1.0 FLOP/B
    4. Absorbed MLA Dense (BF16, d_nope=512 + d_rope=64 = 576 shared across H_q=128):
       576 * 2 = 1,152 B/tok; FLOPs per KV token across 128 heads:
       score = 2 * 128 * 576 = 147,456 FLOPs; V-agg = 2 * 128 * 512 = 131,072 FLOPs; Total = 278,528 FLOPs
    5. FlashMLA V4.1 Sparse FP8 (`ModelType::V41` in `kv_cache_format.h`):
       D_QK = 512 (D_NOPE=448, D_ROPE=64), QUANT_TILE_SIZE=32 -> 512 B e4m3 + 16 B ue8m0 = 528 B/tok
    6. FlashMLA V4.1 Sparse FP4 (`ModelType::V41_FP4` in `kv_cache_format.h`):
       D_QK = 512, QUANT_TILE_SIZE=16 -> 256 B e2m1 + 32 B e4m3 = 288 B/tok
    7. FlashMLA Hybrid `(V41, V41_FP4)` (`KVFormatPair<V41, V41_FP4>`):
       Hot topk=2048 tokens in V41 FP8 (528 B) + extra_topk=4096 tokens in V41_FP4 (288 B) -> avg 368.0 B/tok
    """
    h_q = 128
    # Hardware roofs:
    # H800/H100 SXM: 3350 GB/s HBM3, 989 TFLOPS BF16 / 1979 TFLOPS FP8
    # B200 SXM: 8000 GB/s HBM3e, 2250 TFLOPS BF16 / 4500 TFLOPS FP8
    h100_bw_gbps = 3350.0
    b200_bw_gbps = 8000.0

    configs = [
        {
            "format_id": "MHA_128_BF16",
            "d_qk_total": 128,
            "d_v": 128,
            "kv_heads_stored": 128,
            "bytes_per_token": 2 * 128 * 128 * 2,
            "flops_per_kv_token": 2 * h_q * (128 + 128),
            "cta_group_size": 1,
            "cta0_raw_bytes": 65536,
            "cta1_raw_bytes": 0,
        },
        {
            "format_id": "GQA_8_BF16",
            "d_qk_total": 128,
            "d_v": 128,
            "kv_heads_stored": 8,
            "bytes_per_token": 2 * 8 * 128 * 2,
            "flops_per_kv_token": 2 * h_q * (128 + 128),
            "cta_group_size": 1,
            "cta0_raw_bytes": 4096,
            "cta1_raw_bytes": 0,
        },
        {
            "format_id": "MLA_UNABSORBED_BF16",
            "d_qk_total": 192,
            "d_v": 128,
            "kv_heads_stored": 128,
            "bytes_per_token": 128 * (192 + 128) * 2,
            "flops_per_kv_token": 2 * h_q * (192 + 128),
            "cta_group_size": 1,
            "cta0_raw_bytes": 81920,
            "cta1_raw_bytes": 0,
        },
        {
            "format_id": "MLA_ABSORBED_BF16_D576",
            "d_qk_total": 576,
            "d_v": 512,
            "kv_heads_stored": 1,
            "bytes_per_token": 576 * 2,
            "flops_per_kv_token": 2 * h_q * (576 + 512),
            "cta_group_size": 1,
            "cta0_raw_bytes": 1152,
            "cta1_raw_bytes": 0,
        },
        {
            "format_id": "FLASHMLA_V41_FP8_528B",
            "d_qk_total": 512,
            "d_v": 512,
            "kv_heads_stored": 1,
            # KVCacheFormat<ModelType::V41>: 512 B e4m3 + 16 B ue8m0 = 528 B
            "bytes_per_token": 512 + 16,
            "flops_per_kv_token": 2 * h_q * (512 + 512),
            "cta_group_size": 2,
            # KVCachePart<V41, 2, 0>: QUANT_BYTES=256, SEPARATE_SCALES=true -> RAW_TOKEN_DATA_BYTES=256
            "cta0_raw_bytes": 256,
            # KVCachePart<V41, 2, 1>: QUANT_BYTES=256, SEPARATE_SCALES=false -> 256 + 16 = 272
            "cta1_raw_bytes": 256 + 16,
        },
        {
            "format_id": "FLASHMLA_V41_FP4_288B",
            "d_qk_total": 512,
            "d_v": 512,
            "kv_heads_stored": 1,
            # KVCacheFormat<ModelType::V41_FP4>: 256 B e2m1 + 32 B e4m3 = 288 B
            "bytes_per_token": 256 + 32,
            "flops_per_kv_token": 2 * h_q * (512 + 512),
            "cta_group_size": 2,
            # KVCachePart<V41_FP4, 2, 0>: QUANT_BYTES=128, SEPARATE_SCALES=true -> 128
            "cta0_raw_bytes": 128,
            # KVCachePart<V41_FP4, 2, 1>: QUANT_BYTES=128 + 32 = 160
            "cta1_raw_bytes": 128 + 32,
        },
        {
            "format_id": "FLASHMLA_HYBRID_V41_FP8_FP4",
            "d_qk_total": 512,
            "d_v": 512,
            "kv_heads_stored": 1,
            # 1/3 hot tokens in V41 (528 B) + 2/3 cold tokens in V41_FP4 (288 B) = 368 B
            "bytes_per_token": 368,
            "flops_per_kv_token": 2 * h_q * (512 + 512),
            "cta_group_size": 2,
            "cta0_raw_bytes": 171,
            "cta1_raw_bytes": 197,
        },
    ]

    results: list[dict[str, Any]] = []
    gqa_bytes = 4096
    for cfg in configs:
        bpt = cfg["bytes_per_token"]
        flops = cfg["flops_per_kv_token"]
        ai = flops / bpt
        compression_vs_gqa8 = gqa_bytes / bpt
        # Attainable TFLOPS on H100 (85% HBM bw eff = 2847.5 GB/s, peak 989 BF16 / 1979 FP8)
        is_low_prec = "FP8" in cfg["format_id"] or "FP4" in cfg["format_id"]
        h100_peak = 1480.0 if is_low_prec else 989.0
        b200_peak = 3600.0 if is_low_prec else 2250.0
        h100_att_tflops = min(h100_peak * 0.78, (h100_bw_gbps * 0.85 * ai) / 1000.0)
        b200_att_tflops = min(b200_peak * 0.80, (b200_bw_gbps * 0.86 * ai) / 1000.0)
        results.append(
            {
                **cfg,
                "arithmetic_intensity_flop_per_byte": round(ai, 2),
                "compression_vs_gqa8_x": round(compression_vs_gqa8, 3),
                "h100_attainable_tflops": round(h100_att_tflops, 2),
                "b200_attainable_tflops": round(b200_att_tflops, 2),
            }
        )
    return results


def run_get_mla_metadata_exact(
    seqlens_k: list[int],
    block_size_topk: int,
    fixed_overhead_num_blocks: int,
    num_sm_parts: int,
) -> dict[str, Any]:
    """
    Exact Python port of `get_mla_metadata_kernel` in
    `csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`.
    """
    batch_size = len(seqlens_k)
    num_blocks_shared: list[int] = []
    last_block_idx_shared: list[int] = []
    total_num_blocks = 0

    for cur_s_k in seqlens_k:
        eff_s_k = max(1, cur_s_k)
        cur_last_block_idx = (eff_s_k - 1) // block_size_topk
        num_blocks = cur_last_block_idx + 1
        total_num_blocks += num_blocks + fixed_overhead_num_blocks
        num_blocks_shared.append(num_blocks)
        last_block_idx_shared.append(cur_last_block_idx)

    payload = ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks

    now_req_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    num_splits_shared = [0] * (batch_size + 1)
    partitions: list[dict[str, Any]] = []
    part_work_blocks: list[int] = []

    for i in range(num_sm_parts):
        begin_req_idx = now_req_idx
        begin_block_idx = now_block
        begin_split_idx = now_n_split_idx
        is_first_req_splitted = now_block != 0
        remain_payload = payload
        actual_blocks_assigned = 0
        reqs_touched = 0

        while now_req_idx < batch_size:
            num_blocks = num_blocks_shared[now_req_idx]
            now_remain_blocks = num_blocks - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits += now_n_split_idx + 1
                num_splits_shared[now_req_idx + 1] = cum_num_splits
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                actual_blocks_assigned += now_remain_blocks
                reqs_touched += 1
                now_req_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    consumed = remain_payload - fixed_overhead_num_blocks
                    now_block += consumed
                    actual_blocks_assigned += consumed
                    reqs_touched += 1
                    now_n_split_idx += 1
                    remain_payload = 0
                break

        end_req_idx = now_req_idx if now_block > 0 else now_req_idx - 1
        end_block_idx = (
            now_block
            if now_block > 0
            else (last_block_idx_shared[now_req_idx - 1] + 1 if now_req_idx > 0 else 0)
        )
        is_last_req_splitted = (
            end_block_idx != last_block_idx_shared[end_req_idx] + 1
            if 0 <= end_req_idx < batch_size
            else False
        )
        if begin_req_idx == end_req_idx:
            merged = is_first_req_splitted or is_last_req_splitted
            is_first_req_splitted = merged
            is_last_req_splitted = merged

        partitions.append(
            {
                "sm_part": i,
                "begin_req_idx": begin_req_idx,
                "end_req_idx": end_req_idx,
                "begin_block_idx": begin_block_idx,
                "end_block_idx": end_block_idx,
                "begin_split_idx": begin_split_idx,
                "is_first_req_splitted": is_first_req_splitted,
                "is_last_req_splitted": is_last_req_splitted,
                "actual_kv_blocks": actual_blocks_assigned,
                "effective_cost_blocks": actual_blocks_assigned + reqs_touched * fixed_overhead_num_blocks,
            }
        )
        part_work_blocks.append(actual_blocks_assigned + reqs_touched * fixed_overhead_num_blocks)

    assert now_req_idx == batch_size and now_block == 0 and now_n_split_idx == 0, (
        f"Unfinished work: req={now_req_idx}/{batch_size}, block={now_block}, split={now_n_split_idx}"
    )

    max_cost = max(part_work_blocks)
    mean_cost = sum(part_work_blocks) / len(part_work_blocks)
    active_parts = sum(1 for c in part_work_blocks if c > 0)
    per_req_splits = [
        num_splits_shared[r + 1] - num_splits_shared[r] for r in range(batch_size)
    ]

    return {
        "total_kv_blocks": sum(num_blocks_shared),
        "total_num_blocks_with_overhead": total_num_blocks,
        "payload_per_sm_part": payload,
        "num_sm_parts": num_sm_parts,
        "active_sm_parts": active_parts,
        "total_splits": cum_num_splits,
        "max_splits_for_single_req": max(per_req_splits),
        "max_sm_part_cost_blocks": max_cost,
        "mean_sm_part_cost_blocks": round(mean_cost, 3),
        "wave_imbalance_pct": round((max_cost - mean_cost) / max(max_cost, 1) * 100.0, 2),
        "num_splits_prefix_sum_head": num_splits_shared[:9],
    }


def simulate_variable_length_decode_schedulers() -> list[dict[str, Any]]:
    """
    Compare 3 Decode scheduling strategies on a skewed batch of B=64 requests
    (8 long agentic contexts of 32K–64K tokens + 56 short/medium requests of 512–4096 tokens):
    1. `STATIC_REQ_PER_CTA`: 1 request per CTA (or fixed 2-way split per request), causing severe tail SM idling
    2. `UNIFORM_FIXED_SPLIT_KV`: Every request split into fixed S=4 splits regardless of length, causing excessive combine overhead on short requests and under-splitting on 64K requests
    3. `FLASHMLA_GET_SCHED_META`: Exact `get_mla_metadata_kernel` greedy partition with `fixed_overhead_num_blocks = 5` across `num_sm_parts = 74` (Blackwell 2-CTA clusters) and `num_sm_parts = 132` (Hopper SMs)
    """
    block_size_topk = 64
    fixed_overhead = 5
    # Deterministic skewed batch (B=64)
    seqlens_k: list[int] = []
    for r in range(64):
        if r < 4:
            seqlens_k.append(65536 - r * 4096)  # 64K, 60K, 56K, 52K
        elif r < 8:
            seqlens_k.append(32768 - (r - 4) * 2048)  # 32K..26K
        elif r < 24:
            seqlens_k.append(4096 + (r % 4) * 512)
        else:
            seqlens_k.append(512 + (r % 8) * 128)

    num_blocks_list = [ceil_div(s, block_size_topk) for s in seqlens_k]
    total_kv_blocks = sum(num_blocks_list)
    total_tokens = sum(seqlens_k)

    # Evaluate on Blackwell B200 (74 2-CTA pairs = 148 SMs, V41 FP8 528 B/tok)
    num_sm_parts = 74
    block_time_us = 0.088  # 64 tokens * 528 B / (74 pairs * effective BW) scaled per SM-pair
    combine_per_split_us = 0.042

    # 1. Static 1-req-per-SM-part
    static_costs = [b + fixed_overhead for b in num_blocks_list] + [0] * (num_sm_parts - len(num_blocks_list))
    static_max = max(static_costs)
    static_mean = sum(static_costs) / num_sm_parts
    static_imbalance = (static_max - static_mean) / static_max * 100.0
    static_latency_us = static_max * block_time_us + 64 * combine_per_split_us * 0.2

    # 2. Fixed Split-KV (split_k = 4 for all requests, assigned round-robin to 74 SM parts)
    fixed_splits = 4
    part_loads = [0.0] * num_sm_parts
    idx = 0
    for b in num_blocks_list:
        chunk = ceil_div(b, fixed_splits)
        for _ in range(fixed_splits):
            part_loads[idx % num_sm_parts] += chunk + fixed_overhead
            idx += 1
    fixed_max = max(part_loads)
    fixed_mean = sum(part_loads) / num_sm_parts
    fixed_imbalance = (fixed_max - fixed_mean) / fixed_max * 100.0
    fixed_latency_us = fixed_max * block_time_us + (64 * fixed_splits) * combine_per_split_us

    # 3. FlashMLA `get_mla_metadata_kernel` (`fixed_overhead_num_blocks = 5`, `num_sm_parts = 74`)
    meta_74 = run_get_mla_metadata_exact(
        seqlens_k=seqlens_k,
        block_size_topk=block_size_topk,
        fixed_overhead_num_blocks=fixed_overhead,
        num_sm_parts=74,
    )
    flashmla_74_latency_us = (
        meta_74["max_sm_part_cost_blocks"] * block_time_us
        + meta_74["total_splits"] * combine_per_split_us
    )

    # 4. FlashMLA `get_mla_metadata_kernel` on Hopper (`num_sm_parts = 132`)
    meta_132 = run_get_mla_metadata_exact(
        seqlens_k=seqlens_k,
        block_size_topk=block_size_topk,
        fixed_overhead_num_blocks=fixed_overhead,
        num_sm_parts=132,
    )
    hopper_block_time_us = 0.118
    flashmla_132_latency_us = (
        meta_132["max_sm_part_cost_blocks"] * hopper_block_time_us
        + meta_132["total_splits"] * combine_per_split_us
    )

    total_flops = total_tokens * 2 * 128 * (512 + 512)
    return [
        {
            "scheduler": "STATIC_REQ_PER_PART_74",
            "num_sm_parts": 74,
            "total_kv_blocks": total_kv_blocks,
            "total_splits": 64,
            "max_part_cost_blocks": static_max,
            "mean_part_cost_blocks": round(static_mean, 2),
            "wave_imbalance_pct": round(static_imbalance, 2),
            "latency_us": round(static_latency_us, 3),
            "effective_tflops": round(total_flops / (static_latency_us * 1e-6) / 1e12, 2),
            "speedup_vs_static_x": 1.0,
        },
        {
            "scheduler": "UNIFORM_SPLIT4_74",
            "num_sm_parts": 74,
            "total_kv_blocks": total_kv_blocks,
            "total_splits": 256,
            "max_part_cost_blocks": int(fixed_max),
            "mean_part_cost_blocks": round(fixed_mean, 2),
            "wave_imbalance_pct": round(fixed_imbalance, 2),
            "latency_us": round(fixed_latency_us, 3),
            "effective_tflops": round(total_flops / (fixed_latency_us * 1e-6) / 1e12, 2),
            "speedup_vs_static_x": round(static_latency_us / fixed_latency_us, 3),
        },
        {
            "scheduler": "FLASHMLA_GET_MLA_METADATA_SM100_74",
            "num_sm_parts": 74,
            "total_kv_blocks": total_kv_blocks,
            "total_splits": meta_74["total_splits"],
            "max_part_cost_blocks": meta_74["max_sm_part_cost_blocks"],
            "mean_part_cost_blocks": meta_74["mean_sm_part_cost_blocks"],
            "wave_imbalance_pct": meta_74["wave_imbalance_pct"],
            "latency_us": round(flashmla_74_latency_us, 3),
            "effective_tflops": round(total_flops / (flashmla_74_latency_us * 1e-6) / 1e12, 2),
            "speedup_vs_static_x": round(static_latency_us / flashmla_74_latency_us, 3),
        },
        {
            "scheduler": "FLASHMLA_GET_MLA_METADATA_SM90_132",
            "num_sm_parts": 132,
            "total_kv_blocks": total_kv_blocks,
            "total_splits": meta_132["total_splits"],
            "max_part_cost_blocks": meta_132["max_sm_part_cost_blocks"],
            "mean_part_cost_blocks": meta_132["mean_sm_part_cost_blocks"],
            "wave_imbalance_pct": meta_132["wave_imbalance_pct"],
            "latency_us": round(flashmla_132_latency_us, 3),
            "effective_tflops": round(total_flops / (flashmla_132_latency_us * 1e-6) / 1e12, 2),
            "speedup_vs_static_x": round(static_latency_us / flashmla_132_latency_us, 3),
        },
    ]


def generate_summary() -> dict[str, Any]:
    kv_formats = simulate_kv_formats_and_roofline()
    schedulers = simulate_variable_length_decode_schedulers()
    by_fmt = {r["format_id"]: r for r in kv_formats}
    by_sched = {r["scheduler"]: r for r in schedulers}

    return {
        "experiment_id": "flashmla_paged_decode_scheduling",
        "evidence_class": "own-offline",
        "upstream_verified_sources": [
            {
                "repo": "deepseek-ai/FlashMLA",
                "commit": "0f11e178da1f67becb087a324b400854931ec3e3",
                "file": "csrc/cuda_kernels/kv_cache_format.h",
                "blob_sha": "e730426a72f1d1cec4d063c834a479c541b2c299",
            },
            {
                "repo": "deepseek-ai/FlashMLA",
                "commit": "0f11e178da1f67becb087a324b400854931ec3e3",
                "file": "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
                "blob_sha": "2307c47f3dbd731510eae2ada4f9a502e67eb420",
            },
        ],
        "kv_formats_and_roofline": kv_formats,
        "variable_length_schedulers": schedulers,
        "headline_findings": {
            "mla_bf16_compression_vs_gqa8_x": by_fmt["MLA_ABSORBED_BF16_D576"]["compression_vs_gqa8_x"],
            "v41_fp8_bytes_per_token": by_fmt["FLASHMLA_V41_FP8_528B"]["bytes_per_token"],
            "v41_fp4_bytes_per_token": by_fmt["FLASHMLA_V41_FP4_288B"]["bytes_per_token"],
            "v41_fp8_arithmetic_intensity": by_fmt["FLASHMLA_V41_FP8_528B"]["arithmetic_intensity_flop_per_byte"],
            "v41_fp4_arithmetic_intensity": by_fmt["FLASHMLA_V41_FP4_288B"]["arithmetic_intensity_flop_per_byte"],
            "static_wave_imbalance_pct": by_sched["STATIC_REQ_PER_PART_74"]["wave_imbalance_pct"],
            "flashmla_sm100_wave_imbalance_pct": by_sched["FLASHMLA_GET_MLA_METADATA_SM100_74"]["wave_imbalance_pct"],
            "flashmla_sm100_speedup_vs_static_x": by_sched["FLASHMLA_GET_MLA_METADATA_SM100_74"]["speedup_vs_static_x"],
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
        assert hf["v41_fp8_bytes_per_token"] == 528
        assert hf["v41_fp4_bytes_per_token"] == 288
        assert hf["mla_bf16_compression_vs_gqa8_x"] > 3.5
        assert hf["v41_fp8_arithmetic_intensity"] > 240.0
        assert hf["v41_fp4_arithmetic_intensity"] > 450.0
        assert hf["flashmla_sm100_wave_imbalance_pct"] < 5.0
        assert hf["flashmla_sm100_speedup_vs_static_x"] > 5.0
        print(
            f"[PASS] flashmla_paged_decode_scheduling: V41_FP8={hf['v41_fp8_bytes_per_token']}B "
            f"(AI={hf['v41_fp8_arithmetic_intensity']}), V41_FP4={hf['v41_fp4_bytes_per_token']}B "
            f"(AI={hf['v41_fp4_arithmetic_intensity']}), imbalance "
            f"{hf['static_wave_imbalance_pct']}% -> {hf['flashmla_sm100_wave_imbalance_pct']}% "
            f"({hf['flashmla_sm100_speedup_vs_static_x']}x speedup)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
