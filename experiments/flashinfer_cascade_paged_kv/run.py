#!/usr/bin/env python3
"""Deterministic simulator for FlashInfer Multi-Level Cascade Attention & Base-2 LSE State Merging.

Models the exact algorithms verified in `flashinfer-ai/flashinfer`
(`commit f64b4ba8762ded12a4f4b9a2092974c9a395e7ca`):
- `flashinfer/cascade.py` (`MultiLevelCascadeAttentionWrapper`, `merge_state_in_place`)
- `include/flashinfer/attention/cascade.cuh` (`MergeStateKernel`, `MergeStateInPlaceKernel`,
  `PersistentVariableLengthMergeStatesKernel` with base-2 LSE `log2(sum(exp2(x)))` and `enable_pdl`)

Evaluates:
1. Exact numerical equivalence of multi-level base-2 LSE state merging (`MergeStateInPlaceKernel`)
   against monolithic full-sequence attention, plus empty-prefix `-inf` guard validation.
2. HBM KV cache read traffic (GiB) and kernel latency (ms) across batch sizes `B in {16, 32, 64, 128, 256}`
   and tree depths `L in {1 (flat Paged KV), 2 (Shared System Prompt + Suffix), 3 (System + Few-Shot/RAG + Suffix)}`
   on H100 SXM (`3.35 TB/s` HBM3, `989 TFLOPS` BF16).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any


RESULTS_PATH = pathlib.Path(__file__).resolve().parent / "results.json"


def _log2(x: float) -> float:
    return math.log(x, 2.0)


def _exp2(x: float) -> float:
    if x <= -1000.0:
        return 0.0
    return math.pow(2.0, x)


def compute_segment_attention_base2(
    q: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    sm_scale: float,
) -> tuple[list[float], float]:
    """Compute attention output V_seg and base-2 log-sum-exp S_seg for one KV segment.

    In FlashInfer (`cascade.cuh`), LSE is stored in base-2 (`log2` of sum of `2^(score * log2(e))`).
    If the segment is empty (`len(keys) == 0`), returns zero vector and `-inf` LSE.
    """
    head_dim = len(q)
    if not keys:
        return [0.0] * head_dim, float("-inf")

    rcp_ln2 = math.log2(math.e)
    logits_base2: list[float] = []
    for k in keys:
        dot = sum(q[d] * k[d] for d in range(head_dim)) * sm_scale
        logits_base2.append(dot * rcp_ln2)

    m_seg = max(logits_base2)
    exp2_weights = [_exp2(score - m_seg) for score in logits_base2]
    d_seg = sum(exp2_weights)
    lse_base2 = _log2(d_seg) + m_seg

    out = [0.0] * head_dim
    for w, v in zip(exp2_weights, values):
        scale = w / d_seg
        for d in range(head_dim):
            out[d] += scale * v[d]
    return out, lse_base2


def merge_state_in_place_base2(
    v_a: list[float],
    s_a: float,
    v_b: list[float],
    s_b: float,
) -> tuple[list[float], float]:
    """Exact replica of FlashInfer `MergeStateInPlaceKernel` (`include/flashinfer/attention/cascade.cuh:78-122`).

    ```cuda
    float s_max = max(s_a, s_b);
    s_a = math::ptx_exp2(max(s_a - s_max, -math::inf));
    s_b = math::ptx_exp2(max(s_b - s_max, -math::inf));
    float d_sum = s_a + s_b;
    float scale = d_sum > 0.f ? s_a / d_sum : 0.f;
    float other_scale = d_sum > 0.f ? s_b / d_sum : 0.f;
    v = scale * v + other_scale * v_other;
    s_a = d_sum > 0.f ? math::ptx_log2(d_sum) + s_max : -math::inf;
    ```
    """
    s_max = max(s_a, s_b)
    if math.isinf(s_max) and s_max < 0:
        return [0.0] * len(v_a), float("-inf")

    w_a = _exp2(max(s_a - s_max, -1e9)) if not (math.isinf(s_a) and s_a < 0) else 0.0
    w_b = _exp2(max(s_b - s_max, -1e9)) if not (math.isinf(s_b) and s_b < 0) else 0.0
    d_sum = w_a + w_b
    scale_a = (w_a / d_sum) if d_sum > 0.0 else 0.0
    scale_b = (w_b / d_sum) if d_sum > 0.0 else 0.0
    v_merged = [scale_a * a + scale_b * b for a, b in zip(v_a, v_b)]
    s_merged = (_log2(d_sum) + s_max) if d_sum > 0.0 else float("-inf")
    return v_merged, s_merged


def verify_multi_level_cascade_numerics() -> dict[str, Any]:
    """Verify 3-level cascade merge (`Leaf -> Level 1 -> Level 0`) matches monolithic attention."""
    head_dim = 16
    sm_scale = 1.0 / math.sqrt(head_dim)

    def make_vec(seed: int, scale: float = 0.25) -> list[float]:
        return [math.sin(seed * 1.7 + d * 0.9) * scale + math.cos(seed * 0.5 - d * 0.4) * scale for d in range(head_dim)]

    q = make_vec(101, 0.4)
    # Level 0: shared system prompt (12 tokens)
    k_l0 = [make_vec(10 + i) for i in range(12)]
    v_l0 = [make_vec(210 + i) for i in range(12)]
    # Level 1: shared RAG/few-shot group prefix (8 tokens)
    k_l1 = [make_vec(40 + i) for i in range(8)]
    v_l1 = [make_vec(240 + i) for i in range(8)]
    # Level 2: unique per-request suffix (5 tokens)
    k_l2 = [make_vec(80 + i) for i in range(5)]
    v_l2 = [make_vec(280 + i) for i in range(5)]

    # Monolithic baseline over all 25 tokens
    v_mono, lse_mono = compute_segment_attention_base2(
        q, k_l0 + k_l1 + k_l2, v_l0 + v_l1 + v_l2, sm_scale
    )

    # FlashInfer MultiLevelCascadeAttentionWrapper order (`cascade.py:553-556`):
    # Run leaf level (-1) first, then iteratively merge upper levels in place.
    v_leaf, lse_leaf = compute_segment_attention_base2(q, k_l2, v_l2, sm_scale)
    v_lvl0, lse_lvl0 = compute_segment_attention_base2(q, k_l0, v_l0, sm_scale)
    v_lvl1, lse_lvl1 = compute_segment_attention_base2(q, k_l1, v_l1, sm_scale)

    v_acc, lse_acc = merge_state_in_place_base2(v_leaf, lse_leaf, v_lvl0, lse_lvl0)
    v_acc, lse_acc = merge_state_in_place_base2(v_acc, lse_acc, v_lvl1, lse_lvl1)

    max_abs_err = max(abs(a - b) for a, b in zip(v_mono, v_acc))
    lse_abs_err = abs(lse_mono - lse_acc)

    # Also verify empty segment (-inf LSE) guard (`d_sum > 0.f ? ... : 0.f`)
    v_empty, lse_empty = compute_segment_attention_base2(q, [], [], sm_scale)
    v_with_empty, lse_with_empty = merge_state_in_place_base2(v_acc, lse_acc, v_empty, lse_empty)
    empty_guard_err = max(abs(a - b) for a, b in zip(v_acc, v_with_empty))
    both_empty_v, both_empty_lse = merge_state_in_place_base2(v_empty, lse_empty, v_empty, lse_empty)

    return {
        "head_dim": head_dim,
        "segment_lengths": {"level0_system": 12, "level1_rag_group": 8, "level2_unique_suffix": 5},
        "monolithic_lse_base2": round(lse_mono, 8),
        "cascade_merged_lse_base2": round(lse_acc, 8),
        "max_output_abs_error": max_abs_err,
        "lse_abs_error": lse_abs_err,
        "empty_segment_guard_error": empty_guard_err,
        "both_empty_returns_neg_inf": math.isinf(both_empty_lse) and both_empty_lse < 0 and all(x == 0.0 for x in both_empty_v),
    }


def simulate_cascade_paged_kv_sweep() -> dict[str, Any]:
    """Compare Flat Paged-KV vs 2-Level Shared-Prefix Cascade vs 3-Level MultiLevelCascadeAttentionWrapper.

    Model configuration: Llama-3.1-70B single layer GQA (`H_q = 64`, `H_kv = 8`, `d = 128`),
    evaluated per layer and scaled across 80 layers on 4x H100 TP=4 (`H_q_local = 16`, `H_kv_local = 2`).
    Context structure per request during Decode/Chunked-Append (`q_len = 1` decode token per request):
    - `L_sys = 16,384` tokens (shared across all `B` requests in the batch)
    - `L_rag = 8,192` tokens (shared within `G = 4` tenant/RAG subgroups, so `B / 4` requests share each RAG prefix)
    - `L_uniq = 1,024` tokens (unique per-request conversation tail)
    Total effective sequence length per request: `L_total = 25,600` tokens.
    """
    h_q_local = 16
    h_kv_local = 2
    head_dim = 128
    num_layers = 80
    bytes_per_elem = 2  # BF16 KV cache
    l_sys = 16384
    l_rag = 8192
    l_uniq = 1024
    l_total = l_sys + l_rag + l_uniq
    num_rag_groups = 4

    # H100 SXM effective HBM bandwidth & launch/merge overhead parameters
    hbm_bw_bytes_per_sec = 2.85e12  # 2.85 TB/s effective achieved on paged attention
    tensor_core_flops_per_sec = 720e12  # 720 TFLOPS effective BF16 attention math
    standard_launch_overhead_us = 5.5  # per kernel launch without PDL
    pdl_launch_overhead_us = 1.6  # with cudaLaunchAttributeProgrammaticStreamSerialization

    kv_bytes_per_token_per_layer = 2 * h_kv_local * head_dim * bytes_per_elem  # 2 (K+V) * 2 * 128 * 2 = 1024 B

    batch_sizes = [16, 32, 64, 128, 256]
    rows: list[dict[str, Any]] = []

    for b in batch_sizes:
        # 1. Flat Paged-KV Decode: Even when physical KV pages of L_sys and L_rag are deduplicated in HBM,
        # independent CTA blocks per request stream all L_total KV pages from HBM because L2 capacity (50 MB)
        # cannot hold 80 layers * 25.6K tokens across requests, and each CTA only processes 1 query head group (GQA=8).
        flat_kv_bytes_layer = b * l_total * kv_bytes_per_token_per_layer
        flat_q_o_bytes_layer = b * h_q_local * head_dim * bytes_per_elem * 2
        flat_total_bytes_80l = (flat_kv_bytes_layer + flat_q_o_bytes_layer) * num_layers
        flat_flops_80l = b * l_total * (4 * h_q_local * head_dim) * num_layers
        flat_time_ms = (
            max(flat_total_bytes_80l / hbm_bw_bytes_per_sec, flat_flops_80l / tensor_core_flops_per_sec) * 1e3
            + (num_layers * standard_launch_overhead_us) / 1e3
        )

        # 2. 2-Level Cascade (`BatchDecodeWithSharedPrefixPagedKVCacheWrapper`):
        # Level 0 reads L_sys KV ONCE for all B*H_q_local packed queries (computes GEMM-like BatchPrefill with causal=False).
        # Level 1 reads L_rag + L_uniq per request via Paged-KV Decode, then runs MergeStateInPlaceKernel.
        l2_sys_kv_bytes_layer = 1 * l_sys * kv_bytes_per_token_per_layer
        l2_suffix_kv_bytes_layer = b * (l_rag + l_uniq) * kv_bytes_per_token_per_layer
        # Merge state reads/writes partial V (BF16) + LSE (FP32 = 4B)
        l2_merge_bytes_layer = b * h_q_local * (head_dim * bytes_per_elem * 3 + 4 * 3)
        l2_total_bytes_80l = (l2_sys_kv_bytes_layer + l2_suffix_kv_bytes_layer + l2_merge_bytes_layer) * num_layers
        l2_flops_80l = flat_flops_80l + (b * h_q_local * head_dim * 6) * num_layers
        l2_time_ms = (
            max(l2_total_bytes_80l / hbm_bw_bytes_per_sec, l2_flops_80l / tensor_core_flops_per_sec) * 1e3
            + (num_layers * 3 * standard_launch_overhead_us) / 1e3
        )

        # 3. 3-Level MultiLevelCascadeAttentionWrapper + PersistentVariableLengthMergeStates + PDL (`enable_pdl=True`):
        # Level 0 reads L_sys KV ONCE across all B queries.
        # Level 1 reads L_rag KV ONCE per RAG group (G=4 groups).
        # Level 2 reads only L_uniq (1,024 tokens) per request, followed by fused multi-state merge with PDL overlap.
        l3_sys_kv_bytes_layer = 1 * l_sys * kv_bytes_per_token_per_layer
        l3_rag_kv_bytes_layer = num_rag_groups * l_rag * kv_bytes_per_token_per_layer
        l3_uniq_kv_bytes_layer = b * l_uniq * kv_bytes_per_token_per_layer
        # PersistentVariableLengthMergeStatesKernel merges all 3 partial states in a single pass
        l3_merge_bytes_layer = b * h_q_local * (head_dim * bytes_per_elem * 4 + 4 * 4)
        l3_total_bytes_80l = (
            l3_sys_kv_bytes_layer + l3_rag_kv_bytes_layer + l3_uniq_kv_bytes_layer + l3_merge_bytes_layer
        ) * num_layers
        l3_flops_80l = flat_flops_80l + (b * h_q_local * head_dim * 10) * num_layers
        l3_time_ms_no_pdl = (
            max(l3_total_bytes_80l / hbm_bw_bytes_per_sec, l3_flops_80l / tensor_core_flops_per_sec) * 1e3
            + (num_layers * 4 * standard_launch_overhead_us) / 1e3
        )
        l3_time_ms_pdl = (
            max(l3_total_bytes_80l / hbm_bw_bytes_per_sec, l3_flops_80l / tensor_core_flops_per_sec) * 1e3
            + (num_layers * 4 * pdl_launch_overhead_us) / 1e3
        )

        # Effective Arithmetic Intensity (FLOPs / Byte) on the shared Level-0 kernel vs Flat Paged-KV
        ai_flat = flat_flops_80l / flat_total_bytes_80l
        ai_cascade_l3 = l3_flops_80l / l3_total_bytes_80l

        rows.append(
            {
                "batch_size": b,
                "flat_paged_kv_hbm_gib": round(flat_total_bytes_80l / (1024**3), 3),
                "cascade_2level_hbm_gib": round(l2_total_bytes_80l / (1024**3), 3),
                "cascade_3level_hbm_gib": round(l3_total_bytes_80l / (1024**3), 3),
                "hbm_traffic_reduction_pct": round((1.0 - l3_total_bytes_80l / flat_total_bytes_80l) * 100.0, 2),
                "flat_paged_kv_latency_ms": round(flat_time_ms, 3),
                "cascade_2level_latency_ms": round(l2_time_ms, 3),
                "cascade_3level_no_pdl_latency_ms": round(l3_time_ms_no_pdl, 3),
                "cascade_3level_pdl_latency_ms": round(l3_time_ms_pdl, 3),
                "speedup_3level_pdl_vs_flat": round(flat_time_ms / l3_time_ms_pdl, 2),
                "arithmetic_intensity_flat_flops_per_byte": round(ai_flat, 2),
                "arithmetic_intensity_3level_flops_per_byte": round(ai_cascade_l3, 2),
            }
        )

    b128 = next(r for r in rows if r["batch_size"] == 128)
    numerics = verify_multi_level_cascade_numerics()

    return {
        "experiment_id": "flashinfer_cascade_paged_kv",
        "model_config": {
            "model": "Llama-3.1-70B-GQA-TP4",
            "num_layers": num_layers,
            "h_q_local": h_q_local,
            "h_kv_local": h_kv_local,
            "head_dim": head_dim,
            "l_sys_tokens": l_sys,
            "l_rag_tokens": l_rag,
            "l_uniq_tokens": l_uniq,
            "l_total_tokens": l_total,
            "num_rag_groups": num_rag_groups,
        },
        "numerics_verification": numerics,
        "batch_sweep": rows,
        "summary": {
            "b128_flat_hbm_gib": b128["flat_paged_kv_hbm_gib"],
            "b128_cascade3_hbm_gib": b128["cascade_3level_hbm_gib"],
            "b128_hbm_reduction_pct": b128["hbm_traffic_reduction_pct"],
            "b128_flat_latency_ms": b128["flat_paged_kv_latency_ms"],
            "b128_cascade3_pdl_latency_ms": b128["cascade_3level_pdl_latency_ms"],
            "b128_speedup_x": b128["speedup_3level_pdl_vs_flat"],
            "b128_ai_flat_flops_per_byte": b128["arithmetic_intensity_flat_flops_per_byte"],
            "b128_ai_cascade3_flops_per_byte": b128["arithmetic_intensity_3level_flops_per_byte"],
            "max_merge_abs_error": numerics["max_output_abs_error"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json matches.")
    args = parser.parse_args()

    results = simulate_cascade_paged_kv_sweep()
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
