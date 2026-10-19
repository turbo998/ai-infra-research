#!/usr/bin/env python3
"""Deterministic simulator for Decoupled Attention-DP + Expert-EP Serving (SGLang dp_attention).

Models:
1. MLA KV Cache Replication vs Attention-DP (DP_attn=64, EP=64) HBM capacity & max concurrency.
2. SGLang DpPaddingMode (python/sglang/srt/layers/dp_attention.py):
   - MAX_LEN without pad masking (stale hidden rows routed to experts -> GEMM blow-up)
   - MAX_LEN + _mask_dp_pad_topk_ids_kernel (topk_ids = -1 sentinel filtering)
   - SUM_LEN + _dp_gather_via_all_gatherv_fp8 (group-128 uint8 + fp32 scale wire format)
   - Two-Batch Overlap (TBO: dp_gather_partial_async + dp_reduce_scatterv_async on dp_tbo_comm
     with persistent _tbo_event & _TBO_PERSIST_BUF) + KV-Byte Watermark Routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).with_name("results.json")


def simulate_kv_capacity_comparison() -> dict[str, Any]:
    """Compare TP=8 replicated MLA KV vs DP_attn=64 partitioned MLA KV across 8x 8-H100 nodes."""
    num_gpus = 64
    hbm_per_gpu_gib = 80.0
    # 671B FP8 weights sharded across EP=64 (~10.5 GiB/GPU + 3.5 GiB runtime/buffers = 14.0 GiB)
    tp8_weight_plus_runtime_gib = 24.5  # TP=8 replicates non-MoE & larger workspace buffers
    dp64_weight_plus_runtime_gib = 15.2  # DP_attn=64 + persistent TBO buffer (_TBO_PERSIST_BUF)
    # DeepSeek-V3/R1 MLA KV Cache: 61 layers * (512 latent + 64 RoPE) * 1 byte (FP8) = 35,136 B/tok
    mla_kv_bytes_per_token = 35136

    tp8_free_kv_gib_per_gpu = hbm_per_gpu_gib - tp8_weight_plus_runtime_gib
    dp64_free_kv_gib_per_gpu = hbm_per_gpu_gib - dp64_weight_plus_runtime_gib

    # Under TP=8 (8 DP groups of 8 GPUs), MLA KV is replicated across all 8 TP ranks!
    tp8_effective_pod_kv_gib = round((num_gpus / 8) * tp8_free_kv_gib_per_gpu, 2)
    # Under DP_attn=64 (attn_tp_size=1), each GPU holds independent requests' MLA KV!
    dp64_effective_pod_kv_gib = round(num_gpus * dp64_free_kv_gib_per_gpu, 2)

    tp8_max_tokens_m = round(
        (tp8_effective_pod_kv_gib * (1024**3)) / mla_kv_bytes_per_token / 1e6, 3
    )
    dp64_max_tokens_m = round(
        (dp64_effective_pod_kv_gib * (1024**3)) / mla_kv_bytes_per_token / 1e6, 3
    )

    return {
        "pod_gpus": num_gpus,
        "mla_kv_bytes_per_token": mla_kv_bytes_per_token,
        "tp8_replicated_pod_kv_gib": tp8_effective_pod_kv_gib,
        "dp64_partitioned_pod_kv_gib": dp64_effective_pod_kv_gib,
        "tp8_max_concurrent_tokens_m": tp8_max_tokens_m,
        "dp64_max_concurrent_tokens_m": dp64_max_tokens_m,
        "kv_capacity_multiplier_x": round(dp64_max_tokens_m / tp8_max_tokens_m, 2),
        "max_32k_sessions_tp8": int((tp8_max_tokens_m * 1e6) // 32768),
        "max_32k_sessions_dp64": int((dp64_max_tokens_m * 1e6) // 32768),
    }


def simulate_dp_gather_and_skew_modes() -> list[dict[str, Any]]:
    """Evaluate 61-layer hybrid batch (64 DP ranks, skewed token counts) across 5 DP-MoE modes."""
    num_dp_ranks = 64
    num_layers = 61
    hidden_size = 7168

    # Skewed token distribution across 64 DP ranks (mean = 42.5 tokens/rank, max = 128, sum = 2720)
    # Represents mixed chunked-extend + variable-speculative-decode microbatches
    sum_tokens = 2720
    max_tokens_per_rank = 128
    padded_max_tokens = num_dp_ranks * max_tokens_per_rank  # 8192 rows under MAX_LEN

    modes = [
        {
            "mode_id": "max_len_unmasked_bf16",
            "label": "MAX_LEN AllGather (BF16, Unmasked Pad Rows Routed to Experts)",
            "wire_bytes_per_elem": 2.0,
            "gathered_rows": padded_max_tokens,
            "expert_routed_rows": padded_max_tokens,
            "attn_skew_Wait_pct": 38.4,
            "tbo_overlap_ratio": 0.0,
            "reserved_allocator_gib": 268.0,  # Per-layer dynamic allocation churn without _TBO_PERSIST_BUF
        },
        {
            "mode_id": "max_len_masked_topk_neg1",
            "label": "MAX_LEN AllGather (BF16 + _mask_dp_pad_topk_ids_kernel topk=-1)",
            "wire_bytes_per_elem": 2.0,
            "gathered_rows": padded_max_tokens,
            "expert_routed_rows": sum_tokens,  # Pad rows dropped via -1 sentinel
            "attn_skew_Wait_pct": 38.4,
            "tbo_overlap_ratio": 0.0,
            "reserved_allocator_gib": 42.0,
        },
        {
            "mode_id": "sum_len_gatherv_bf16",
            "label": "SUM_LEN AllGatherv + ReduceScatterv (BF16 Variable-Length)",
            "wire_bytes_per_elem": 2.0,
            "gathered_rows": sum_tokens,
            "expert_routed_rows": sum_tokens,
            "attn_skew_Wait_pct": 38.4,
            "tbo_overlap_ratio": 0.0,
            "reserved_allocator_gib": 34.5,
        },
        {
            "mode_id": "sum_len_gatherv_fp8_group128",
            "label": "SUM_LEN AllGatherv FP8 Wire (_dp_gather_via_all_gatherv_fp8 Group-128)",
            "wire_bytes_per_elem": 1.03125,  # 1B uint8 + 4B fp32 scale per 128 elems
            "gathered_rows": sum_tokens,
            "expert_routed_rows": sum_tokens,
            "attn_skew_Wait_pct": 38.4,
            "tbo_overlap_ratio": 0.0,
            "reserved_allocator_gib": 29.8,
        },
        {
            "mode_id": "sum_len_fp8_tbo_kv_balanced",
            "label": "SUM_LEN FP8 Gatherv + Two-Batch Overlap (dp_tbo_comm) + KV-Watermark Router",
            "wire_bytes_per_elem": 1.03125,
            "gathered_rows": sum_tokens,
            "expert_routed_rows": sum_tokens,
            "attn_skew_Wait_pct": 4.2,  # KV-byte watermark routing balances per-DP attention time
            "tbo_overlap_ratio": 0.78,  # 78% of DP gather/combine hidden behind other ubatch compute
            "reserved_allocator_gib": 10.4,  # Persistent _TBO_PERSIST_BUF + _tbo_event reuse
        },
    ]

    rows: list[dict[str, Any]] = []
    base_attn_mean_ms = 5.80  # 61 layers mean MLA attention compute across 64 DP ranks
    for m in modes:
        # Attention barrier time = mean / (1 - skew_wait)
        attn_effective_ms = round(
            base_attn_mean_ms / (1.0 - m["attn_skew_Wait_pct"] / 100.0), 3
        )
        attn_barrier_bubble_ms = round(attn_effective_ms - base_attn_mean_ms, 3)

        # Gather + ReduceScatter bytes per layer across 64 ranks (effective fabric 140 GB/s hybrid)
        gather_bytes_per_layer = m["gathered_rows"] * hidden_size * m["wire_bytes_per_elem"]
        combine_bytes_per_layer = m["gathered_rows"] * hidden_size * 2.0  # Combine stays BF16 for SUM
        raw_dp_comm_ms = round(
            num_layers * ((gather_bytes_per_layer + combine_bytes_per_layer) / 140e9) * 1e3,
            3,
        )
        exposed_dp_comm_ms = round(raw_dp_comm_ms * (1.0 - m["tbo_overlap_ratio"]), 3)

        # MoE Expert Grouped GEMM + All-to-All scales with expert_routed_rows
        moe_compute_and_ep_ms = round(6.90 * (m["expert_routed_rows"] / sum_tokens), 3)

        step_latency_ms = round(
            attn_effective_ms + exposed_dp_comm_ms + moe_compute_and_ep_ms, 3
        )
        throughput_tok_per_s = round((sum_tokens / step_latency_ms) * 1000.0, 1)

        rows.append(
            {
                **m,
                "attn_mean_compute_ms": base_attn_mean_ms,
                "attn_barrier_bubble_ms": attn_barrier_bubble_ms,
                "raw_dp_comm_ms": raw_dp_comm_ms,
                "exposed_dp_comm_ms": exposed_dp_comm_ms,
                "moe_compute_and_ep_ms": moe_compute_and_ep_ms,
                "step_latency_ms": step_latency_ms,
                "pod_throughput_tok_per_s": throughput_tok_per_s,
            }
        )

    base_tps = rows[0]["pod_throughput_tok_per_s"]
    for r in rows:
        r["throughput_gain_vs_unmasked_x"] = round(
            r["pod_throughput_tok_per_s"] / base_tps, 2
        )
    return rows


def build_results() -> dict[str, Any]:
    kv_cap = simulate_kv_capacity_comparison()
    modes = simulate_dp_gather_and_skew_modes()

    m_unmasked = modes[0]
    m_masked = modes[1]
    m_fp8_tbo = modes[4]

    return {
        "experiment_id": "attention_dp_moe_ep_schedule",
        "status": "own-offline",
        "source_contracts": [
            "sgl-project/sglang@62cee1b15c22350b17a9256fc6ad5777feb2ec70:python/sglang/srt/layers/dp_attention.py",
        ],
        "summary_metrics": {
            "kv_capacity_multiplier_x": kv_cap["kv_capacity_multiplier_x"],
            "tp8_max_concurrent_tokens_m": kv_cap["tp8_max_concurrent_tokens_m"],
            "dp64_max_concurrent_tokens_m": kv_cap["dp64_max_concurrent_tokens_m"],
            "unmasked_max_len_step_ms": m_unmasked["step_latency_ms"],
            "masked_topk_neg1_step_ms": m_masked["step_latency_ms"],
            "fp8_tbo_balanced_step_ms": m_fp8_tbo["step_latency_ms"],
            "unmasked_tps": m_unmasked["pod_throughput_tok_per_s"],
            "fp8_tbo_balanced_tps": m_fp8_tbo["pod_throughput_tok_per_s"],
            "throughput_gain_x": m_fp8_tbo["throughput_gain_vs_unmasked_x"],
            "allocator_reserved_reduction_gib": round(
                m_unmasked["reserved_allocator_gib"] - m_fp8_tbo["reserved_allocator_gib"], 1
            ),
        },
        "kv_capacity_comparison": kv_cap,
        "dp_gather_modes": modes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against sealed results.json")
    args = parser.parse_args()

    computed = build_results()
    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}")
        expected = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if computed != expected:
            raise SystemExit("attention_dp_moe_ep_schedule: results mismatch")
        print("attention_dp_moe_ep_schedule: OK")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
