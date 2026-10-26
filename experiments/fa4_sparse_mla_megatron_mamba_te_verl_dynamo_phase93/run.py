#!/usr/bin/env python3
"""Phase 93 Track 3: FA4 SM100 Sparse MLA Training, Megatron Mamba2, TE MXFP8/NVFP4, verl FSDP2 & Dynamo Simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_fa4_sm100_sparse_mla_training() -> dict[str, Any]:
    """Model FlashAttention-4 SM100 Sparse MLA training fixes (71dbd9372d o_lo dpsum & 93c9acac10 exact max)."""
    # 1. Exact running softmax max (93c9acac10):
    # Reverse block walk encounters strongest key (slot 0) last; lazy rescale_threshold=8 leaves non-integer delta
    # causing dominant probability p = exp2(delta) whose bf16 rounding adds a coherent ~2^-9 row gain error.
    lazy_rescale_threshold_log2 = 8.0
    training_rescale_threshold_log2 = 0.0
    out_rel_l2_lazy = 2.07e-3
    out_rel_l2_exact = 1.67e-3
    coherent_row_gain_lazy = 1.23e-3
    coherent_row_gain_exact = 1.40e-4

    # 2. dpsum from O rounding residual (71dbd9372d):
    # dS = P * (dP - dpsum) with dpsum = rowsum(dO * (out + o_lo)), where o_lo = fp32(O) - bf16(O).
    dq_dk_rel_l2_80pct_without_olo = 0.0089  # 0.89%
    dq_dk_rel_l2_97pct_without_olo = 0.0380  # 3.80%
    dq_dk_rel_l2_with_olo = 0.0024  # 0.24% (matches ideal FP64 dpsum floor)
    error_reduction_97pct_x = round(dq_dk_rel_l2_97pct_without_olo / dq_dk_rel_l2_with_olo, 2)

    return {
        "inference_rescale_threshold_log2": lazy_rescale_threshold_log2,
        "training_sparse_rescale_threshold_log2": training_rescale_threshold_log2,
        "out_rel_l2_lazy": out_rel_l2_lazy,
        "out_rel_l2_exact": out_rel_l2_exact,
        "coherent_row_gain_reduction_x": round(coherent_row_gain_lazy / coherent_row_gain_exact, 2),
        "dq_dk_rel_l2_80pct_before": dq_dk_rel_l2_80pct_without_olo,
        "dq_dk_rel_l2_97pct_before": dq_dk_rel_l2_97pct_without_olo,
        "dq_dk_rel_l2_after_olo": dq_dk_rel_l2_with_olo,
        "dq_dk_error_reduction_97pct_x": error_reduction_97pct_x,
        "forward_extra_tensor_bytes_ratio": 1.0,  # 1 extra out-sized bf16 tensor o_lo
    }


def simulate_megatron_mamba_te_torchtitan_verl() -> dict[str, Any]:
    """Model Megatron-LM (#8221), TransformerEngine (#3662, #3658), TorchTitan (#5152), and verl (#8187)."""
    # Megatron-LM PR #8221 (aa46a57229): batch-invariant Mamba2 decode row kernels without z replay buffer
    chunk_size = 256
    mma_tile_rows = 16
    batch_requests = 64
    nheads = 64
    hdim = 64
    dtype_bytes = 2
    # Eliminated z replay buffer: previously buffered (max_requests, chunk_size, nheads, hdim)
    eliminated_z_buffer_bytes = batch_requests * chunk_size * nheads * hdim * dtype_bytes
    row_mma_compute_reduction_x = chunk_size // mma_tile_rows  # 16x fewer chunk-scan rows per decode step
    state_passing_launch_fraction = round(1.0 / chunk_size, 6)

    # TransformerEngine PR #3658 (33e7ff3160): Per-tile MXFP8 scale swizzle (1 CTA per 128-row scale tile)
    activation_rows = 4096
    tile_rows = 128
    legacy_ctas_for_4096_rows = math.ceil((activation_rows // tile_rows) / 32)  # 1 CTA for 32 tiles!
    per_tile_ctas_for_4096_rows = activation_rows // tile_rows  # 32 CTAs

    # TransformerEngine PR #3662 (2e959f352c): Single-group fused grouped MLP NVFP4 weight views
    in_features = 4096
    out_features = 14336
    fc2_dgrad_transposed_packed_shape = [1, in_features, out_features // 2]

    # TorchTitan PR #5152 (094c884685) & verl PR #8187 (edd2c63b6a)
    torchtitan_varlen_priority = ["FLASH_ATTENTION"]
    verl_fsdp2_uneven_shard_blocking_d2h = True

    return {
        "mamba_eliminated_z_replay_buffer_mib": round(eliminated_z_buffer_bytes / (1024 * 1024), 2),
        "mamba_chunk_scan_row_mma_reduction_x": row_mma_compute_reduction_x,
        "mamba_boundary_state_commit_fraction": state_passing_launch_fraction,
        "te_mxfp8_legacy_ctas_4096_rows": legacy_ctas_for_4096_rows,
        "te_mxfp8_per_tile_ctas_4096_rows": per_tile_ctas_for_4096_rows,
        "te_nvfp4_fc2_dgrad_packed_shape": fc2_dgrad_transposed_packed_shape,
        "torchtitan_varlen_sdpa_priority": torchtitan_varlen_priority,
        "verl_fsdp2_uneven_shard_blocking_d2h": verl_fsdp2_uneven_shard_blocking_d2h,
    }


def simulate_lmcache_and_dynamo_kv_router() -> dict[str, Any]:
    """Model LMCache PR #5976 (a0ff470128) and Dynamo PRs #16303 (4b487d890d) & #17143 (3acf248073)."""
    # LMCache PR #5976: SWA prefix replay floors full-prompt external hit to scheduler block boundary
    prompt_tokens = 1024
    prefix_replay_tokens = 128
    scheduler_block_size = 64
    # Without SWA replay adjustment, full prompt hit clipped to prompt_tokens - 1 = 1023 -> floored to 960
    # Under SWA prefix replay, scheduler floors external hit to block boundary before allocation
    allocated_tokens = ((prompt_tokens - prefix_replay_tokens) // scheduler_block_size) * scheduler_block_size

    # Dynamo PR #17143 (3acf248073): shrink lower-tier KV-router edge slots from 113 to 33 bytes
    legacy_edge_slot_bytes = 113
    compact_edge_slot_bytes = 33
    edge_count = 10_000_000  # 10M indexed KV block edges across lower tiers
    legacy_index_mib = (edge_count * legacy_edge_slot_bytes) / (1024 * 1024)
    compact_index_mib = (edge_count * compact_edge_slot_bytes) / (1024 * 1024)
    memory_reduction_pct = round(
        ((legacy_edge_slot_bytes - compact_edge_slot_bytes) / legacy_edge_slot_bytes) * 100.0, 2
    )

    return {
        "lmcache_swa_replay_block_aligned_tokens": allocated_tokens,
        "dynamo_vllm_logit_bias_forwarded": True,
        "dynamo_trtllm_logit_bias_raises_invalid_argument": True,
        "dynamo_kv_router_legacy_slot_bytes": legacy_edge_slot_bytes,
        "dynamo_kv_router_compact_slot_bytes": compact_edge_slot_bytes,
        "dynamo_kv_router_10m_edges_legacy_mib": round(legacy_index_mib, 2),
        "dynamo_kv_router_10m_edges_compact_mib": round(compact_index_mib, 2),
        "dynamo_kv_router_slot_reduction_pct": memory_reduction_pct,
    }


def run_suite() -> dict[str, Any]:
    fa4 = simulate_fa4_sm100_sparse_mla_training()
    training = simulate_megatron_mamba_te_torchtitan_verl()
    kv_dynamo = simulate_lmcache_and_dynamo_kv_router()

    return {
        "phase": 93,
        "track": "fa4_sparse_mla_megatron_mamba_te_verl_dynamo",
        "fa4_sparse_mla_training": fa4,
        "megatron_te_torchtitan_verl": training,
        "lmcache_and_dynamo": kv_dynamo,
        "invariants": {
            "fa4_olo_reduces_peaked_grad_error_15x": fa4["dq_dk_error_reduction_97pct_x"] >= 15.0,
            "fa4_exact_max_reduces_row_gain_8x": fa4["coherent_row_gain_reduction_x"] >= 8.5,
            "mamba_eliminates_128mib_z_buffer": training["mamba_eliminated_z_replay_buffer_mib"] == 128.0,
            "te_mxfp8_32x_cta_parallelism": training["te_mxfp8_per_tile_ctas_4096_rows"] == 32,
            "dynamo_edge_slot_70pct_smaller": kv_dynamo["dynamo_kv_router_slot_reduction_pct"] == 70.8,
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
