"""Phase 95 Track 2: SGLang Pipelined Ulysses Attention, DSV4 Non-EP TBO, AMD gfx950 KDA Decode, HiCache L3 Layout, Confirmed-First Pending Prefix Router, vLLM DSV4.1 DSpark PCP, OOT Block-Scaled FP8, H100 Batch-Invariant Matmul, KV Offload Prefix Order & TensorRT-LLM Modular Mamba Cache Simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_sglang_ulysses_dsv4_gfx950_hicache_router() -> dict[str, object]:
    """Model SGLang pipelined Ulysses own-row copy overlap (PR #46308), DSV4 non-EP TBO decode child padding
    (PR #33250), AMD gfx950 Kimi-K3 KDA fused decode (PR #45713), HiCache Mooncake L3 key exposure (PR #40467),
    and sgl-router confirmed-first pending prefix policy with follower cap (PR #46505).
    """
    # PR #46308: fill_all(blocks, part) fills q (part=0) and launches q All-to-All exchange while k (part=1) is normed/copied
    q_norm_copy_us = 42.0
    q_a2a_exchange_us = 95.0
    k_norm_copy_us = 42.0
    k_a2a_exchange_us = 95.0
    serial_ulysses_prep_us = q_norm_copy_us + k_norm_copy_us + q_a2a_exchange_us + k_a2a_exchange_us  # 274.0 us
    pipelined_ulysses_prep_us = (
        q_norm_copy_us + max(q_a2a_exchange_us, k_norm_copy_us) + k_a2a_exchange_us
    )  # 42 + 95 + 95 = 232.0 us
    hidden_k_copy_us = serial_ulysses_prep_us - pipelined_ulysses_prep_us  # 42.0 us

    # PR #33250: DeepSeek-V4 non-EP TBO decode child padding for attn_tp > 1
    # Previously padded with batch_size instead of decode token count when MTP/spec tokens > 1 per req
    attn_tp_size = 4
    decode_batch_reqs = 16
    tokens_per_decode_req = 2
    tbo_padded_len = 32
    buggy_unpadded_tokens = decode_batch_reqs  # 16 (mismatched vs 32 tokens)
    fixed_decode_child_tokens = decode_batch_reqs * tokens_per_decode_req  # 32

    # PR #45713: AMD gfx950 Kimi-K3 KDA HIP fused decode kernel
    # 1 workgroup of 256 threads per (token, head), 12 heads of 128 dim per rank, 128x128 FP32 state, fuses 4 stages
    gfx950_threads_per_wg = 256
    gfx950_heads_per_rank = 12
    gfx950_head_dim = 128
    unfused_kernel_launches = 4
    fused_kernel_launches = 1

    # PR #40467: HiCache Mooncake L3 object-key layout merged across PP stages
    pp_stages = 4
    dp_ranks = 2
    merged_l3_pool_descriptors = pp_stages * dp_ranks  # 8

    # PR #46505: sgl-router confirmed-first pending prefixes (--cache-pending-prefix-consult-below <ratio>)
    consult_below_ratio = 0.5
    follower_cap = 4
    # Cold prompt with confirmed match = 0.25 (< 0.5) and pending match = 0.75 (> 0.25) -> follows pending up to cap
    # Warm prompt with confirmed match = 0.60 (>= 0.5) and pending match = 0.90 -> stays on confirmed worker
    cold_routed_to_pending = sum(1 for i in range(8) if 0.25 < consult_below_ratio and i < follower_cap)
    warm_routed_to_confirmed = True

    return {
        "serial_ulysses_prep_us": serial_ulysses_prep_us,
        "pipelined_ulysses_prep_us": pipelined_ulysses_prep_us,
        "hidden_k_copy_us": hidden_k_copy_us,
        "attn_tp_size": attn_tp_size,
        "buggy_unpadded_tokens": buggy_unpadded_tokens,
        "fixed_decode_child_tokens": fixed_decode_child_tokens,
        "tbo_padded_len": tbo_padded_len,
        "gfx950_threads_per_wg": gfx950_threads_per_wg,
        "gfx950_heads_per_rank": gfx950_heads_per_rank,
        "gfx950_head_dim": gfx950_head_dim,
        "unfused_kernel_launches": unfused_kernel_launches,
        "fused_kernel_launches": fused_kernel_launches,
        "merged_l3_pool_descriptors": merged_l3_pool_descriptors,
        "cold_routed_to_pending_capped": cold_routed_to_pending,
        "warm_routed_to_confirmed": warm_routed_to_confirmed,
    }


def simulate_vllm_pcp_dspark_fp8_batch_invariant_kv_offload_trtllm() -> dict[str, object]:
    """Model vLLM DeepSeek-V4.1 PCP + fixed-length DSpark spec decode (PR #61994),
    OOT block-scaled FP8 linear kernel registration (PR #62026),
    H100 batch-invariant matmul configs for Llama-3.1-8B & Qwen3-8B (PR #62219),
    KV offload LRU prefix preservation via get_offload_key_position (PR #61050),
    and TensorRT-LLM modular Mamba cache managers (PR #18843).
    """
    # PR #61994: vLLM DeepSeek-V4.1 PCP (pcp_size in {2, 4}) supports fixed-length DSpark (tail_size=1),
    # rejects MTP and adaptive DSpark under PCP
    supported_pcp_sizes = [2, 4]
    dspark_fixed_supported = True
    mtp_under_pcp_rejected = True
    adaptive_dspark_under_pcp_rejected = True

    # PR #62026: vLLM register_linear_kernel supported kernel_types expanded from 3 to 8
    supported_oot_kernel_types = ["mp", "int8", "fp8", "fp8_block", "mxfp8", "nvfp4", "mxfp4", "mxfp6"]

    # PR #62219: H100 batch-invariant matmul shape (24576, 4096) for Qwen3-8B gate_up_proj
    qwen3_8b_gate_up_shape = [24576, 4096]
    batch_invariant_block_k = 128
    batch_invariant_m_buckets = [1, 4, 8, 16, 32, 64, 256, 512, 1024, 2048]

    # PR #61050: KV offload LRU store batch preserves prefix order via req_context.get_offload_key_position(key)
    # Suppose cache capacity = 4 chunks, and a running request stores 6 chunks [0, 1, 2, 3, 4, 5] in arbitrary hash order
    cache_capacity = 4
    stored_positions_sorted_by_req_context = [0, 1, 2, 3, 4, 5]
    # To preserve usable prefix [0, 1, 2, 3] when evicting tail or ordering LRU so earliest prefix chunks stay warmest:
    # PR #61050 orders insertion so prefix chunks 0..3 are retained as a contiguous prefix instead of evicting chunk 0!
    retained_prefix_chunks_after_fix = [0, 1, 2, 3]
    usable_prefix_hit_chunks_before_fix = 0  # chunk 0 evicted first when inserted in forward order into smaller LRU!
    usable_prefix_hit_chunks_after_fix = len(retained_prefix_chunks_after_fix)

    # PR #18843: TensorRT-LLM modular Mamba cache manager binds mamba_layer_cache views once per runner
    # instead of calling get_intermediate_conv_states / get_intermediate_ssm_states every forward step
    gdn_layers = 48
    decode_steps = 100
    view_allocations_before = gdn_layers * 2 * decode_steps  # 9600
    view_allocations_after = gdn_layers * 2  # 96 (bound once in _bind_step_buffers)

    return {
        "supported_pcp_sizes": supported_pcp_sizes,
        "dspark_fixed_supported": dspark_fixed_supported,
        "mtp_under_pcp_rejected": mtp_under_pcp_rejected,
        "adaptive_dspark_under_pcp_rejected": adaptive_dspark_under_pcp_rejected,
        "supported_oot_kernel_types_count": len(supported_oot_kernel_types),
        "qwen3_8b_gate_up_shape": qwen3_8b_gate_up_shape,
        "batch_invariant_block_k": batch_invariant_block_k,
        "batch_invariant_m_buckets_count": len(batch_invariant_m_buckets),
        "cache_capacity": cache_capacity,
        "usable_prefix_hit_chunks_before_fix": usable_prefix_hit_chunks_before_fix,
        "usable_prefix_hit_chunks_after_fix": usable_prefix_hit_chunks_after_fix,
        "trtllm_mamba_view_allocations_before": view_allocations_before,
        "trtllm_mamba_view_allocations_after": view_allocations_after,
    }


def build_summary() -> dict[str, object]:
    return {
        "experiment": "sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95",
        "phase": 95,
        "priority": 104,
        "evidence_class": "own-offline",
        "sglang_ulysses_dsv4_gfx950_hicache_router": simulate_sglang_ulysses_dsv4_gfx950_hicache_router(),
        "vllm_pcp_fp8_batch_invariant_kv_trtllm": simulate_vllm_pcp_dspark_fp8_batch_invariant_kv_offload_trtllm(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json mismatch")
        print("sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95: OK")
        return 0

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
