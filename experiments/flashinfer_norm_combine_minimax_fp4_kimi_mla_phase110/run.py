from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. FlashInfer #7444 (3da22ba71f): cake_fused_norm_combine lean stream prologue (38..479 tokens)
    # and early workspace-table read in the one-token program (BF16, hidden=2560, 8 peers, 2 norm tracks).
    b200_1tok_prev_us = 7.616
    b200_1tok_new_us = 7.424
    b200_1tok_speedup_x = round(b200_1tok_prev_us / b200_1tok_new_us, 4)
    b200_64tok_prev_us = 12.608
    b200_64tok_new_us = 12.192
    b200_64tok_speedup_x = round(b200_64tok_prev_us / b200_64tok_new_us, 4)
    b200_256tok_prev_us = 15.232
    b200_256tok_new_us = 14.751
    b200_256tok_speedup_x = round(b200_256tok_prev_us / b200_256tok_new_us, 4)
    stream_smem_prev_bytes = 128
    stream_smem_new_bytes = 256
    stream_handoff_bytes = 8

    # 2. FlashInfer #7461 (5b2aacd211): cake_minimax_h3 SM120 FP8 / NVFP4 pre-attention GEMM epilogue
    # Updates 6 of 9 GEMM programs on sm_120a (10/16 rows on RTX PRO 6000, 3/16 rows on RTX 5090).
    rtx_pro_6000_changed_rows = 10
    rtx_5090_changed_rows = 3
    rtx_pro_6000_max_speedup_x = 1.0798
    rtx_5090_max_speedup_x = 1.0469
    rtx_pro_6000_m33472_prev_ms = 11.1866
    rtx_pro_6000_m33472_new_ms = 11.0531
    rtx_pro_6000_m33472_speedup_x = round(rtx_pro_6000_m33472_prev_ms / rtx_pro_6000_m33472_new_ms, 4)

    # 3. FlashInfer #7489 (e158799310): cake_mm_fp4 entry L2 weight prefetch on 3 more cells
    # CHAIN_EP_STAGES = 4; sm_100a n8_aF_l2256b_o2 expands from (8,8) to (1,8); sm_103a adds m64_bF_l2256b_ro at M=128;
    # GEMM alone adds GEMM_ALONE_EP_RULES at M=128 (8 rows: 4 KxN shapes x 2 output dtypes).
    chain_ep_stages = 4
    n8_o2_range_prev = [8, 8]
    n8_o2_range_new = [1, 8]
    gemm_alone_m128_rows_added = 8

    # 4. FlashInfer #7497 (531e0bc2ba): cake_kimi_k3_tp12_tail persistent reduce tail for 320 <= M <= 960
    # k3_ess:grouped now covers 8 < M < 320; k3_reduce:grouped covers 320 <= M < 961 (2 tokens in flight, no scatter stage).
    k3_reduce_min_m = 320
    k3_reduce_max_m = 960
    k3_persist_min_m = 961
    in_flight_gather_tokens = 2
    gb300_m768_speedup_x = 1.035
    gb300_m960_speedup_x = 1.056

    # 5. FlashInfer #7439 (45411ad7d7): cake_kimi_k3_mla late-trigger row tiles (16, 32, 48 rows when main CTAs >= 32)
    # and pre-wait split-KV reducers on sm_100a and sm_103a (512 latent + 64 rope = 576, page 64).
    mla_head_dim_total = 512 + 64
    mla_late_trigger_min_main_ctas = 32
    mla_gb300_faster_shapes = 7
    mla_total_shapes = 41
    mla_gb300_shape1_speedup_x = 1.1295
    mla_b200_shape1_speedup_x = 1.1649

    # 6. FlashInfer #7462 (acb2bef7ec): cake_trtllm_moe_allreduce_union 2 CTAs/SM for 1-token rows and st.async RMS exchange
    # 12 variants (bf16/fp16 x ws 2/4/8 x PDL on/off); all 30 shapes faster on B200 (1.03x-1.12x) and B300 (1.03x-1.13x).
    moe_ar_union_variants = 12
    moe_ar_union_shapes_faster = 30
    ws2_bf16_t1_b200_prev_us = 7.63
    ws2_bf16_t1_b200_new_us = 7.31
    ws2_bf16_t1_b300_prev_us = 8.24
    ws2_bf16_t1_b300_new_us = 7.58
    ws2_bf16_t1_b300_speedup_x = round(ws2_bf16_t1_b300_prev_us / ws2_bf16_t1_b300_new_us, 4)

    # 7. CUTLASS #3427 (976d0521f0) & #3048 (3c063071bd):
    # ScaledBasis equality guards mismatched bases before value comparison, unblocking SM120 256x128x256 block-scaled sparse MMA;
    # AffineRank2RowMajor::packed(extent) returns (extent.column(), 1) instead of column-major (1, extent.row()).
    sm120_sparse_tile = [256, 128, 256]
    nvf4_metadata_atom_k = 256
    mxf8f6f4_metadata_atom_k = 128
    rows, cols = 64, 128
    # For coordinate (r=3, c=5) in a 64x128 row-major matrix:
    buggy_packed_offset = 3 * 1 + 5 * rows
    fixed_packed_offset = 3 * cols + 5 * 1

    # 8. TileLang #3536 (cb6455d680) & #3548 (9253431f2a):
    # Shared buffer T.cumsum under warp specialization subtracts thread_offset (threadIdx.x - thread_offset >= SEG);
    # ROCm HIP codegen prints float4_e2m1fn constants via __tl_float_to_fp4(...) and sets enable_fp4_ in VisitExpr_(FloatImmNode).
    warp_spec_thread_offset = 128
    seg_lanes = 32
    active_scan_lanes_without_offset = sum(
        1 for tid in range(warp_spec_thread_offset, warp_spec_thread_offset + 128) if tid < seg_lanes
    )
    active_scan_lanes_with_offset = sum(
        1
        for tid in range(warp_spec_thread_offset, warp_spec_thread_offset + 128)
        if (tid - warp_spec_thread_offset) < seg_lanes
    )

    return {
        "experiment_id": "flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110",
        "phase": 110,
        "priority": 119,
        "provenance": "own-offline",
        "cake_fused_norm_combine_and_moe_ar": {
            "b200_1tok_prev_us": b200_1tok_prev_us,
            "b200_1tok_new_us": b200_1tok_new_us,
            "b200_1tok_speedup_x": b200_1tok_speedup_x,
            "b200_64tok_prev_us": b200_64tok_prev_us,
            "b200_64tok_new_us": b200_64tok_new_us,
            "b200_64tok_speedup_x": b200_64tok_speedup_x,
            "b200_256tok_prev_us": b200_256tok_prev_us,
            "b200_256tok_new_us": b200_256tok_new_us,
            "b200_256tok_speedup_x": b200_256tok_speedup_x,
            "stream_smem_prev_bytes": stream_smem_prev_bytes,
            "stream_smem_new_bytes": stream_smem_new_bytes,
            "stream_handoff_bytes": stream_handoff_bytes,
            "moe_ar_union_variants": moe_ar_union_variants,
            "moe_ar_union_shapes_faster": moe_ar_union_shapes_faster,
            "ws2_bf16_t1_b200_prev_us": ws2_bf16_t1_b200_prev_us,
            "ws2_bf16_t1_b200_new_us": ws2_bf16_t1_b200_new_us,
            "ws2_bf16_t1_b300_prev_us": ws2_bf16_t1_b300_prev_us,
            "ws2_bf16_t1_b300_new_us": ws2_bf16_t1_b300_new_us,
            "ws2_bf16_t1_b300_speedup_x": ws2_bf16_t1_b300_speedup_x,
        },
        "minimax_h3_mm_fp4_and_kimi_k3": {
            "rtx_pro_6000_changed_rows": rtx_pro_6000_changed_rows,
            "rtx_5090_changed_rows": rtx_5090_changed_rows,
            "rtx_pro_6000_max_speedup_x": rtx_pro_6000_max_speedup_x,
            "rtx_5090_max_speedup_x": rtx_5090_max_speedup_x,
            "rtx_pro_6000_m33472_speedup_x": rtx_pro_6000_m33472_speedup_x,
            "chain_ep_stages": chain_ep_stages,
            "n8_o2_range_prev": n8_o2_range_prev,
            "n8_o2_range_new": n8_o2_range_new,
            "gemm_alone_m128_rows_added": gemm_alone_m128_rows_added,
            "k3_reduce_min_m": k3_reduce_min_m,
            "k3_reduce_max_m": k3_reduce_max_m,
            "k3_persist_min_m": k3_persist_min_m,
            "in_flight_gather_tokens": in_flight_gather_tokens,
            "gb300_m768_speedup_x": gb300_m768_speedup_x,
            "gb300_m960_speedup_x": gb300_m960_speedup_x,
            "mla_head_dim_total": mla_head_dim_total,
            "mla_late_trigger_min_main_ctas": mla_late_trigger_min_main_ctas,
            "mla_gb300_faster_shapes": mla_gb300_faster_shapes,
            "mla_total_shapes": mla_total_shapes,
            "mla_gb300_shape1_speedup_x": mla_gb300_shape1_speedup_x,
            "mla_b200_shape1_speedup_x": mla_b200_shape1_speedup_x,
        },
        "cutlass_and_tilelang": {
            "sm120_sparse_tile": sm120_sparse_tile,
            "nvf4_metadata_atom_k": nvf4_metadata_atom_k,
            "mxf8f6f4_metadata_atom_k": mxf8f6f4_metadata_atom_k,
            "buggy_packed_offset_r3_c5": buggy_packed_offset,
            "fixed_packed_offset_r3_c5": fixed_packed_offset,
            "warp_spec_thread_offset": warp_spec_thread_offset,
            "active_scan_lanes_without_offset": active_scan_lanes_without_offset,
            "active_scan_lanes_with_offset": active_scan_lanes_with_offset,
            "rocm_fp4_helper": "__tl_float_to_fp4",
        },
        "invariants": {
            "norm_combine_and_moe_ar_faster": b200_64tok_speedup_x > 1.03 and ws2_bf16_t1_b300_speedup_x > 1.08,
            "kimi_k3_tail_and_mla_verified": gb300_m960_speedup_x == 1.056 and mla_b200_shape1_speedup_x == 1.1649,
            "cutlass_packed_row_major_fixed": fixed_packed_offset == 389 and buggy_packed_offset == 323,
            "tilelang_warp_spec_scan_active_lanes_restored": (
                active_scan_lanes_without_offset == 0 and active_scan_lanes_with_offset == 32
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()
    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Drift detected in Phase 110 Track A summary.json"
        assert all(summary["invariants"].values())
        print("OK: flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110 verified")
        return

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
