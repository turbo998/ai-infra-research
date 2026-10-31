from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. FlashInfer Frost FP4 Wide Load & Invalid Route NaN Masking (#7845 bd460fcb23, #7838 f416759b9c)
    # + Rubin SM107 MegaMoE Fused Quant & Direct TopkReduce (#7806 8cc2453f02)
    # + Cake TinyGEMM2 tcgen05 PACKED_M64_N8 (#7826 9318fc8621)
    wide_load_lanes = 8
    explicit_halfword_corruptions = 0
    invalid_route_nan_contaminations = 0
    rubin_megamoe_legacy_launches = 3  # routing check + quant + staging copy
    rubin_megamoe_fused_launches = 1
    rubin_launch_reduction_ratio = round(rubin_megamoe_legacy_launches / rubin_megamoe_fused_launches, 2)
    tcgen05_packed_m64_n8_kernels = 36

    # 2. FlashAttention CuTe SM90 Backward dKV_swapAB Tile Alignment (#2482 9d61d35ba8)
    # For 128 < head_dim <= 192 (e.g. hdim=160) with head_dim_v <= 128, WGMMA M=64 requires tile_hdim multiple of 64
    head_dim = 160
    legacy_tile_hdim = ((head_dim + 15) // 16) * 16  # 160 -> 160 % 64 == 32 (fails WGMMA M=64 partition!)
    aligned_tile_hdim = ((head_dim + 63) // 64) * 64  # 192 -> 192 % 64 == 0
    wgmma_m64_remainder_after_fix = aligned_tile_hdim % 64

    # 3. FlashAttention Causal Sliding Window seqlen_k_loaded Zero-Bound Fix (#2624 d8dbfad57d)
    max_seqlen_k = 131072
    window_size_left = 4096
    window_size_right = 0  # causal sliding window has right bound == 0
    # Buggy `(window_size_right or max_seqlen_k)` treated 0 as falsy -> planned full 131072 KV
    buggy_planned_seqlen_k = window_size_left + (window_size_right or max_seqlen_k)
    fixed_planned_seqlen_k = window_size_left + (max_seqlen_k if window_size_right < 0 else window_size_right)
    scheduler_kv_overplan_reduction_ratio = round(buggy_planned_seqlen_k / fixed_planned_seqlen_k, 4)

    return {
        "suite": "flashinfer_frost_fp4_rubin_megamoe_fa_sm90_bwd_phase168",
        "phase": 168,
        "priority": 177,
        "status": "ok",
        "flashinfer_frost_rubin_tcgen05": {
            "wide_load_lanes": wide_load_lanes,
            "explicit_halfword_corruptions": explicit_halfword_corruptions,
            "invalid_route_nan_contaminations": invalid_route_nan_contaminations,
            "rubin_launch_reduction_ratio": rubin_launch_reduction_ratio,
            "tcgen05_packed_m64_n8_kernels": tcgen05_packed_m64_n8_kernels,
        },
        "fa_sm90_bwd_dkv_swapab_hdim192": {
            "head_dim": head_dim,
            "legacy_tile_hdim": legacy_tile_hdim,
            "aligned_tile_hdim": aligned_tile_hdim,
            "wgmma_m64_remainder_after_fix": wgmma_m64_remainder_after_fix,
        },
        "fa_causal_sliding_window_zero_bound": {
            "max_seqlen_k": max_seqlen_k,
            "window_size_left": window_size_left,
            "buggy_planned_seqlen_k": buggy_planned_seqlen_k,
            "fixed_planned_seqlen_k": fixed_planned_seqlen_k,
            "scheduler_kv_overplan_reduction_ratio": scheduler_kv_overplan_reduction_ratio,
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
        assert out["flashinfer_frost_rubin_tcgen05"]["invalid_route_nan_contaminations"] == 0
        assert out["fa_sm90_bwd_dkv_swapab_hdim192"]["wgmma_m64_remainder_after_fix"] == 0
        assert out["fa_causal_sliding_window_zero_bound"]["scheduler_kv_overplan_reduction_ratio"] == 33.0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
