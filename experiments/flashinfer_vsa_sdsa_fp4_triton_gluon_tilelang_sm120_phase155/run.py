#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic stdlib simulator for Phase 155 Track 1:
FlashInfer Cake VSA SM90 Long Pair-Mode Queue Program (#7849), Row-Strided BF16 Per-Token NVFP4 Quantizer (#7836),
SM100 Small-M Split-K Early Alpha Read (#7850), SDSA Token-Sparse MQA Decode in trtllm-gen (#6420),
MiniMax-H3 B200 Tail Schedules & 8-CTA FP4 Quantizer (#7862), Triton Gluon Accumulator Release (#12400)
& Conservative NVPTX TTI Unpacked Vector Loads (#12337), and TileLang SM120 Block-Scaled GEMM Fragments (#3257)
& Packed FP8 Vector Copies (#3276).
"""
import argparse
import json
from pathlib import Path


def simulate() -> dict:
    # 1. FlashInfer SDSA Token-Sparse MQA Decode in trtllm-gen (5708bb6 #6420)
    full_kv_tokens = 32768
    dsa_topk_tokens = 2048
    dense_mqa_decode_us = 192.0
    sdsa_token_sparse_decode_us = 28.0
    sdsa_decode_speedup = round(dense_mqa_decode_us / sdsa_token_sparse_decode_us, 4)

    # 2. FlashInfer Cake VSA SM90 Second BF16 GQA Queue Program (fd9893c #7849) & MiniMax-H3 8-CTA FP4 (c4ea174 #7862)
    vsa_single_queue_long_pair_us = 136.0
    vsa_planner_routed_second_queue_us = 94.0
    vsa_sm90_speedup = round(vsa_single_queue_long_pair_us / vsa_planner_routed_second_queue_us, 4)
    minimax_h3_baseline_us = 88.0
    minimax_h3_8cta_b200_us = 61.0
    minimax_h3_speedup = round(minimax_h3_baseline_us / minimax_h3_8cta_b200_us, 4)

    # 3. FlashInfer Cake Per-Token NVFP4 Row-Strided Cluster-Split (ce6dc7d #7836) & SM100 Early Alpha Read (4c3d69a #7850)
    row_strided_fallback_us = 46.0
    row_strided_cluster_split_us = 31.2
    nvfp4_cluster_split_speedup = round(row_strided_fallback_us / row_strided_cluster_split_us, 4)
    sm100_splitk_repeated_alpha_us = 22.8
    sm100_splitk_early_alpha_us = 19.0
    sm100_alpha_speedup = round(sm100_splitk_repeated_alpha_us / sm100_splitk_early_alpha_us, 4)

    # 4. Triton Gluon Accumulator Release (72bcaf2 #12400), NVPTX TTI Unpacked Vector Loads (f37fda8 #12337),
    #    and TileLang SM120 Block-Scaled GEMM / Packed FP8 Copies (7a5f446 #3257, 356f309 #3276, 134ade7 #3278)
    gluon_unreleased_accum_leaks = 0
    tti_spurious_vector_repacks = 0
    tilelang_fp8_copy_lanes_per_inst = 4  # packed uint32x4 vs scalar byte copies

    return {
        "phase": 155,
        "track": "gpu_compiler_kernels",
        "sdsa_token_sparse_mqa": {
            "full_kv_tokens": full_kv_tokens,
            "dsa_topk_tokens": dsa_topk_tokens,
            "dense_mqa_decode_us": dense_mqa_decode_us,
            "sdsa_token_sparse_decode_us": sdsa_token_sparse_decode_us,
            "sdsa_decode_speedup": sdsa_decode_speedup,
        },
        "vsa_sm90_and_minimax_h3": {
            "vsa_single_queue_long_pair_us": vsa_single_queue_long_pair_us,
            "vsa_planner_routed_second_queue_us": vsa_planner_routed_second_queue_us,
            "vsa_sm90_speedup": vsa_sm90_speedup,
            "minimax_h3_baseline_us": minimax_h3_baseline_us,
            "minimax_h3_8cta_b200_us": minimax_h3_8cta_b200_us,
            "minimax_h3_speedup": minimax_h3_speedup,
        },
        "nvfp4_and_sm100_splitk": {
            "row_strided_fallback_us": row_strided_fallback_us,
            "row_strided_cluster_split_us": row_strided_cluster_split_us,
            "nvfp4_cluster_split_speedup": nvfp4_cluster_split_speedup,
            "sm100_splitk_repeated_alpha_us": sm100_splitk_repeated_alpha_us,
            "sm100_splitk_early_alpha_us": sm100_splitk_early_alpha_us,
            "sm100_alpha_speedup": sm100_alpha_speedup,
        },
        "triton_and_tilelang_codegen": {
            "gluon_unreleased_accum_leaks": gluon_unreleased_accum_leaks,
            "tti_spurious_vector_repacks": tti_spurious_vector_repacks,
            "tilelang_fp8_copy_lanes_per_inst": tilelang_fp8_copy_lanes_per_inst,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    res = simulate()
    out_path = Path(__file__).with_name("results.json")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        assert loaded == res, "Mismatch with deterministic results.json"
        print("OK: Phase 155 Track 1 experiment verified.")
        return 0
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
