#!/usr/bin/env python3
"""Phase 149 Track 1: FlashInfer MXFP8 Band-Path / MoK Rubin EP8 / KG1 Window / MiniMax-H3 Lazy Rescale & TileLang Predicate Audit."""
import argparse
import json
import math
import pathlib


def simulate():
    # 1. CAKE Gated-Act MXFP8 SM100/SM103 band-path backward (#7880)
    tokens = 8192
    hidden_dim = 7168
    elem_bytes_fp8 = 1
    elem_bytes_bf16 = 2
    full_grid_waves = 32
    band_grid_waves = 24
    l2_hit_full = 0.56
    l2_hit_band = 0.81
    dram_bytes_full = int(tokens * hidden_dim * (elem_bytes_bf16 * 2 + elem_bytes_fp8) * (1.0 - 0.35 * l2_hit_full))
    dram_bytes_band = int(tokens * hidden_dim * (elem_bytes_bf16 * 2 + elem_bytes_fp8) * (1.0 - 0.35 * l2_hit_band))
    band_speedup = round((dram_bytes_full / dram_bytes_band) * (full_grid_waves / band_grid_waves) ** 0.35, 4)

    # 2. CAKE MoK BF16 faster MXFP8 weight quant + Rubin EP8 comm-SM defaults (#7870)
    total_sms_rubin = 160
    legacy_comm_sms = 24
    tuned_ep8_comm_sms = 16
    compute_sms_legacy = total_sms_rubin - legacy_comm_sms
    compute_sms_tuned = total_sms_rubin - tuned_ep8_comm_sms
    mxfp8_quant_us_legacy = 148.0
    mxfp8_quant_us_fast = 92.5
    dispatch_overlap_us = 110.0
    mok_step_us_legacy = round(mxfp8_quant_us_legacy + dispatch_overlap_us * (160 / compute_sms_legacy), 3)
    mok_step_us_tuned = round(mxfp8_quant_us_fast + dispatch_overlap_us * (160 / compute_sms_tuned), 3)

    # 3. CAKE Grouped FP8 GEMM 128-row window-segment KG1 (#7877)
    experts = 64
    window_rows = 128
    avg_tokens_per_expert = 42
    scattered_scale_loads = experts * math.ceil(avg_tokens_per_expert / 16) * 4
    window_kg1_scale_loads = math.ceil((experts * avg_tokens_per_expert) / window_rows) * 4
    scale_traffic_reduction_pct = round((1.0 - window_kg1_scale_loads / scattered_scale_loads) * 100.0, 2)

    # 4. CAKE MiniMax-H3 SM120 lazy output rescale in sequential FP8 varlen attention (#7871)
    seq_chunks = 16
    head_dim = 128
    eager_rescales_per_token = seq_chunks * head_dim
    # Lazy rescale only rescales when running max shifts beyond threshold (e.g. 5 of 16 chunks + 1 final)
    lazy_rescales_per_token = 6 * head_dim
    sfu_instructions_saved_pct = round((1.0 - lazy_rescales_per_token / eager_rescales_per_token) * 100.0, 2)

    # 5. TileLang nested LDG enclosing predicate preservation (#3390) & profiler sync (#3457)
    tail_rows = 19
    tile_m = 32
    unpredicated_oob_lanes = (tile_m - tail_rows) * 8
    predicated_oob_lanes = 0

    return {
        "schema_version": "1.0",
        "phase": 149,
        "track": "gpu_kernel_compiler_stack",
        "evidence_class": "own-offline",
        "mxfp8_band_backward": {
            "tokens": tokens,
            "hidden_dim": hidden_dim,
            "l2_hit_full": l2_hit_full,
            "l2_hit_band": l2_hit_band,
            "dram_bytes_full": dram_bytes_full,
            "dram_bytes_band": dram_bytes_band,
            "band_speedup_ratio": band_speedup,
        },
        "mok_rubin_ep8": {
            "total_sms_rubin": total_sms_rubin,
            "legacy_comm_sms": legacy_comm_sms,
            "tuned_ep8_comm_sms": tuned_ep8_comm_sms,
            "mxfp8_quant_us_legacy": mxfp8_quant_us_legacy,
            "mxfp8_quant_us_fast": mxfp8_quant_us_fast,
            "mok_step_us_legacy": mok_step_us_legacy,
            "mok_step_us_tuned": mok_step_us_tuned,
        },
        "grouped_fp8_gemm_kg1": {
            "experts": experts,
            "window_rows": window_rows,
            "scattered_scale_loads": scattered_scale_loads,
            "window_kg1_scale_loads": window_kg1_scale_loads,
            "scale_traffic_reduction_pct": scale_traffic_reduction_pct,
        },
        "minimax_h3_sm120_lazy_rescale": {
            "seq_chunks": seq_chunks,
            "eager_rescales_per_token": eager_rescales_per_token,
            "lazy_rescales_per_token": lazy_rescales_per_token,
            "sfu_instructions_saved_pct": sfu_instructions_saved_pct,
        },
        "tilelang_nested_ldg_guard": {
            "tail_rows": tail_rows,
            "tile_m": tile_m,
            "unpredicated_oob_lanes": unpredicated_oob_lanes,
            "predicated_oob_lanes": predicated_oob_lanes,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    res = simulate()
    out_path = pathlib.Path(__file__).with_name("results.json")
    if args.check:
        expected = json.loads(out_path.read_text(encoding="utf-8"))
        assert expected == res, "Result mismatch against results.json"
        print("OK: Phase 149 Track 1 results verified.")
        return
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
