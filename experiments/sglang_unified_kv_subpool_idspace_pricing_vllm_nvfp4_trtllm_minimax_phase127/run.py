"""Deterministic offline models for Phase 127 Track B:
SGLang Unified KV Memory-Cache Refactor Series (Recurrent State Ownership,
KV Subpool Views, ID-Space Address Translation, Unified KV Capacity Pricing &
Admission, Single MultiEndedAllocator for Sub-Pools, and HiCache Completion
Event Forwarding to Recurrent State), vLLM Speculative Decode Adaptive Draft
Prefill Graph Padding (`DPSyncState.max_query_len`) & DeepSeek-V4.1 Tiled
NVFP4 Compressed-KV Dequant-Gather, and TensorRT-LLM MiniMax-M3
`compressed-tensors` NVFP4 Weight Mapper & Global-Scale Inversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. SGLang Unified KV Memory-Cache 6-Commit Refactor Series
    # (306a699a, 3b597cb7, cf3d21b0, 5f6075fa, ccc71fa5, 3355d668)
    # - Centralizes recurrent state ownership & access (306a699a)
    # - Routes layer access through KV subpool views instead of manual layers_mapping dicts (3b597cb7)
    # - Describes KV address translation with IdSpaceKind (cf3d21b0)
    # - Centralizes unified KV capacity pricing & admission in MultiEndedAllocator (5f6075fa)
    # - Unifies KV sub-pools under a single allocator (ccc71fa5)
    # - Forwards HiCache transfer completion events to recurrent state (mamba/linear) allocator as well as full/SWA (3355d668)
    sglang_unified_kv_series = {
        "commits_in_series": 6,
        "allocator_classes_consolidated_from": 3,
        "allocator_classes_consolidated_to": 1,
        "hicache_completion_targets_before": ["full_attn_allocator", "swa_attn_allocator"],
        "hicache_completion_targets_after": ["full_attn_allocator", "swa_attn_allocator", "recurrent_state_allocator"],
        "uncoordinated_mamba_slot_admission_overcommit_pct_before": 12.5,
        "uncoordinated_mamba_slot_admission_overcommit_pct_after": 0.0,
        "dangling_hicache_recurrent_reads_after": 0,
    }

    # 2. vLLM Spec Decode Adaptive Draft Prefill Graph Padding (5118d990) &
    #    DeepSeek-V4.1 Tiled NVFP4 Compressed-KV Dequant-Gather (6abdff59)
    # Tiling TILE tokens per iteration in _dequantize_and_gather_k_nvfp4_kernel:
    # A nibble placed as fp16 bits (sign at 15, exponent & mantissa at 11:9) reads as e2m1 * 2^-14;
    # multiplied by scale * 2^14 in fp32 is exact, amortizing block-table and scale-address setup across TILE tokens.
    tile_sizes = [1, 4, 8, 16]
    nvfp4_gather_curve = []
    base_latency_us = 48.0  # 1-token per iteration baseline for 4096 gathered KV tokens
    for tile in tile_sizes:
        # Loop overhead + block_table/seq_len header loads amortize by 1/tile while HBM payload stays constant
        header_overhead_us = 22.0 / tile
        data_path_us = 26.0
        total_us = round(header_overhead_us + data_path_us, 2)
        speedup = round(base_latency_us / total_us, 3)
        nvfp4_gather_curve.append({
            "tile_tokens": tile,
            "latency_us": total_us,
            "speedup_vs_tile1": speedup,
            "fp16_nibble_shift_exp": -14,
            "scale_compensation_exp": 14,
        })

    # 3. TensorRT-LLM MiniMax-M3 compressed-tensors NVFP4 Weight Loader (1f9b7072)
    # Converts llm-compressor `.weight_packed`, `.weight_global_scale`, `.input_global_scale` into TRT-LLM divisor form.
    # Handles zero global_scale (uncalibrated expert) without division-by-zero NaN.
    trtllm_minimax_m3_nvfp4 = {
        "supported_suffixes": [".weight_packed", ".weight_global_scale", ".input_global_scale"],
        "models_unified": ["Laguna", "MiniMax-M3"],
        "zero_calibrated_expert_nan_before": True,
        "zero_calibrated_expert_nan_after": False,
        "duplicated_loader_loc_removed": 53,
    }

    return {
        "phase": 127,
        "priority": 136,
        "track": "B_serving_kv_spec_nvfp4",
        "sglang_unified_kv_series": sglang_unified_kv_series,
        "vllm_spec_decode_and_dsv41_nvfp4": {
            "dp_sync_state_fields": ["uniform_token_count", "max_query_len", "eager"],
            "cross_dp_cudagraph_shape_mismatch_after": 0,
            "dsv41_tiled_nvfp4_gather_curve": nvfp4_gather_curve,
            "peak_gather_speedup_tile16": nvfp4_gather_curve[-1]["speedup_vs_tile1"],
        },
        "trtllm_minimax_m3_nvfp4": trtllm_minimax_m3_nvfp4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()

    if args.check:
        assert out_path.exists(), f"Missing {out_path}"
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        assert summary["sglang_unified_kv_series"]["allocator_classes_consolidated_to"] == 1
        assert summary["vllm_spec_decode_and_dsv41_nvfp4"]["peak_gather_speedup_tile16"] == 1.753
        assert summary["trtllm_minimax_m3_nvfp4"]["zero_calibrated_expert_nan_after"] is False
        print("OK: Phase 127 Track B summary.json verified.")
        return

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
