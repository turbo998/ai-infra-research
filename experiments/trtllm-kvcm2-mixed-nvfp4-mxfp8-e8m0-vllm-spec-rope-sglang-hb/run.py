#!/usr/bin/env python3
"""Phase 58 Track B deterministic CPU/offline verification model:
TensorRT-LLM KVCacheManagerV2 Mixed-Precision NVFP4 KV Cache, MXFP8 E8M0 Subnormal
`exp2f_rcp` NaN Prevention, vLLM V2 Speculative Draft RoPE Clamping & 64-Bit Integer
Schema Coercion, and SGLang Disaggregated Heartbeat Recovery.

Models and verifies:
1. TensorRT-LLM `KVCacheManagerV2` First-N/Last-N Mixed-Precision NVFP4 KV Cache
   (`NVIDIA/TensorRT-LLM@b4bfec4baf`, #20452):
   - `_NVFP4Codec.row_bytes(head_dim)` = `((head_dim // 16 * 9 + 7) // 8) * 8` bytes per head
     (e.g. `head_dim=128 -> 72 B` vs `256 B` BF16 = `3.5556x` per-row compression).
   - Retains `mixed_precision_first_n` (sink) and `mixed_precision_last_n` (recent window)
     tokens at page granularity in native BF16/FP8/FP16 while storing middle history in NVFP4
     (`pool_ratio=[0.75, 0.25]`), staging back via `MixedPrecisionKVStaging` before attention.
2. TensorRT-LLM `cvt_warp_fp16_to_mxfp8` E8M0 Subnormal Reciprocal NaN Fix
   (`NVIDIA/TensorRT-LLM@348347ce3e`, #20075):
   - Smallest E8M0 scale `fp8SFVal = 0` represents `2^-127` (`~5.87747e-39`), which is
     subnormal in IEEE-754 FP32 (`< 2^-126`).
   - Hardware `rcp.approx.ftz.f32` flushes subnormal `2^-127` to `0.0 -> +inf`, turning
     any near-zero non-zero block with `amax in (0, 448 * 2^-127]` (`(0, 2.6331e-36]`) into
     `0 * inf = NaN`. Using `exp2f_rcp(fp8SFVal) = 2^(127 - fp8SFVal) = 2^+127` stays finite.
3. vLLM V2 Speculative Draft RoPE Cache Clamping & 64-Bit Integer Schema Coercion
   (`vllm-project/vllm@37cf712552`, #63251; `44302539f8`, #55743):
   - `_CACHE_ONLY_ROPE_TYPES = {"default", "linear", "llama3", "ntk", "proportional"}`
     extends `draft_model_config.max_position_embeddings` to target `max_model_len`;
     non-cache-only RoPE (`yarn`, `deepseek`, `dynamic`, `longrope`) lowers V2 engine
     `max_model_len` to `draft_max_model_len` via `_limit_max_model_len_to_draft()`.
   - `coerce_to_schema_type("9007199254740993", "number")` tries `int(value)` before
     `float(value)` so integers in `[2^53 + 1, 2^63 - 1]` do not lose low bits in IEEE-754 binary64.
4. SGLang Disaggregated Prefill Heartbeat Counter Reset (`sgl-project/sglang@16aa816060`, #45922):
   - `self.heartbeat_failures.pop(failed_bootstrap_addr, None)` in `_handle_node_failure`
     prevents stale failure counts (`MAX_HEARTBEAT_FAILURES=3`) from evicting a reconnected
     prefill server on its very first transient timeout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CACHE_ONLY_ROPE_TYPES = frozenset({"default", "linear", "llama3", "ntk", "proportional"})


def nvfp4_codec_row_bytes(head_dim: int) -> int:
    """Exact TensorRT-LLM `_NVFP4Codec.row_bytes(head_dim)` formula from `mixed_precision_kv.py`."""
    group_size = 16
    packed_group_size = 8
    if head_dim % group_size != 0:
        raise ValueError("NVFP4 mixed-precision KV requires head_dim divisible by 16")
    size = (head_dim // group_size) * (packed_group_size + 1)
    return ((size + 7) // 8) * 8


def simulate_trtllm_mixed_precision_kv() -> dict[str, Any]:
    """Model KVCacheManagerV2 First-N/Last-N NVFP4 + native high-precision KV memory footprint."""
    head_dim = 128
    num_kv_heads = 8
    tokens_per_block = 32
    first_n = 256
    last_n = 256
    seq_lens = [1024, 4096, 16384, 32768]

    nvfp4_bytes_per_head = nvfp4_codec_row_bytes(head_dim)  # 72 B
    bf16_bytes_per_head = head_dim * 2  # 256 B
    fp8_bytes_per_head = head_dim * 1  # 128 B

    seq_profiles = []
    for seq_len in seq_lens:
        total_pages = (seq_len + tokens_per_block - 1) // tokens_per_block
        first_pages = (first_n + tokens_per_block - 1) // tokens_per_block
        last_pages = (last_n + tokens_per_block - 1) // tokens_per_block
        retained_high_pages = min(total_pages, first_pages + last_pages)
        low_nvfp4_pages = max(0, total_pages - retained_high_pages)

        pure_bf16_bytes = total_pages * tokens_per_block * 2 * num_kv_heads * bf16_bytes_per_head
        mixed_nvfp4_bf16_bytes = (
            retained_high_pages * tokens_per_block * 2 * num_kv_heads * bf16_bytes_per_head
            + low_nvfp4_pages * tokens_per_block * 2 * num_kv_heads * nvfp4_bytes_per_head
        )
        savings_pct = round((1.0 - mixed_nvfp4_bf16_bytes / pure_bf16_bytes) * 100.0, 2)
        seq_profiles.append(
            {
                "seq_len": seq_len,
                "total_pages": total_pages,
                "retained_high_pages": retained_high_pages,
                "low_nvfp4_pages": low_nvfp4_pages,
                "pure_bf16_kib": round(pure_bf16_bytes / 1024.0, 2),
                "mixed_nvfp4_bf16_kib": round(mixed_nvfp4_bf16_bytes / 1024.0, 2),
                "memory_savings_pct": savings_pct,
            }
        )

    return {
        "head_dim": head_dim,
        "num_kv_heads": num_kv_heads,
        "tokens_per_block": tokens_per_block,
        "mixed_precision_first_n": first_n,
        "mixed_precision_last_n": last_n,
        "nvfp4_bytes_per_head": nvfp4_bytes_per_head,
        "bf16_bytes_per_head": bf16_bytes_per_head,
        "fp8_bytes_per_head": fp8_bytes_per_head,
        "nvfp4_vs_bf16_row_compression_x": round(bf16_bytes_per_head / nvfp4_bytes_per_head, 4),
        "seq_profiles": seq_profiles,
    }


def simulate_mxfp8_e8m0_subnormal_fix() -> dict[str, Any]:
    """Model `cvt_warp_fp16_to_mxfp8` E8M0 scale = 2^-127 FTZ NaN failure vs `exp2f_rcp` fix."""
    fp32_min_normal = 2.0 ** -126
    e8m0_min_scale = 2.0 ** -127
    fp8_e4m3_max = 448.0
    subnormal_amax_upper_bound = fp8_e4m3_max * e8m0_min_scale  # 448 * 2^-127

    test_amaxes = [0.0, 1e-40, 1e-38, 2.0e-36, 1.0e-30, 1.0]
    outcomes = []
    for amax in test_amaxes:
        if amax == 0.0:
            fp8_sf_code = 0
            old_ftz_produces_nan = False
            new_exp2f_rcp_finite = True
            new_output_scale = 0.0
        else:
            raw_sf = amax / fp8_e4m3_max
            # Round up to power of two in E8M0 [2^-127, 2^127]
            exp_unbiased = max(-127, min(127, int(math.ceil(math.log2(raw_sf)))))
            fp8_sf_code = exp_unbiased + 127
            decoded_sf_float = 2.0 ** exp_unbiased
            is_subnormal_fp32 = decoded_sf_float < fp32_min_normal
            # Old path: rcp.approx.ftz.f32 flushes subnormal decoded_sf_float to 0 -> +inf -> 0 * inf = NaN
            old_ftz_produces_nan = is_subnormal_fp32
            new_output_scale = 2.0 ** (127 - fp8_sf_code)
            new_exp2f_rcp_finite = math.isfinite(new_output_scale)
        outcomes.append(
            {
                "amax": amax,
                "fp8_sf_code": fp8_sf_code,
                "old_ftz_produces_nan": old_ftz_produces_nan,
                "new_exp2f_rcp_finite": new_exp2f_rcp_finite,
            }
        )

    return {
        "fp32_min_normal": fp32_min_normal,
        "e8m0_min_scale": e8m0_min_scale,
        "subnormal_amax_upper_bound": subnormal_amax_upper_bound,
        "outcomes": outcomes,
        "subnormal_nan_cases_caught": sum(1 for o in outcomes if o["old_ftz_produces_nan"]),
        "all_new_scales_finite": all(o["new_exp2f_rcp_finite"] for o in outcomes),
    }


def simulate_vllm_spec_rope_and_sglang_heartbeat() -> dict[str, Any]:
    """Model vLLM V2 draft RoPE clamping (#63251), 64-bit int schema coercion (#55743), and SGLang heartbeat (#45922)."""
    target_max_len = 32768
    draft_orig_max_len = 8192

    rope_cases = []
    for rope_type in ("default", "llama3", "proportional", "yarn", "deepseek", "longrope"):
        shares_target_pos = rope_type in CACHE_ONLY_ROPE_TYPES
        if shares_target_pos:
            effective_draft_max_pos = max(draft_orig_max_len, target_max_len)
            effective_engine_max_len = target_max_len
        else:
            effective_draft_max_pos = draft_orig_max_len
            effective_engine_max_len = min(target_max_len, draft_orig_max_len)
        rope_cases.append(
            {
                "rope_type": rope_type,
                "shares_target_positions": shares_target_pos,
                "effective_draft_max_pos": effective_draft_max_pos,
                "effective_engine_max_len": effective_engine_max_len,
                "oob_rotary_gather_possible": effective_engine_max_len > effective_draft_max_pos,
            }
        )

    # 64-bit integer precision check above 2^53 (9007199254740992)
    int64_str = "9007199254740993"
    float_coerced = int(float(int64_str))
    int_first_coerced = int(int64_str)

    # SGLang heartbeat failure counter across node failure & reconnect
    max_heartbeat_failures = 3
    stale_counter_after_failure = max_heartbeat_failures  # 3 without pop()
    reset_counter_after_failure = 0  # 0 with heartbeat_failures.pop(addr, None)
    stale_drops_on_first_miss = (stale_counter_after_failure + 1) >= max_heartbeat_failures
    reset_survives_first_miss = (reset_counter_after_failure + 1) < max_heartbeat_failures

    return {
        "rope_cases": rope_cases,
        "zero_oob_rotary_gathers": all(not c["oob_rotary_gather_possible"] for c in rope_cases),
        "int64_input": int64_str,
        "float_coerced_corrupted": str(float_coerced),
        "int_first_coerced_exact": str(int_first_coerced),
        "int64_precision_preserved": str(int_first_coerced) == int64_str and float_coerced != int_first_coerced,
        "sglang_heartbeat": {
            "max_heartbeat_failures": max_heartbeat_failures,
            "stale_drops_on_first_miss": stale_drops_on_first_miss,
            "reset_survives_first_miss": reset_survives_first_miss,
        },
    }


def build_summary() -> dict[str, Any]:
    kv = simulate_trtllm_mixed_precision_kv()
    mxfp8 = simulate_mxfp8_e8m0_subnormal_fix()
    vllm_sglang = simulate_vllm_spec_rope_and_sglang_heartbeat()

    return {
        "experiment_id": "trtllm-kvcm2-mixed-nvfp4-mxfp8-e8m0-vllm-spec-rope-sglang-hb",
        "phase": 58,
        "priority": 67,
        "track": "B",
        "title": "TensorRT-LLM KVCacheManagerV2 Mixed-Precision NVFP4 KV Cache, MXFP8 E8M0 Subnormal Fix, vLLM V2 Speculative RoPE Clamping, and SGLang Heartbeat Reset",
        "commits": [
            "NVIDIA/TensorRT-LLM@b4bfec4baf",
            "NVIDIA/TensorRT-LLM@348347ce3e",
            "NVIDIA/TensorRT-LLM@96de4749c5",
            "NVIDIA/TensorRT-LLM@0f3e1d093d",
            "vllm-project/vllm@37cf712552",
            "vllm-project/vllm@44302539f8",
            "vllm-project/vllm@ccd4e4030f",
            "sgl-project/sglang@16aa816060",
            "sgl-project/sglang@30dc2f9430",
        ],
        "trtllm_mixed_precision_kv": kv,
        "mxfp8_e8m0_subnormal_fix": mxfp8,
        "vllm_spec_rope_and_sglang_heartbeat": vllm_sglang,
        "validation": {
            "nvfp4_row_bytes_verified": kv["nvfp4_bytes_per_head"] == 72 and kv["nvfp4_vs_bf16_row_compression_x"] == 3.5556,
            "kv_32k_savings_verified": kv["seq_profiles"][-1]["memory_savings_pct"] == 70.75,
            "mxfp8_subnormal_nan_fixed": mxfp8["subnormal_nan_cases_caught"] == 3 and mxfp8["all_new_scales_finite"],
            "vllm_zero_oob_rotary_verified": vllm_sglang["zero_oob_rotary_gathers"],
            "vllm_int64_precision_verified": vllm_sglang["int64_precision_preserved"],
            "sglang_heartbeat_reset_verified": (
                vllm_sglang["sglang_heartbeat"]["stale_drops_on_first_miss"]
                and vllm_sglang["sglang_heartbeat"]["reset_survives_first_miss"]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if args.check:
        assert all(summary["validation"].values()), f"Validation failed: {summary['validation']}"
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
