#!/usr/bin/env python3
"""Deterministic CPU/offline model for Phase 66 Track B:
1. SGLang GGUF MoE Routing/Weighting Kernel (`mul.rn.f32`) & Multi-Warp `tile_y_ds` Indexing Fix (sglang@e6451c1a956d)
   + Cosmos Dreams Host-Allocator Pinned Output Stream Drain Ordering (sglang@194abb8e54a8)
2. TensorRT-LLM MXFP8 E8M0 Subnormal Scale `exp2f_rcp` NaN Prevention & Non-MPI Ray `mapping.world_size` (TensorRT-LLM@348347ce3e93)
3. vLLM ROCm Muse Spark Shared FP32 RoPE Cache (`max_model_len`) & `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (vllm@b1c66e55450d, vllm@d143a9de3915)
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def float32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_float32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def is_fp32_subnormal(val: float) -> bool:
    bits = float32_bits(val)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    return exp == 0 and mant != 0


def model_trtllm_mxfp8_e8m0_subnormal_nan() -> dict[str, Any]:
    """Model TensorRT-LLM cvt_warp_fp16_to_mxfp8 E8M0 subnormal scale reciprocal NaN bug."""
    # E8M0 encodes 2^(u8 - 127) for u8 in [0, 254], and NaN for 255.
    # When u8 == 0, the scale is 2^-127 (~5.877471754e-39), which has IEEE-754 FP32 biased exponent 0
    # and mantissa 0x400000 -> FP32 subnormal!
    # Any non-zero block whose vecMax is in (0, 448 * 2^-127] rounds up to E8M0 raw byte 0 (2^-127).
    smallest_e8m0_raw = 0
    smallest_e8m0_scale = math.ldexp(1.0, smallest_e8m0_raw - 127)  # 2^-127
    subnormal_flag = is_fp32_subnormal(smallest_e8m0_scale)
    max_vecmax_for_e8m0_zero = math.ldexp(448.0, -127)

    # Legacy path: reciprocal_approximate_ftz(SFValue)
    # Because rcp.approx.ftz.f32 flushes subnormal inputs to +0.0, 1.0 / +0.0 = +inf.
    # Multiplying a 0.0 element inside that non-zero block by +inf produces 0.0 * inf = NaN!
    legacy_ftz_input = 0.0 if subnormal_flag else smallest_e8m0_scale
    legacy_output_scale = math.inf if legacy_ftz_input == 0.0 else (1.0 / legacy_ftz_input)
    zero_elem_times_legacy_scale_is_nan = math.isnan(0.0 * legacy_output_scale)

    # Fixed path (TensorRT-LLM@348347ce3e93): exp2f_rcp(fp8SFVal) computes 2^(127 - fp8SFVal) = 2^127 directly
    fixed_output_scale = math.ldexp(1.0, 127 - smallest_e8m0_raw)  # 2^127 = 1.7014118346046923e+38 (finite normal FP32!)
    zero_elem_times_fixed_scale = 0.0 * fixed_output_scale
    nonzero_elem_normalized = max_vecmax_for_e8m0_zero * fixed_output_scale  # 448.0

    return {
        "smallest_e8m0_raw_byte": smallest_e8m0_raw,
        "smallest_e8m0_scale_fp32": smallest_e8m0_scale,
        "smallest_e8m0_scale_fp32_hex": hex(float32_bits(smallest_e8m0_scale)),
        "is_fp32_subnormal": subnormal_flag,
        "vulnerable_amax_upper_bound": max_vecmax_for_e8m0_zero,
        "legacy_rcp_ftz_output_scale": "inf",
        "legacy_zero_lane_product_is_nan": zero_elem_times_legacy_scale_is_nan,
        "fixed_exp2f_rcp_output_scale": fixed_output_scale,
        "fixed_exp2f_rcp_is_finite": math.isfinite(fixed_output_scale),
        "fixed_zero_lane_product": zero_elem_times_fixed_scale,
        "fixed_max_lane_product": nonzero_elem_normalized,
        "ray_non_mpi_world_size_source": "mapping.world_size",
    }


def model_sglang_gguf_moe_and_cosmos_drain() -> dict[str, Any]:
    """Model SGLang GGUF MoE multi-warp tile_y_ds indexing, mul.rn.f32 & Cosmos Dreams stream drain."""
    # In python/sglang/kernels/aot/csrc/quantization/gguf/moe.cuh:
    # When mmq_x > nwarps (e.g., mmq_x = 64, nwarps = 8 -> 8 iterations i = 0, 8, ..., 56),
    # legacy code only loaded tile_y_ds[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby] (only i=0!),
    # leaving rows [nwarps, mmq_x) with uninitialized shared memory scales (7/8 = 87.5% of tile rows!).
    mmq_x = 64
    nwarps = 8
    legacy_populated_rows = nwarps
    fixed_populated_rows = sum(nwarps for _ in range(0, mmq_x, nwarps))
    legacy_uninitialized_row_fraction = round(
        (mmq_x - legacy_populated_rows) / mmq_x, 4
    )

    # In apply_gguf_moe_weights_kernel (gguf_moe_weights.cuh):
    # Negative router id (< 0) zeros the row; valid router id uses explicit `mul.rn.f32`
    # to preserve subnormal products when SM90 compilation enables -use_fast_math / ftz.
    subnormal_val = math.ldexp(1.0, -126) * 0.5  # 2^-127 (FP32 subnormal)
    weight = 0.5  # product = 2^-128 (still subnormal in FP32, since min subnormal is 2^-149)
    ieee_rn_product = subnormal_val * weight
    fast_math_ftz_product = 0.0

    # Cosmos Dreams CausalDecodeQueue.__exit__:
    # Freeing pinned outputs in self.outputs.clear() enqueues host-allocator events on self.stream.
    # Synchronizing before outputs.clear() leaves those events unsynchronized on exit.
    return {
        "gguf_moe_mmq_x": mmq_x,
        "gguf_moe_nwarps": nwarps,
        "legacy_tile_y_ds_populated_rows": legacy_populated_rows,
        "fixed_tile_y_ds_populated_rows": fixed_populated_rows,
        "legacy_uninitialized_row_fraction": legacy_uninitialized_row_fraction,
        "negative_id_zeros_output_row": True,
        "sm90_mul_rn_f32_subnormal_product": ieee_rn_product,
        "sm90_fast_math_ftz_product": fast_math_ftz_product,
        "mul_rn_f32_preserves_subnormal": ieee_rn_product > 0.0 and is_fp32_subnormal(ieee_rn_product),
        "cosmos_dreams_drain_after_outputs_clear": True,
    }


def model_vllm_shared_rope_and_shutdown_grace() -> dict[str, Any]:
    """Model vLLM ROCm Muse Spark shared FP32 RoPE cache savings & EngineCore shutdown timeout."""
    # Muse Spark 1x target model example:
    # 32 target layers, head_dim = 128, config.max_position_embeddings = 131072,
    # deployed vllm_config.model_config.max_model_len = 8192, FP32 cos_sin_cache (4 bytes/elem).
    num_layers = 32
    head_dim = 128
    config_max_pos = 131072
    deployed_max_model_len = 8192
    bytes_per_fp32 = 4

    # Legacy: each layer instantiated its own RotaryEmbedding sized to max(config_max_pos, max_model_len)
    legacy_bytes_per_layer = config_max_pos * head_dim * bytes_per_fp32  # 64 MiB per layer
    legacy_total_bytes = num_layers * legacy_bytes_per_layer  # 2048 MiB (2.0 GiB)

    # Fixed (vllm@b1c66e55450d): target layers share a single get_rope() FP32 instance sized to max_model_len
    fixed_total_bytes = deployed_max_model_len * head_dim * bytes_per_fp32  # 4 MiB total across all 32 layers!
    memory_reduction_ratio = round(legacy_total_bytes / fixed_total_bytes, 1)
    saved_mib = round((legacy_total_bytes - fixed_total_bytes) / (1024 * 1024), 2)

    # EngineCore shutdown timeout matrix (vllm@d143a9de3915)
    def resolve_timeout(
        is_rocm: bool,
        worker_env: float | None,
        request_timeout: float | None,
        process_timeout: float | None,
    ) -> float | None:
        if request_timeout == 0 and process_timeout == 0:
            if worker_env is not None:
                return float(worker_env)
            if is_rocm:
                return 15.0
        return process_timeout

    timeout_cases = {
        "rocm_default_abort": resolve_timeout(True, None, 0, 0),
        "cuda_default_abort": resolve_timeout(False, None, 0, 0),
        "cuda_custom_env_30s_abort": resolve_timeout(False, 30.0, 0, 0),
        "cuda_custom_env_30s_after_drain_expired": resolve_timeout(False, 30.0, 7.0, 0),
    }

    return {
        "num_target_layers": num_layers,
        "head_dim": head_dim,
        "config_max_position_embeddings": config_max_pos,
        "deployed_max_model_len": deployed_max_model_len,
        "legacy_rope_vram_mib": round(legacy_total_bytes / (1024 * 1024), 2),
        "fixed_rope_vram_mib": round(fixed_total_bytes / (1024 * 1024), 2),
        "rope_vram_saved_mib": saved_mib,
        "rope_vram_reduction_x": memory_reduction_ratio,
        "timeout_cases": timeout_cases,
    }


def run_experiment() -> dict[str, Any]:
    mxfp8 = model_trtllm_mxfp8_e8m0_subnormal_nan()
    sglang = model_sglang_gguf_moe_and_cosmos_drain()
    vllm = model_vllm_shared_rope_and_shutdown_grace()

    summary: dict[str, Any] = {
        "experiment": "sglang-gguf-moe-trtllm-mxfp8-e8m0-vllm-shared-rope",
        "phase": 66,
        "priority": 75,
        "track": "Track B — SGLang GGUF MoE & Drain, TensorRT-LLM MXFP8 E8M0 Subnormal Fix & vLLM Shared RoPE",
        "trtllm_mxfp8_e8m0_subnormal_nan": mxfp8,
        "sglang_gguf_moe_and_cosmos_drain": sglang,
        "vllm_shared_rope_and_shutdown_grace": vllm,
        "invariants": {
            "e8m0_zero_is_fp32_subnormal": mxfp8["is_fp32_subnormal"],
            "legacy_mxfp8_rcp_ftz_produces_nan": mxfp8["legacy_zero_lane_product_is_nan"],
            "fixed_mxfp8_exp2f_rcp_is_finite": mxfp8["fixed_exp2f_rcp_is_finite"],
            "gguf_moe_fixed_tile_y_ds_rows": sglang["fixed_tile_y_ds_populated_rows"],
            "vllm_rope_vram_reduction_x": vllm["rope_vram_reduction_x"],
            "vllm_drain_deadline_not_extended_by_worker_env": (
                vllm["timeout_cases"]["cuda_custom_env_30s_after_drain_expired"] == 0
            ),
        },
    }
    return summary


def verify_summary(summary: dict[str, Any]) -> None:
    inv = summary["invariants"]
    assert inv["e8m0_zero_is_fp32_subnormal"] is True
    assert inv["legacy_mxfp8_rcp_ftz_produces_nan"] is True
    assert inv["fixed_mxfp8_exp2f_rcp_is_finite"] is True
    assert inv["gguf_moe_fixed_tile_y_ds_rows"] == 64
    assert inv["vllm_rope_vram_reduction_x"] == 512.0
    assert inv["vllm_drain_deadline_not_extended_by_worker_env"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    verify_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.check:
        print(f"[PASS] {summary['experiment']} invariants verified -> {out_path}")


if __name__ == "__main__":
    main()
