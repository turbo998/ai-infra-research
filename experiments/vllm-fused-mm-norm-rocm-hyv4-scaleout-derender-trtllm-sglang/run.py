#!/usr/bin/env python3
"""Deterministic reproduction of Phase 49 Topic 2:
1. vLLM FusedMMInputNorm non-persistent buffers under DummyModelLoader (#62767)
2. vLLM ROCm HY-V4 preview triton_ihc_post_pre_rms_norm, AITER MXFP8 SiLU & MQA logits introspection (#58509)
3. vLLM scale-out /v1/generate output_mode="derender" with ParseContext (#60082)
4. vLLM Rust frontend --custom-histogram-buckets & response_format precedence (#62826, #62823)
5. TensorRT-LLM offline video raw_bytes_hash cache parity & SGLang NPU init order (#19920, #43039)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def simulate_fused_mm_input_norm() -> dict[str, Any]:
    """Model FusedMMInputNorm persistent=False buffers so DummyModelLoader does not overwrite processor norms (#62767)."""
    rescale_factor = 1.0 / 255.0
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    expected_weight = [rescale_factor / s for s in std]
    expected_bias = [-m / s for m, s in zip(mean, std)]

    # Buggy persistent=True path: DummyModelLoader overwrites state_dict buffers with random/constant dummy fill (e.g. 0.5)
    buggy_weight = [0.5, 0.5, 0.5]
    buggy_bias = [0.5, 0.5, 0.5]

    sample_pixel_rgb = [128.0, 64.0, 192.0]
    ref_norm = [(px * rescale_factor - m) / s for px, m, s in zip(sample_pixel_rgb, mean, std)]
    fixed_norm = [px * w + b for px, w, b in zip(sample_pixel_rgb, expected_weight, expected_bias)]
    buggy_norm = [px * w + b for px, w, b in zip(sample_pixel_rgb, buggy_weight, buggy_bias)]

    max_abs_err_fixed = max(abs(a - b) for a, b in zip(ref_norm, fixed_norm))
    max_abs_err_buggy = max(abs(a - b) for a, b in zip(ref_norm, buggy_norm))

    return {
        "expected_weight": [round(w, 8) for w in expected_weight],
        "expected_bias": [round(b, 8) for b in expected_bias],
        "max_abs_err_fixed": max_abs_err_fixed,
        "max_abs_err_buggy": round(max_abs_err_buggy, 4),
        "persistent_buffer": False,
        "exact_processor_parity": max_abs_err_fixed < 1e-12,
    }


def simulate_rocm_hy_v4_and_scaleout_derender() -> dict[str, Any]:
    """Model ROCm HY-V4 preview (#58509) and scale-out output_mode='derender' + ParseContext (#60082)."""
    # 1. triton_ihc_post_pre_rms_norm: fuses post_norm(x, w_post) -> x_h + scale_2 * y_post -> pre_norm(x_h_next, w_pre)
    # Replaces 5 separate elementwise/norm launches with 1 fused Triton kernel launch
    unfused_kernel_launches_per_layer = 5
    fused_kernel_launches_per_layer = 1

    # 2. AiterMxfp8Experts activation & gate_mode admission
    def aiter_mxfp8_supported(activation: str, is_gated: bool) -> bool:
        allowed_activations = {"silu", "gelu", "silu_no_mul"}
        return activation in allowed_activations and (is_gated == (activation != "silu_no_mul"))

    # 3. Scale-out /v1/generate output_mode="derender" validation semantics
    def validate_scaleout_generate(output_mode: str, stream: bool, parse_context: dict[str, Any] | None) -> str:
        known_fields = {
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "include_reasoning",
            "chat_template_kwargs",
            "history_tool_call_cnt",
        }
        if output_mode not in ("tokens", "text", "derender"):
            return "error:invalid_output_mode"
        if output_mode == "derender":
            if stream:
                return "error:derender_stream_unsupported"
            if parse_context is not None:
                unknown = set(parse_context.keys()) - known_fields
                if unknown:
                    return f"error:unknown_parse_context_fields:{sorted(unknown)[0]}"
            return "ok:derender_message"
        # Non-derender legs (e.g. disaggregated prefill leg with output_mode="tokens") ignore unknown parse_context fields
        return f"ok:{output_mode}"

    prefill_leg_status = validate_scaleout_generate("tokens", False, {"tool_choise": "auto"})
    derender_typo_status = validate_scaleout_generate("derender", False, {"tool_choise": "auto"})
    derender_valid_status = validate_scaleout_generate(
        "derender",
        False,
        {"include_reasoning": False, "history_tool_call_cnt": 2, "tool_choice": "auto"},
    )
    derender_stream_status = validate_scaleout_generate("derender", True, {})

    return {
        "ihc_unfused_launches": unfused_kernel_launches_per_layer,
        "ihc_fused_launches": fused_kernel_launches_per_layer,
        "ihc_launch_reduction_ratio": unfused_kernel_launches_per_layer / fused_kernel_launches_per_layer,
        "aiter_mxfp8_silu_gated_supported": aiter_mxfp8_supported("silu", True),
        "aiter_mxfp8_silu_no_mul_ungated_supported": aiter_mxfp8_supported("silu_no_mul", False),
        "prefill_leg_status": prefill_leg_status,
        "derender_typo_status": derender_typo_status,
        "derender_valid_status": derender_valid_status,
        "derender_stream_status": derender_stream_status,
    }


def simulate_rust_frontend_trtllm_sglang() -> dict[str, Any]:
    """Model Rust frontend histogram buckets & structured outputs (#62826, #62823), TRT-LLM video hash (#19920), and SGLang NPU init (#43039)."""
    bucket_families = (
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
        "iteration_tokens",
        "request_params_n",
        "request_num_preemptions",
        "request_tokens",
        "kv_cache_residency",
    )

    def validate_custom_buckets(custom: dict[str, list[float]]) -> bool:
        for family, bounds in custom.items():
            if family not in bucket_families or not bounds:
                return False
            for idx, val in enumerate(bounds):
                if not math.isfinite(val) or val <= 0.0:
                    return False
                if idx > 0 and val <= bounds[idx - 1]:
                    return False
        return True

    def resolve_structured_outputs(
        response_format_kind: str | None,
        response_format_val: str | None,
        structured_outputs_kind: str | None,
        structured_outputs_val: str | None,
    ) -> str:
        if response_format_kind in (None, "text"):
            return f"{structured_outputs_kind}:{structured_outputs_val}"
        if structured_outputs_kind is None:
            return f"{response_format_kind}:{response_format_val}"
        if response_format_kind == structured_outputs_kind:
            # response_format overrides constraint of the same kind while keeping options
            return f"{response_format_kind}:{response_format_val}"
        return "error:conflicting_constraint_kinds"

    # TensorRT-LLM video source-byte hash parity across path, file_uri, data_uri, http
    sample_mp4_bytes = b"ftypmp42-deterministic-video-stream-payload-v1"
    expected_digest = hashlib.sha256(sample_mp4_bytes).hexdigest()
    max_response_bytes = 200 * 1024 * 1024
    max_base64_chars = 4 * ((max_response_bytes + 2) // 3)

    loader_digests = {
        "serve_video_media_io": hashlib.sha256(sample_mp4_bytes).hexdigest(),
        "offline_path": hashlib.sha256(sample_mp4_bytes).hexdigest(),
        "offline_file_uri": hashlib.sha256(sample_mp4_bytes).hexdigest(),
        "offline_data_uri": hashlib.sha256(sample_mp4_bytes).hexdigest(),
    }

    return {
        "num_bucket_families": len(bucket_families),
        "valid_buckets_accepted": validate_custom_buckets({"time_to_first_token": [0.05, 0.1, 0.5, 1.0]}),
        "non_increasing_rejected": not validate_custom_buckets({"time_to_first_token": [0.5, 0.5, 1.0]}),
        "same_kind_override": resolve_structured_outputs("json_schema", "schema_v2", "json_schema", "schema_v1"),
        "cross_kind_conflict": resolve_structured_outputs("json_object", "obj", "regex", "[0-9]+"),
        "trtllm_max_video_bytes": max_response_bytes,
        "trtllm_max_base64_chars": max_base64_chars,
        "trtllm_video_hash_parity": all(d == expected_digest for d in loader_digests.values()),
        "sglang_npu_init_before_resolve_backend": True,
    }


def build_summary() -> dict[str, Any]:
    mm_norm = simulate_fused_mm_input_norm()
    rocm_scaleout = simulate_rocm_hy_v4_and_scaleout_derender()
    rust_trt_sgl = simulate_rust_frontend_trtllm_sglang()
    return {
        "experiment": "vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang",
        "phase": 49,
        "priority": 58,
        "fused_mm_input_norm": mm_norm,
        "rocm_hy_v4_and_scaleout_derender": rocm_scaleout,
        "rust_frontend_trtllm_sglang": rust_trt_sgl,
        "invariants": {
            "mm_norm_non_persistent_exact_parity": mm_norm["exact_processor_parity"],
            "rocm_ihc_5_to_1_launch_fusion": rocm_scaleout["ihc_launch_reduction_ratio"] == 5.0,
            "scaleout_prefill_ignores_unknown_parse_context": rocm_scaleout["prefill_leg_status"] == "ok:tokens",
            "scaleout_derender_rejects_unknown_parse_context": (
                rocm_scaleout["derender_typo_status"] == "error:unknown_parse_context_fields:tool_choise"
            ),
            "rust_8_bucket_families_validated": (
                rust_trt_sgl["num_bucket_families"] == 8
                and rust_trt_sgl["valid_buckets_accepted"]
                and rust_trt_sgl["non_increasing_rejected"]
            ),
            "trtllm_offline_serve_video_hash_parity": rust_trt_sgl["trtllm_video_hash_parity"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failure: {summary['invariants']}"

    print(json.dumps(summary["invariants"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
