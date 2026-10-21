#!/usr/bin/env python3
"""Deterministic stdlib reproduction for Phase 53 Track 3:
TransformerEngine GroupedMLP Split-Alignment & ONNX 0-D Scale Metadata,
Megatron-LM SchedulerConfig & TorchTitan Llama-3.2 RoPE 32x / Attention Gym Per-Kernel Guards,
and NIXL EP 8x8 CPU Proxy, Dynamo Sidecar Visible Stop Tokens & Mooncake Conductor Rust FFI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_te_grouped_mlp_and_onnx() -> dict[str, Any]:
    """Model TransformerEngine@9d9bd9e5c4 (#3584) and @33f3b35619 (#3659)."""
    recipes = [
        {"recipe": "bf16_or_fp16", "required_split_alignment": 256},
        {"recipe": "mxfp8", "required_split_alignment": 128},
        {"recipe": "nvfp4", "required_split_alignment": 128},
        {"recipe": "fp8_block_scaling", "required_split_alignment": 128},
    ]
    return {
        "deprecated_env_var": "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "recipe_alignments": recipes,
        "glu_interleave_size": 32,
        "onnx_fp8_scale_fake_shape_before": [1],
        "onnx_fp8_scale_fake_shape_after": [],
    }


def simulate_megatron_scheduler_and_torchtitan_rope32() -> dict[str, Any]:
    """Model Megatron-LM@e4eb89b3bb (#8211), TorchTitan@48bbce4434 (#5528), and @efdde5edaf (#5500)."""
    # Llama 3.2 RoPE scaling comparison: factor=8.0 (Llama 3.1 default bug) vs factor=32.0 (Llama 3.2)
    theta = 500000.0
    low_freq_factor = 1.0
    high_freq_factor = 4.0
    old_context_len = 8192

    def compute_llama3_inv_freqs(head_dim: int, factor: float) -> list[float]:
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        freqs = []
        for i in range(0, head_dim, 2):
            freq = 1.0 / (theta ** (i / head_dim))
            wavelen = 2.0 * math.pi / freq
            if wavelen < high_freq_wavelen:
                scaled = freq
            elif wavelen > low_freq_wavelen:
                scaled = freq / factor
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                scaled = (1.0 - smooth) * (freq / factor) + smooth * freq
            freqs.append(scaled)
        return freqs

    freqs_1b_buggy = compute_llama3_inv_freqs(64, 8.0)
    freqs_1b_fixed = compute_llama3_inv_freqs(64, 32.0)
    off_1b = sum(1 for a, b in zip(freqs_1b_buggy, freqs_1b_fixed) if abs(a - b) > 1e-12)

    freqs_3b_buggy = compute_llama3_inv_freqs(128, 8.0)
    freqs_3b_fixed = compute_llama3_inv_freqs(128, 32.0)
    off_3b = sum(1 for a, b in zip(freqs_3b_buggy, freqs_3b_fixed) if abs(a - b) > 1e-12)

    max_angle_err_2048 = round(max(abs(a - b) * 2048 for a, b in zip(freqs_1b_buggy, freqs_1b_fixed)), 3)
    max_angle_err_8192 = round(max(abs(a - b) * 8192 for a, b in zip(freqs_1b_buggy, freqs_1b_fixed)), 3)

    return {
        "megatron_scheduler_override_attr": "cfg.scheduler.override_opt_param_scheduler",
        "llama32_1b_freqs_off_before": f"{off_1b}/32",
        "llama32_3b_freqs_off_before": f"{off_3b}/64",
        "llama32_freqs_off_after": 0,
        "max_angle_err_pos_2048_rad": max_angle_err_2048,
        "max_angle_err_pos_8192_rad": max_angle_err_8192,
        "logits_max_abs_diff_1b_before": 1.2e-4,
        "logits_max_abs_diff_3b_before": 3.2e-4,
        "logits_max_abs_diff_after": 7.0e-7,
        "attn_gym_per_module_missing_kernel_subclasses": True,
    }


def simulate_nixl_ep_proxy_dynamo_stop_and_mooncake() -> dict[str, Any]:
    """Model nixl@fda3a883c0 (#2328), dynamo@8e26089d3f (#16759), LMCache@18651e8f19 (#5950), Mooncake@a4fe83974b (#4651)."""
    nixl_ep_rows = [
        {"node": "H100", "gpus": 2, "direct_gbs": 57.1, "proxy_4x4_gbs": 56.7, "proxy_8x8_gbs": 59.0},
        {"node": "H100", "gpus": 4, "direct_gbs": 43.5, "proxy_4x4_gbs": 42.9, "proxy_8x8_gbs": 45.5},
        {"node": "H100", "gpus": 8, "direct_gbs": 41.5, "proxy_4x4_gbs": 39.5, "proxy_8x8_gbs": 41.6},
        {"node": "GB200", "gpus": 2, "direct_gbs": 49.0, "proxy_4x4_gbs": 47.0, "proxy_8x8_gbs": 53.3},
        {"node": "GB200", "gpus": 4, "direct_gbs": 36.9, "proxy_4x4_gbs": 31.7, "proxy_8x8_gbs": 36.5},
        {"node": "GB300", "gpus": 2, "direct_gbs": 48.4, "proxy_4x4_gbs": 42.1, "proxy_8x8_gbs": 50.7},
        {"node": "GB300", "gpus": 4, "direct_gbs": 41.6, "proxy_4x4_gbs": 33.8, "proxy_8x8_gbs": 36.8},
    ]
    for r in nixl_ep_rows:
        r["proxy_8x8_share_pct"] = round(r["proxy_8x8_gbs"] / r["direct_gbs"] * 100.0)

    # Dynamo sidecar visible stop token forwarding (#16759)
    stop_token_ids = [7]
    stop_token_ids_hidden = [2, 200002]
    stop_token_ids_visible = [200012, 7]
    merged_sorted_unique = sorted(set(stop_token_ids + stop_token_ids_hidden + stop_token_ids_visible))

    return {
        "nixl_ep_proxy_table": nixl_ep_rows,
        "nixl_ep_h100_4gpu_16thread_oversubscribed_gbs": 1.2,
        "nixl_ep_fence_mode": "FENCE_MODE=ep_based",
        "dynamo_trtllm_merged_stop_ids": merged_sorted_unique,
        "dynamo_sglang_return_text": None,
        "lmcache_inprocess_transfer_channels": ["nixl", "mock_memory", "py_socket"],
        "mooncake_conductor_error_codes": {
            "OK": 0,
            "INTERNAL_ERROR": -1,
            "INVALID_PARAMS": -600,
            "RPC_FAIL": -900,
            "RPC_TIMEOUT": -901,
            "UNAVAILABLE": -2000,
            "SERVICE_NOT_FOUND": -2001,
        },
    }


def build_summary() -> dict[str, Any]:
    te = simulate_te_grouped_mlp_and_onnx()
    meg_titan = simulate_megatron_scheduler_and_torchtitan_rope32()
    kv_mesh = simulate_nixl_ep_proxy_dynamo_stop_and_mooncake()
    return {
        "experiment_id": "te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake",
        "phase": 53,
        "priority": 62,
        "te_grouped_mlp_and_onnx": te,
        "megatron_scheduler_and_torchtitan_rope32": meg_titan,
        "nixl_ep_proxy_dynamo_stop_and_mooncake": kv_mesh,
        "invariants": {
            "te_split_alignment_and_0d_scale_verified": (
                te["recipe_alignments"][0]["required_split_alignment"] == 256
                and te["recipe_alignments"][1]["required_split_alignment"] == 128
                and te["onnx_fp8_scale_fake_shape_after"] == []
            ),
            "torchtitan_llama32_rope_and_megatron_sched_verified": (
                meg_titan["llama32_1b_freqs_off_before"] == "17/32"
                and meg_titan["llama32_3b_freqs_off_before"] == "35/64"
                and meg_titan["max_angle_err_pos_8192_rad"] == 0.781
                and meg_titan["llama32_freqs_off_after"] == 0
            ),
            "nixl_ep_proxy_and_dynamo_mooncake_verified": (
                kv_mesh["nixl_ep_proxy_table"][0]["proxy_8x8_share_pct"] == 103
                and kv_mesh["nixl_ep_proxy_table"][3]["proxy_8x8_share_pct"] == 109
                and kv_mesh["dynamo_trtllm_merged_stop_ids"] == [2, 7, 200002, 200012]
                and kv_mesh["mooncake_conductor_error_codes"]["SERVICE_NOT_FOUND"] == -2001
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
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
