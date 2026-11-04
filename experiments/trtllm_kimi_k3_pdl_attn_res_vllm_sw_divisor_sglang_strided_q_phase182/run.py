#!/usr/bin/env python3
"""Phase 182: TensorRT-LLM Kimi-K3 CuTe DSL PDL Decode Attention-Residual Mix & GPU Vision Pixel Normalization, vLLM Hybrid Spec Decode SW Divisor Block Sizing, CUTLASS FP8 IPC Restarts & Lazy KV Offload Eviction-End Placement, SGLang R3 Non-Output Rank Capture Skip, DSpark Single Mamba Tracking Refresh & AITER Zero-Copy Strided-Q Unified Extend, and Dynamo PoolRelay & Remote Prefill Cancellation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_inference_serving_phase182() -> dict[str, object]:
    # 1. TensorRT-LLM (#21399, c8d72931196a): Kimi-K3 decode attention-residual CuTe DSL PDL mix + folded MoE residual + MXFP8 handoff to KDA
    # GB300 M=7..64: C++ 3.3-11.0 us -> CuTe DSL PDL 1.7-2.6 us (max speedup 11.0 / 2.6 = 4.231x)
    # Rubin M=32, N=9 folded mix + MoE add: 6.01 -> 2.83 us (2.124x); Rubin helix16 b1 mtp7 decode step: 11.45 -> 9.93 ms
    trtllm_kimi_k3_gb300_cpp_max_us = 11.0
    trtllm_kimi_k3_gb300_cute_pdl_us = 2.6
    trtllm_kimi_k3_gb300_attn_res_max_speedup = round(
        trtllm_kimi_k3_gb300_cpp_max_us / trtllm_kimi_k3_gb300_cute_pdl_us, 3
    )  # 4.231
    trtllm_kimi_k3_rubin_folded_prev_us = 6.01
    trtllm_kimi_k3_rubin_folded_new_us = 2.83
    trtllm_kimi_k3_rubin_folded_moe_speedup = round(
        trtllm_kimi_k3_rubin_folded_prev_us / trtllm_kimi_k3_rubin_folded_new_us, 3
    )  # 2.124
    trtllm_kimi_k3_rubin_mtp7_step_prev_ms = 11.45
    trtllm_kimi_k3_rubin_mtp7_step_new_ms = 9.93
    trtllm_kimi_k3_rubin_mtp7_step_reduction_pct = round(
        (1.0 - trtllm_kimi_k3_rubin_mtp7_step_new_ms / trtllm_kimi_k3_rubin_mtp7_step_prev_ms) * 100.0, 2
    )  # 13.28
    trtllm_kimi_k3_fused_attn_res_max_tokens = 64

    # 2. TensorRT-LLM (#20973, 0160bf7a89f4): GPU vision pixel normalization via uint8 LUT on 4x GB200 (Qwen3-VL-235B-A22B NVFP4)
    # Steady screenshot TTFT: 183.2 -> 151.8 ms (-17.14%); cold chat HTTP CPU: 110.0 -> 88.8 ms (-19.27%); H2D bytes halved (uint8 vs bf16 = 2.0x)
    trtllm_vision_screenshot_ttft_off_ms = 183.2
    trtllm_vision_screenshot_ttft_on_ms = 151.8
    trtllm_vision_gpu_norm_ttft_reduction_pct = round(
        (1.0 - trtllm_vision_screenshot_ttft_on_ms / trtllm_vision_screenshot_ttft_off_ms) * 100.0, 2
    )  # 17.14
    trtllm_vision_http_cpu_off_ms = 110.0
    trtllm_vision_http_cpu_on_ms = 88.8
    trtllm_vision_http_cpu_reduction_pct = round(
        (1.0 - trtllm_vision_http_cpu_on_ms / trtllm_vision_http_cpu_off_ms) * 100.0, 2
    )  # 19.27
    trtllm_vision_h2d_bandwidth_compression_ratio = 2.0

    # 3. vLLM (#60252, 59519ec0ee0b; #64119, 69d27bf8002a; #59383, a92efadb30f5): SW draft block divisor of primary block (1648 vs inflated LCM 6592)
    vllm_sw_lcm_inflated_block_tokens = 6592
    vllm_sw_divisor_unified_block_tokens = 1648
    vllm_sw_lcm_inflation_reduction_ratio = round(
        vllm_sw_lcm_inflated_block_tokens / vllm_sw_divisor_unified_block_tokens, 2
    )  # 4.0
    vllm_rtx_pro_5000_prefix_hit_rate_before_pct = 43.8
    vllm_rtx_pro_5000_prefix_hit_rate_after_pct = 71.2
    vllm_rtx_pro_5000_prefix_hit_gain_pts = round(
        vllm_rtx_pro_5000_prefix_hit_rate_after_pct - vllm_rtx_pro_5000_prefix_hit_rate_before_pct, 1
    )  # 27.4
    vllm_dgx_spark_agentic_replay_before_min = 26.1
    vllm_dgx_spark_agentic_replay_after_min = 20.0
    vllm_dgx_spark_agentic_replay_speedup = round(
        vllm_dgx_spark_agentic_replay_before_min / vllm_dgx_spark_agentic_replay_after_min, 3
    )  # 1.305

    # 4. SGLang (#47822, 27aedf7f93cc; #47553, 744af7ce724f; #47494, 435418253e5e; #48327, fd19cfb660f4)
    # AITER zero-copy strided-Q unified extend on gfx1250 (7,548 tok): TP2 window 72.7 -> 44.2 us (1.645x, -39.20%); TP1 window 252.5 -> 196.9 us (1.282x, -22.02%)
    sglang_aiter_tp2_window_old_us = 72.7
    sglang_aiter_tp2_window_new_us = 44.2
    sglang_aiter_strided_q_tp2_window_speedup = round(
        sglang_aiter_tp2_window_old_us / sglang_aiter_tp2_window_new_us, 3
    )  # 1.645
    sglang_r3_tp8_device_buf_before_mb = 96  # 8 x 12 MB
    sglang_r3_tp8_device_buf_after_mb = 12   # 1 x 12 MB
    sglang_r3_tp8_buffer_reduction_ratio = round(
        sglang_r3_tp8_device_buf_before_mb / sglang_r3_tp8_device_buf_after_mb, 1
    )  # 8.0
    sglang_dspark_mamba_track_refreshes_per_step = 1
    sglang_retirer_consolidated_finish_callsites = 7

    # 5. Dynamo (#17716, 8b4ad6a1b041 & #14775, e6b5a99a7a45): PoolRelay live state sharing & remote prefill cancellation propagation
    dynamo_pool_relay_connection_modes = 2  # relay-initiated and router-initiated
    dynamo_orphaned_remote_prefill_on_compatible_cancel = 0

    return {
        "suite": "trtllm_kimi_k3_pdl_attn_res_vllm_sw_divisor_sglang_strided_q_phase182",
        "phase": 182,
        "priority": 191,
        "trtllm_kimi_k3_gb300_attn_res_max_speedup": trtllm_kimi_k3_gb300_attn_res_max_speedup,
        "trtllm_kimi_k3_rubin_folded_moe_speedup": trtllm_kimi_k3_rubin_folded_moe_speedup,
        "trtllm_kimi_k3_rubin_mtp7_step_reduction_pct": trtllm_kimi_k3_rubin_mtp7_step_reduction_pct,
        "trtllm_kimi_k3_fused_attn_res_max_tokens": trtllm_kimi_k3_fused_attn_res_max_tokens,
        "trtllm_vision_gpu_norm_ttft_reduction_pct": trtllm_vision_gpu_norm_ttft_reduction_pct,
        "trtllm_vision_http_cpu_reduction_pct": trtllm_vision_http_cpu_reduction_pct,
        "trtllm_vision_h2d_bandwidth_compression_ratio": trtllm_vision_h2d_bandwidth_compression_ratio,
        "vllm_sw_lcm_inflation_reduction_ratio": vllm_sw_lcm_inflation_reduction_ratio,
        "vllm_rtx_pro_5000_prefix_hit_gain_pts": vllm_rtx_pro_5000_prefix_hit_gain_pts,
        "vllm_dgx_spark_agentic_replay_speedup": vllm_dgx_spark_agentic_replay_speedup,
        "sglang_aiter_strided_q_tp2_window_speedup": sglang_aiter_strided_q_tp2_window_speedup,
        "sglang_r3_tp8_buffer_reduction_ratio": sglang_r3_tp8_buffer_reduction_ratio,
        "sglang_dspark_mamba_track_refreshes_per_step": sglang_dspark_mamba_track_refreshes_per_step,
        "sglang_retirer_consolidated_finish_callsites": sglang_retirer_consolidated_finish_callsites,
        "dynamo_pool_relay_connection_modes": dynamo_pool_relay_connection_modes,
        "dynamo_orphaned_remote_prefill_on_compatible_cancel": dynamo_orphaned_remote_prefill_on_compatible_cancel,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    results = simulate_inference_serving_phase182()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert results["trtllm_kimi_k3_gb300_attn_res_max_speedup"] == 4.231
        assert results["trtllm_kimi_k3_rubin_folded_moe_speedup"] == 2.124
        assert results["trtllm_kimi_k3_rubin_mtp7_step_reduction_pct"] == 13.28
        assert results["trtllm_kimi_k3_fused_attn_res_max_tokens"] == 64
        assert results["trtllm_vision_gpu_norm_ttft_reduction_pct"] == 17.14
        assert results["trtllm_vision_http_cpu_reduction_pct"] == 19.27
        assert results["trtllm_vision_h2d_bandwidth_compression_ratio"] == 2.0
        assert results["vllm_sw_lcm_inflation_reduction_ratio"] == 4.0
        assert results["vllm_rtx_pro_5000_prefix_hit_gain_pts"] == 27.4
        assert results["vllm_dgx_spark_agentic_replay_speedup"] == 1.305
        assert results["sglang_aiter_strided_q_tp2_window_speedup"] == 1.645
        assert results["sglang_r3_tp8_buffer_reduction_ratio"] == 8.0
        assert results["sglang_dspark_mamba_track_refreshes_per_step"] == 1
        assert results["sglang_retirer_consolidated_finish_callsites"] == 7
        assert results["dynamo_pool_relay_connection_modes"] == 2
        assert results["dynamo_orphaned_remote_prefill_on_compatible_cancel"] == 0

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
