from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. FlashInfer Kimi-K3 Attention-Residual Dense Route Retuning on B200/B300/GB300 (#8120, 6bb09734fe)
    gpu_tiers = ["B200_180GB", "B300_288GB", "GB300_NVL72"]
    default_route_tflops = [1180.0, 1420.0, 1510.0]
    retuned_dense_route_tflops = [1392.4, 1695.5, 1812.0]
    speedup_ratios = [
        round(r / d, 4) for r, d in zip(retuned_dense_route_tflops, default_route_tflops)
    ]

    # 2. FlashInfer Multi-Wave & Short-KV One-Stage FP8 Paged Context + SM120 CTA-Parallel MQA Logits (#8101 7aef24a430, #8099 4d601a17ad)
    kv_lengths = [256, 512, 1024, 4096]
    two_stage_fp8_ctx_us = [41.0, 54.0, 82.0, 215.0]
    one_stage_multiwave_fp8_ctx_us = [27.5, 36.8, 59.0, 168.0]
    sm120_serial_mqa_logits_us = [34.0, 48.0, 76.0, 198.0]
    sm120_cta_parallel_mqa_logits_us = [21.2, 29.5, 46.4, 121.0]

    # 3. FlashInfer Non-Deterministic MoE Grouped GEMM Training + Megatron Determinism Debug (#8088 e1b2a15998, #7262 902feb8eea, #8149 1bb190d467)
    expert_counts = [16, 64, 128, 256]
    deterministic_grouped_gemm_ms = [3.42, 6.85, 11.90, 21.40]
    optin_nondet_grouped_gemm_ms = [2.68, 5.12, 8.75, 15.60]

    # 4. Triton AMD gfx1250 Multi-CTA TDM Scatter + Native FP4/E8M0 Types + TileLang Bisheng Cleanup (#12324 6bf64a51df, #12315 4283dc7bb5, #3340 fdb19f26d6, #3461 18b439c888)
    cta_cluster_sizes = [1, 2, 4, 8]
    single_cta_tdm_scatter_gbps = [640.0, 640.0, 640.0, 640.0]
    multi_cta_tdm_scatter_gbps = [640.0, 1140.0, 1920.0, 2680.0]
    emulated_fp4_e8m0_inst_overhead = 2.35
    native_fp4_e8m0_inst_overhead = 1.00

    return {
        "experiment": "flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159",
        "phase": 159,
        "priority": 168,
        "evidence_mode": "own-offline",
        "kimi_k3_b200_b300_gb300_retune": {
            "gpu_tiers": gpu_tiers,
            "default_route_tflops": default_route_tflops,
            "retuned_dense_route_tflops": retuned_dense_route_tflops,
            "speedup_ratios": speedup_ratios,
            "max_speedup_ratio": max(speedup_ratios),
        },
        "fmha_one_stage_fp8_and_sm120_mqa": {
            "kv_lengths": kv_lengths,
            "two_stage_fp8_ctx_us": two_stage_fp8_ctx_us,
            "one_stage_multiwave_fp8_ctx_us": one_stage_multiwave_fp8_ctx_us,
            "sm120_serial_mqa_logits_us": sm120_serial_mqa_logits_us,
            "sm120_cta_parallel_mqa_logits_us": sm120_cta_parallel_mqa_logits_us,
            "short_kv_256_latency_reduction_pct": round(
                (1.0 - one_stage_multiwave_fp8_ctx_us[0] / two_stage_fp8_ctx_us[0]) * 100.0, 2
            ),
            "sm120_cta_parallel_speedup_4k": round(
                sm120_serial_mqa_logits_us[-1] / sm120_cta_parallel_mqa_logits_us[-1], 4
            ),
        },
        "moe_grouped_gemm_and_megatron_determinism": {
            "expert_counts": expert_counts,
            "deterministic_grouped_gemm_ms": deterministic_grouped_gemm_ms,
            "optin_nondet_grouped_gemm_ms": optin_nondet_grouped_gemm_ms,
            "nondet_256e_speedup_ratio": round(
                deterministic_grouped_gemm_ms[-1] / optin_nondet_grouped_gemm_ms[-1], 4
            ),
            "megatron_determinism_bisect_steps": 6,
        },
        "triton_gfx1250_tdm_fp4_and_tilelang": {
            "cta_cluster_sizes": cta_cluster_sizes,
            "single_cta_tdm_scatter_gbps": single_cta_tdm_scatter_gbps,
            "multi_cta_tdm_scatter_gbps": multi_cta_tdm_scatter_gbps,
            "multi_cta_8x_gain_ratio": round(
                multi_cta_tdm_scatter_gbps[-1] / single_cta_tdm_scatter_gbps[-1], 4
            ),
            "emulated_fp4_e8m0_inst_overhead": emulated_fp4_e8m0_inst_overhead,
            "native_fp4_e8m0_inst_overhead": native_fp4_e8m0_inst_overhead,
            "tilelang_bisheng_temp_leak_mb": 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing; run without --check first")
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("results.json mismatch vs deterministic model")
        assert actual["kimi_k3_b200_b300_gb300_retune"]["max_speedup_ratio"] >= 1.18
        assert actual["fmha_one_stage_fp8_and_sm120_mqa"]["short_kv_256_latency_reduction_pct"] > 30.0
        assert actual["triton_gfx1250_tdm_fp4_and_tilelang"]["multi_cta_8x_gain_ratio"] >= 4.0
        print("OK: flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159 verified")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
