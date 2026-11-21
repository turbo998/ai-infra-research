from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "llm_quantization_fp8_blockwise_nvfp4_awq_marlin_accuracy_throughput_selection",
        "title": "LLM Quantization Deep Dive: BF16 vs. Per-Tensor FP8 vs. Fine-Grained Block FP8 (1x128/128x128) vs. W4A16 Marlin vs. NVFP4",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model": "Llama-3.3-70B / DeepSeek-R1-671B",
            "gpu_sku": "NVIDIA H200 SXM / B200 SXM",
            "eval_benchmark": "MMLU-Pro + LiveCodeBench + Math-500",
        },
        "tiers": [
            {
                "tier_id": "tier1_unquantized_bf16_weights_and_bf16_kv_cache",
                "name": "Tier 1: Unquantized BF16 (W16A16KV16 — Zero Accuracy Loss, 2x HBM Footprint & Half TensorCore FLOPs)",
                "relative_benchmark_accuracy_retention_pct": 100.0,
                "sla_compliant_goodput_tps_per_gpu": 980,
                "achieved_p99_tpot_ms": 24.6,
                "cost_per_1m_tokens_usd": 0.92,
                "verdict": "Baseline: 100% accuracy, but consumes 2 Bytes/param and caps Goodput at 980 tok/s/GPU",
            },
            {
                "tier_id": "tier2_naive_static_per_tensor_fp8_or_uncalibrated_int4_rtn",
                "name": "Tier 2: Naive Static Per-Tensor FP8 / RTN INT4 (Outlier Activation Clipping Degrades Coding/Math Accuracy)",
                "relative_benchmark_accuracy_retention_pct": 94.2,
                "sla_compliant_goodput_tps_per_gpu": 1950,
                "achieved_p99_tpot_ms": 13.9,
                "cost_per_1m_tokens_usd": 0.46,
                "verdict": "Anti-Pattern: Massive outlier channels in SwiGLU/MoE clip per-tensor scales (-5.8% accuracy drop!)",
            },
            {
                "tier_id": "tier3_w4a16_awq_marlin_kernel_for_low_batch_or_a100",
                "name": "Tier 3: Calibrated W4A16 AWQ + Marlin Kernel (0.55B/param, Best for A100 or Low-Batch B<=16 Serving)",
                "relative_benchmark_accuracy_retention_pct": 98.4,
                "sla_compliant_goodput_tps_per_gpu": 1540,
                "achieved_p99_tpot_ms": 11.8,
                "cost_per_1m_tokens_usd": 0.54,
                "verdict": "Fastest low-batch TPOT (11.8ms) & fits on fewer GPUs, but FP16 compute caps high-batch prefill/decode",
            },
            {
                "tier_id": "tier4_fine_grained_block_fp8_1x128_128x128_and_blackwell_nvfp4",
                "name": "Tier 4: Fine-Grained Block FP8 (1x128 Act × 128x128 Weight + FP8 KV) / Blackwell NVFP4 Micro-Block",
                "relative_benchmark_accuracy_retention_pct": 99.8,
                "sla_compliant_goodput_tps_per_gpu": 2140,
                "achieved_p99_tpot_ms": 12.9,
                "cost_per_1m_tokens_usd": 0.36,
                "verdict": "Recommended: 99.8% accuracy retention (near-lossless), 2,140 Goodput tok/s/GPU (2.18x) & -60.9% $/1M",
            },
        ],
        "headline_findings": {
            "accuracy_retention_pct": 99.8,
            "goodput_tps_per_gpu": 2140,
            "goodput_speedup_vs_bf16_x": 2.18,
            "cost_reduction_vs_bf16_pct": 60.9,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
