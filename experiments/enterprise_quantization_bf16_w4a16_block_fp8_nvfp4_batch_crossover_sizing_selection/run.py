from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unquantized BF16 (W16A16 · 2 Bytes/Param · High HBM Footprint & Half Tensor Core Peak)",
            "weight_size_70b_gb": 140.0,
            "benchmark_accuracy_retention_pct": 100.0,
            "batch_8_tpot_ms": 26.4,
            "batch_64_goodput_tok_s": 1120,
            "cost_per_1m_tokens_usd": 0.44,
        },
        {
            "tier": "Tier 2: W4A16 AWQ/Marlin (Weight-Only 4-Bit · Fast at Batch<=16, Dequantization Bottleneck at Batch>=48)",
            "weight_size_70b_gb": 38.5,
            "benchmark_accuracy_retention_pct": 98.2,
            "batch_8_tpot_ms": 9.8,
            "batch_64_goodput_tok_s": 1540,
            "cost_per_1m_tokens_usd": 0.31,
        },
        {
            "tier": "Tier 3: Fine-Grained Block-FP8 (W8A8 1x128 Act / 128x128 Weight · Hopper H100/H200 Production Standard)",
            "weight_size_70b_gb": 70.0,
            "benchmark_accuracy_retention_pct": 99.8,
            "batch_8_tpot_ms": 13.6,
            "batch_64_goodput_tok_s": 2480,
            "cost_per_1m_tokens_usd": 0.19,
        },
        {
            "tier": "Tier 4: Hardware-Native NVFP4 / MXFP4 (W4A4 Micro-Block E2M1 + E4M3 Scale · Blackwell B200/GB200 & MI355X)",
            "weight_size_70b_gb": 36.8,
            "benchmark_accuracy_retention_pct": 99.3,
            "batch_8_tpot_ms": 6.4,
            "batch_64_goodput_tok_s": 4820,
            "cost_per_1m_tokens_usd": 0.11,
        },
    ]
    return {
        "experiment_id": "enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "w4a16_vs_w8a8_crossover_batch_size": 32,
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_quantization_bf16_w4a16_block_fp8_nvfp4_batch_crossover_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
