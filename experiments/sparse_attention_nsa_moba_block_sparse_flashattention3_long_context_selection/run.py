from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "sparse_attention_nsa_moba_block_sparse_flashattention3_long_context_selection",
        "title": "Hardware-Aligned Sparse Attention Selection (128K-1M Context): DeepSeek NSA vs. Moonshot MoBA vs. Dense FlashAttention-3",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "context_length_tokens": 131072,
            "model": "67B / 70B Long-Context Reasoning & Document Analysis",
            "hardware": "8x NVIDIA H100 / H200 SXM",
        },
        "tiers": [
            {
                "tier": "Tier 1: Dense Full Attention (FlashAttention-3 FP16/BF16, 100% O(S^2) Tokens)",
                "prefill_128k_ttft_sec": 11.80,
                "decode_128k_tpot_ms": 42.0,
                "prefill_speedup_vs_dense": 1.00,
                "ruler_128k_accuracy_pct": 98.8,
                "recommended_scenario": "Short-to-medium context (<32K tokens) where O(S^2) attention is <30% of FLOPs",
            },
            {
                "tier": "Tier 2: Static Sliding Window + Attention Sinks (StreamingLLM 4 Sinks + 4K Window)",
                "prefill_128k_ttft_sec": 1.45,
                "decode_128k_tpot_ms": 8.5,
                "prefill_speedup_vs_dense": 8.14,
                "ruler_128k_accuracy_pct": 24.5,
                "recommended_scenario": "Infinite streaming chat without middle-document retrieval requirements",
            },
            {
                "tier": "Tier 3: MoBA (Mixture of Block Attention, Parameter-Free Top-K Block Routing + FlashVarlen)",
                "prefill_128k_ttft_sec": 2.45,
                "decode_128k_tpot_ms": 18.5,
                "prefill_speedup_vs_dense": 4.82,
                "ruler_128k_accuracy_pct": 99.1,
                "recommended_scenario": "Drop-in sparse adaptation of existing dense models via continued pre-training",
            },
            {
                "tier": "Tier 4 (Recommended): NSA (Native Sparse Attention: Coarse Compression + GQA-Aligned Block Selection + Sliding Window)",
                "prefill_128k_ttft_sec": 1.31,
                "decode_128k_tpot_ms": 10.0,
                "prefill_speedup_vs_dense": 9.01,
                "ruler_128k_accuracy_pct": 99.7,
                "recommended_scenario": "Native 128K-1M pre-training & serving achieving 9.0x Prefill and 4.2x Decode speedup",
            },
        ],
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
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
