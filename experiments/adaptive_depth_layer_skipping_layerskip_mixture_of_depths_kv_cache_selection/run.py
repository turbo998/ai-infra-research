from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "adaptive_depth_layer_skipping_layerskip_mixture_of_depths_kv_cache_selection",
        "title": "Adaptive Depth & Layer Skipping Selection: LayerSkip Self-Speculative Decoding vs. Mixture-of-Depths (MoD) vs. Naive Early-Exit",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "model": "70B (80 Layers, Baseline Accuracy = 82.6%, TPOT = 18.0 ms)",
            "early_exit_layer_e": 28,
            "draft_speculation_tokens_k": 5,
        },
        "tiers": [
            {
                "tier": "Tier 1: Naive Confidence Early-Exit at Layer 28 (Missing Layers 29..80 KV Cache!)",
                "wall_clock_decode_speedup": 1.45,
                "effective_tpot_ms": 12.4,
                "benchmark_accuracy_pct": 64.2,
                "extra_draft_model_hbm_gb": 0.0,
                "recommended_scenario": "Anti-pattern for autoregressive LLMs (missing upper-layer KV states corrupt future tokens)",
            },
            {
                "tier": "Tier 2: KV-Propagation / State-Copying Early-Exit (Approximate Layers 29..80 KV)",
                "wall_clock_decode_speedup": 1.38,
                "effective_tpot_ms": 13.0,
                "benchmark_accuracy_pct": 76.8,
                "extra_draft_model_hbm_gb": 0.0,
                "recommended_scenario": "Lossy heuristic (still loses -5.8 pts accuracy on multi-step reasoning)",
            },
            {
                "tier": "Tier 3: Pre-Trained Mixture-of-Depths (MoD 50% Per-Layer Token Capacity Router)",
                "wall_clock_decode_speedup": 1.62,
                "effective_tpot_ms": 11.1,
                "benchmark_accuracy_pct": 81.9,
                "extra_draft_model_hbm_gb": 0.0,
                "recommended_scenario": "Custom pre-trained foundation models optimizing both Prefill FLOPs & KV Cache (-45%)",
            },
            {
                "tier": "Tier 4 (Recommended): LayerSkip Self-Speculative Decoding (Draft 5 Tok at Layer 28 + Full 80-Layer Parallel Verify & KV Fill)",
                "wall_clock_decode_speedup": 1.86,
                "effective_tpot_ms": 9.7,
                "benchmark_accuracy_pct": 82.6,
                "extra_draft_model_hbm_gb": 0.0,
                "recommended_scenario": "Production 100% lossless acceleration with 0.0 GB extra draft model VRAM",
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
