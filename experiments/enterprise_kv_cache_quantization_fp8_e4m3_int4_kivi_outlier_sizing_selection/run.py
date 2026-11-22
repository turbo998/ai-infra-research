from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    kv_formats = [
        {
            "format": "Tier 1: Uncompressed BF16 / FP16 KV Cache (327.68 KB/tok on Llama-3.3-70B GQA-8)",
            "kv_bytes_per_token_kb": 327.68,
            "longbench_needle_accuracy_pct": 99.1,
            "pd_rdma_transfer_32k_ms_100g": 859.0,
            "max_concurrent_32k_requests_2xh200": 14,
        },
        {
            "format": "Tier 2: Naive Symmetric Per-Token INT4 KV Quantization (Destroys Key Outlier Channels!)",
            "kv_bytes_per_token_kb": 86.0,
            "longbench_needle_accuracy_pct": 68.4,
            "pd_rdma_transfer_32k_ms_100g": 225.0,
            "max_concurrent_32k_requests_2xh200": 52,
        },
        {
            "format": "Tier 3: FP8 E4M3 KV Cache with Per-Attention-Head Calibrated Scaling Factors (Production Default)",
            "kv_bytes_per_token_kb": 163.84,
            "longbench_needle_accuracy_pct": 98.9,
            "pd_rdma_transfer_32k_ms_100g": 429.5,
            "max_concurrent_32k_requests_2xh200": 28,
        },
        {
            "format": "Tier 4: KIVI Asymmetric INT4 KV (Per-Channel Key + Per-Token Value + 128-Token FP16 Residual Window)",
            "kv_bytes_per_token_kb": 90.1,
            "longbench_needle_accuracy_pct": 98.4,
            "pd_rdma_transfer_32k_ms_100g": 236.0,
            "max_concurrent_32k_requests_2xh200": 50,
        },
    ]
    return {
        "experiment_id": "enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "model_reference": "Llama-3.3-70B (80 Layers, GQA-8)",
        "kv_formats": kv_formats,
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
        print("OK: enterprise_kv_cache_quantization_fp8_e4m3_int4_kivi_outlier_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
