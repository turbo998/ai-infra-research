from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    spec_methods = [
        {
            "method": "Tier 1: Standalone 7B Draft Model Pairing (Separate KV Cache + 28 Layers Serial Forwarding)",
            "extra_vram_overhead_gb": 16.5,
            "mean_accepted_tokens_per_step_tau": 2.35,
            "batch_8_tpot_speedup_x": 1.32,
            "batch_48_tpot_speedup_x": 0.72,
        },
        {
            "method": "Tier 2: Prompt Lookup Decoding / N-gram Match (Zero Draft Weights, Copies Spans from Input Prompt)",
            "extra_vram_overhead_gb": 0.05,
            "mean_accepted_tokens_per_step_tau": 2.60,
            "batch_8_tpot_speedup_x": 2.15,
            "batch_48_tpot_speedup_x": 1.18,
        },
        {
            "method": "Tier 3: Medusa Independent Parallel Heads (Non-Autoregressive Heads without Cross-Token Dependency)",
            "extra_vram_overhead_gb": 2.80,
            "mean_accepted_tokens_per_step_tau": 2.10,
            "batch_8_tpot_speedup_x": 1.58,
            "batch_48_tpot_speedup_x": 0.84,
        },
        {
            "method": "Tier 4: EAGLE-3 / Native Multi-Token Prediction (MTP 1-Layer Autoregressive Feature Head + Dynamic Batch<=24 Gate)",
            "extra_vram_overhead_gb": 2.20,
            "mean_accepted_tokens_per_step_tau": 3.45,
            "batch_8_tpot_speedup_x": 2.42,
            "batch_48_tpot_speedup_x": 1.00,
        },
    ]
    return {
        "experiment_id": "enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "dynamic_bypass_batch_threshold": 32,
        "spec_methods": spec_methods,
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
        print("OK: enterprise_speculative_decoding_draft_pld_eagle3_mtp_acceptance_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
