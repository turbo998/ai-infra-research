from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "diffusion_llm_llada_mercury_block_diffusion_vs_autoregressive_selection",
        "title": "Diffusion LLM (LLaDA / Mercury Coder / Block Diffusion) vs. Autoregressive LLM Selection: Breaking the Serial HBM Bandwidth Wall",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "task": "512-Token Code / Structured JSON / Whole-File Edit Generation",
            "hardware": "Single NVIDIA H100 80GB (3.35 TB/s HBM, 989 TFLOPS FP16)",
            "prompt_context_tokens": 4096,
        },
        "tiers": [
            {
                "tier": "Tier 1: Standard Serial Autoregressive Decode (1 Token/Step, Memory-Bandwidth-Bound)",
                "per_user_output_tps": 110.0,
                "e2e_512_token_generation_ms": 4654.0,
                "supports_causal_prefix_kv_cache": True,
                "bidirectional_infilling_and_reversal_acc_pct": 62.5,
                "recommended_scenario": "Sequential token-by-token chat streaming & open-ended long narratives",
            },
            {
                "tier": "Tier 2: Autoregressive + Speculative Decoding (EAGLE-3 / MTP Draft K=4)",
                "per_user_output_tps": 285.0,
                "e2e_512_token_generation_ms": 1796.0,
                "supports_causal_prefix_kv_cache": True,
                "bidirectional_infilling_and_reversal_acc_pct": 62.5,
                "recommended_scenario": "General-purpose AR latency acceleration while preserving causal ordering",
            },
            {
                "tier": "Tier 3: Vanilla Full-Sequence Masked Diffusion LLM (LLaDA Without Prefix KV Cache)",
                "per_user_output_tps": 640.0,
                "e2e_512_token_generation_ms": 800.0,
                "supports_causal_prefix_kv_cache": False,
                "bidirectional_infilling_and_reversal_acc_pct": 94.2,
                "recommended_scenario": "Short-prompt whole-block infilling, reversal curse tasks & fixed-length generation",
            },
            {
                "tier": "Tier 4 (Recommended): Semi-Autoregressive Block Diffusion (BD3-LM / Mercury Style) + Prefix KV Cache + Confidence Unmasking",
                "per_user_output_tps": 980.0,
                "e2e_512_token_generation_ms": 522.0,
                "supports_causal_prefix_kv_cache": True,
                "bidirectional_infilling_and_reversal_acc_pct": 93.8,
                "recommended_scenario": "Ultra-fast IDE whole-file code edits, JSON generation & sub-second agent action blocks (8.9x faster)",
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
