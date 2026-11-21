from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "llm_tokenizer_vocab_sizing_fertility_liger_cut_cross_entropy_lm_head_selection",
        "title": "LLM Tokenizer Vocab Sizing (32K vs. 152K) & Fused Cut Cross-Entropy (Liger / CCE) Selection Guide",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "bilingual_zh_en_code_corpus_chars": 10000,
            "training_tokens_per_gpu_step": 32768,
            "vocab_size_large": 152064,
        },
        "tiers": [
            {
                "tier": "Tier 1: Legacy 32K English-Centric BPE (Llama-2 Style) + Standard PyTorch CrossEntropy",
                "tokens_per_10k_zh_chars": 14500,
                "relative_inference_speedup_zh_code": 1.00,
                "lm_head_logits_peak_hbm_gb": 8.4,
                "supports_tp1_fsdp2_long_seq": True,
                "recommended_scenario": "Legacy English-only models (inflates Chinese/Code KV Cache & E2EL by 2.23x)",
            },
            {
                "tier": "Tier 2: 152K Multilingual BPE (Qwen2.5 / Llama-3) + Naive Materialized FP32 CrossEntropy",
                "tokens_per_10k_zh_chars": 6500,
                "relative_inference_speedup_zh_code": 2.23,
                "lm_head_logits_peak_hbm_gb": 39.8,
                "supports_tp1_fsdp2_long_seq": False,
                "recommended_scenario": "Anti-pattern in training (materializing [32K, 152K] FP32 logits + grad eats 39.8 GB HBM!)",
            },
            {
                "tier": "Tier 3: 152K Multilingual BPE + Megatron Vocab-Parallel CrossEntropy (TP=8 Required)",
                "tokens_per_10k_zh_chars": 6500,
                "relative_inference_speedup_zh_code": 2.23,
                "lm_head_logits_peak_hbm_gb": 5.0,
                "supports_tp1_fsdp2_long_seq": False,
                "recommended_scenario": "Megatron-Core TP=8 pre-training (still costs 5.0 GB/GPU and requires TP>=4)",
            },
            {
                "tier": "Tier 4 (Recommended): 152K Multilingual BPE + Fused Linear Cut Cross-Entropy (Liger FLCE / Apple CCE)",
                "tokens_per_10k_zh_chars": 6500,
                "relative_inference_speedup_zh_code": 2.23,
                "lm_head_logits_peak_hbm_gb": 0.12,
                "supports_tp1_fsdp2_long_seq": True,
                "recommended_scenario": "Production FSDP2 & SFT/RL training + bilingual/code serving (331x less logits HBM + 2.23x faster zh-CN)",
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
