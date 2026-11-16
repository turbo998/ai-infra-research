from __future__ import annotations

import json
from pathlib import Path


def evaluate_vocab_parallel_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unpadded Raw Tokenizer Vocab (V=151,665, Unaligned Scalar Fallback)",
            "padded_vocab_size": 151665,
            "lm_head_gemm_ms": 4.85,
            "tensor_core_util_pct": 48.2,
            "ghost_token_crash_pct": 0.0,
            "pp_tied_ppl_degradation": 0.00,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Padded to TP×128 (V=152,064) Without Ghost-Row -inf Masking",
            "padded_vocab_size": 152064,
            "lm_head_gemm_ms": 2.98,
            "tensor_core_util_pct": 78.6,
            "ghost_token_crash_pct": 1.80,
            "pp_tied_ppl_degradation": 0.00,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Padded + Ghost Masked, Missing PP Stage-0 <-> Stage-(P-1) Tied Sync",
            "padded_vocab_size": 152064,
            "lm_head_gemm_ms": 2.98,
            "tensor_core_util_pct": 78.6,
            "ghost_token_crash_pct": 0.0,
            "pp_tied_ppl_degradation": 0.42,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 4: TP×128 Padded (152,064) + Ghost -inf Mask + PP Endpoint Tied Sync",
            "padded_vocab_size": 152064,
            "lm_head_gemm_ms": 2.98,
            "tensor_core_util_pct": 78.6,
            "ghost_token_crash_pct": 0.0,
            "pp_tied_ppl_degradation": 0.00,
            "decision": "Act automatically",
        },
    ]

    alignment_multiples = [
        {"multiple": 1, "padded_vocab": 151665, "gemm_ms": 4.85, "fp8_tma_aligned": False},
        {"multiple": 8, "padded_vocab": 151680, "gemm_ms": 3.65, "fp8_tma_aligned": False},
        {"multiple": 64, "padded_vocab": 152064, "gemm_ms": 2.98, "fp8_tma_aligned": True},
        {"multiple": 128, "padded_vocab": 152064, "gemm_ms": 2.98, "fp8_tma_aligned": True},
    ]

    summary = {
        "benchmark": "llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "raw_vocab_size": 151665,
        "tp_size": 8,
        "per_rank_multiple": 128,
        "padded_vocab_size": 152064,
        "per_rank_vocab_size": 19008,
        "ghost_rows": 399,
        "unaligned_gemm_ms": 4.85,
        "aligned_gemm_ms": 2.98,
        "gemm_speedup_x": round(4.85 / 2.98, 2),
        "tiers": tiers,
        "alignment_multiples": alignment_multiples,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_vocab_parallel_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
