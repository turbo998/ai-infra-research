from __future__ import annotations

import json
from pathlib import Path


def evaluate_penalty_kernel_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Per-Step [B, L_max] Token History Rescan + Python Loop bincount",
            "penalty_latency_32k_ms": 9.40,
            "workspace_hbm_gb": 2.45,
            "negative_logit_sign_bug": True,
            "context_scaling_order": "O(B * L_max)",
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Batched GPU scatter_add Rebuild from [B, L_max] + Dense [B, V]",
            "penalty_latency_32k_ms": 3.20,
            "workspace_hbm_gb": 1.15,
            "negative_logit_sign_bug": False,
            "context_scaling_order": "O(B * L_max + B * V)",
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: O(1) Incremental Token Counter + Dense [B, V] Penalty Apply",
            "penalty_latency_32k_ms": 0.48,
            "workspace_hbm_gb": 0.31,
            "negative_logit_sign_bug": False,
            "context_scaling_order": "O(B * V)",
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: O(1) Incremental Counter + Sparse CSR Unique-Token & COO Logit-Bias Kernel",
            "penalty_latency_32k_ms": 0.14,
            "workspace_hbm_gb": 0.04,
            "negative_logit_sign_bug": False,
            "context_scaling_order": "O(B * |U_seen|)",
            "decision": "Act automatically",
        },
    ]

    context_scaling = [
        {"context_tokens": 4096, "rescan_ms": 1.25, "incremental_csr_ms": 0.12},
        {"context_tokens": 16384, "rescan_ms": 4.75, "incremental_csr_ms": 0.13},
        {"context_tokens": 32768, "rescan_ms": 9.40, "incremental_csr_ms": 0.14},
        {"context_tokens": 131072, "rescan_ms": 37.60, "incremental_csr_ms": 0.16},
    ]

    summary = {
        "benchmark": "llm_serving_repetition_frequency_penalty_logit_bias_sparse_csr_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 256,
        "vocab_size": 152064,
        "context_tokens": 32768,
        "naive_32k_latency_ms": 9.40,
        "sparse_csr_32k_latency_ms": 0.14,
        "speedup_32k_x": round(9.40 / 0.14, 1),
        "naive_workspace_gb": 2.45,
        "sparse_workspace_gb": 0.04,
        "workspace_reduction_x": round(2.45 / 0.04, 1),
        "tiers": tiers,
        "context_scaling": context_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_penalty_kernel_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
