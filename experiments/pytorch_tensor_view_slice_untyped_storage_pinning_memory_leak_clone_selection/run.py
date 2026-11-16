from __future__ import annotations

import json
from pathlib import Path


def evaluate_view_storage_leak_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Post-lm_head Logit Slice `lm_head(H)[:, -1, :]` + Retained View Buffer",
            "retained_hbm_32steps_gb": 32.00,
            "lm_head_prefill_hbm_gb": 9.96,
            "lm_head_prefill_latency_ms": 40.80,
            "storage_to_logical_ratio_x": 8192,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Pre-lm_head Slice `H[:, -1, :]` Stored Without `.clone()` Detachment",
            "retained_hbm_32steps_gb": 32.00,
            "lm_head_prefill_hbm_gb": 0.0012,
            "lm_head_prefill_latency_ms": 0.08,
            "storage_to_logical_ratio_x": 8192,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: `.clone()` on Output Slice Only (Uncloned View in `ctx.save_for_backward`)",
            "retained_hbm_32steps_gb": 8.40,
            "lm_head_prefill_hbm_gb": 0.0012,
            "lm_head_prefill_latency_ms": 0.08,
            "storage_to_logical_ratio_x": 64,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Pre-lm_head Gather + `.clone()` Detachment + Storage-Ratio Audit Gate",
            "retained_hbm_32steps_gb": 0.004,
            "lm_head_prefill_hbm_gb": 0.0012,
            "lm_head_prefill_latency_ms": 0.08,
            "storage_to_logical_ratio_x": 1,
            "decision": "Act automatically",
        },
    ]

    seq_scaling = [
        {"seq_len": 2048, "uncloned_view_gb": 8.00, "cloned_slice_gb": 0.004},
        {"seq_len": 4096, "uncloned_view_gb": 16.00, "cloned_slice_gb": 0.004},
        {"seq_len": 8192, "uncloned_view_gb": 32.00, "cloned_slice_gb": 0.004},
        {"seq_len": 16384, "uncloned_view_gb": 64.00, "cloned_slice_gb": 0.004},
    ]

    summary = {
        "benchmark": "pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 16,
        "seq_len": 8192,
        "hidden_dim": 4096,
        "logical_slice_kb": 128.0,
        "pinned_storage_mb": 1024.0,
        "storage_amplification_x": 8192,
        "uncloned_32steps_gb": 32.00,
        "cloned_32steps_gb": 0.004,
        "post_lmhead_ms": 40.80,
        "pre_lmhead_ms": 0.08,
        "lmhead_speedup_x": round(40.80 / 0.08, 1),
        "tiers": tiers,
        "seq_scaling": seq_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_view_storage_leak_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
