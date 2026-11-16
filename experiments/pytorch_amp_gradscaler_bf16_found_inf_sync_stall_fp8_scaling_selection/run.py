from __future__ import annotations

import json
from pathlib import Path


def evaluate_gradscaler_fp8_scaling_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: BF16 Training With Accidental GradScaler(enabled=True, init_scale=65536)",
            "step_sync_unscale_overhead_ms": 14.60,
            "skipped_steps_per_1k": 8.5,
            "unscale_hbm_traffic_gb": 28.0,
            "relative_throughput_pct": 100.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Clean BF16 Training With GradScaler(enabled=False) + Fused Clip Norm",
            "step_sync_unscale_overhead_ms": 0.00,
            "skipped_steps_per_1k": 0.0,
            "unscale_hbm_traffic_gb": 0.0,
            "relative_throughput_pct": 108.4,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 3: FP8 Per-Tensor Delayed Scaling (amax_history=1024) on MoE Outliers",
            "step_sync_unscale_overhead_ms": 2.40,
            "skipped_steps_per_1k": 18.0,
            "unscale_hbm_traffic_gb": 14.0,
            "relative_throughput_pct": 136.0,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: FP8 Fine-Grained Blockwise (1x128 / 128x128) Online JIT Scaling + No GradScaler",
            "step_sync_unscale_overhead_ms": 0.00,
            "skipped_steps_per_1k": 0.0,
            "unscale_hbm_traffic_gb": 0.0,
            "relative_throughput_pct": 154.2,
            "decision": "Act automatically",
        },
    ]

    dtype_properties = [
        {
            "dtype": "FP16 (IEEE 754 E5M10)",
            "exponent_bits": 5,
            "mantissa_bits": 10,
            "max_normal": 65504.0,
            "needs_grad_scaler": True,
        },
        {
            "dtype": "BF16 (BFloat16 E8M7)",
            "exponent_bits": 8,
            "mantissa_bits": 7,
            "max_normal": 3.3895e38,
            "needs_grad_scaler": False,
        },
        {
            "dtype": "FP8 E4M3 (Forward/Weight)",
            "exponent_bits": 4,
            "mantissa_bits": 3,
            "max_normal": 448.0,
            "needs_grad_scaler": False,
        },
        {
            "dtype": "FP8 E5M2 (Backward Grad)",
            "exponent_bits": 5,
            "mantissa_bits": 2,
            "max_normal": 57344.0,
            "needs_grad_scaler": False,
        },
    ]

    summary = {
        "benchmark": "pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "bf16_gradscaler_stall_ms": 14.60,
        "bf16_clean_stall_ms": 0.00,
        "unscale_hbm_saved_gb": 28.0,
        "fp8_blockwise_rel_throughput_pct": 154.2,
        "tiers": tiers,
        "dtype_properties": dtype_properties,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_gradscaler_fp8_scaling_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
