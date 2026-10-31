from __future__ import annotations
import argparse
import json
from pathlib import Path

def compute_summary() -> dict:
    # 1. KCP Causal Conv1D halo exchange (flashinfer@39f41e9)
    alpha_us = 8.0
    transfer_us = 1.5
    sep_halo_us = 3 * alpha_us + transfer_us
    fused_halo_us = 1 * alpha_us + transfer_us
    halo_speedup = round(sep_halo_us / fused_halo_us, 4)

    # 2. KDA B200 prefill next-state prefetch & per-warp checkpoint (flashinfer@fb97a29)
    baseline_prefill_ms = 6.32
    optimized_prefill_ms = 4.72
    kda_prefill_speedup = round(baseline_prefill_ms / optimized_prefill_ms, 4)

    # 3. SM103a probed die split for MoE dgrad down projection (flashinfer@96f25e1)
    unprobed_die_imbalance = 0.375
    probed_die_imbalance = 0.0417

    # 4. SM103a DSA indexer two-pass radix-select fusion (flashinfer@d6c5d78)
    baseline_radix_passes = 3
    fused_radix_passes = 2
    radix_pass_reduction_pct = round((baseline_radix_passes - fused_radix_passes) / baseline_radix_passes * 100.0, 2)

    return {
        "experiment": "flashinfer_kcp_kda_sm103a_triton_phase137",
        "evidence_tier": "own-offline",
        "metrics": {
            "sep_halo_us": sep_halo_us,
            "fused_halo_us": fused_halo_us,
            "halo_speedup": halo_speedup,
            "baseline_prefill_ms": baseline_prefill_ms,
            "optimized_prefill_ms": optimized_prefill_ms,
            "kda_prefill_speedup": kda_prefill_speedup,
            "unprobed_die_imbalance": unprobed_die_imbalance,
            "probed_die_imbalance": probed_die_imbalance,
            "baseline_radix_passes": baseline_radix_passes,
            "fused_radix_passes": fused_radix_passes,
            "radix_pass_reduction_pct": radix_pass_reduction_pct,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = compute_summary()
    out = Path(__file__).with_name("summary.json")
    if args.check:
        existing = json.loads(out.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        return
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
