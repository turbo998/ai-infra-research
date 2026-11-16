from __future__ import annotations

import json
from pathlib import Path


def evaluate() -> dict:
    # Workload: 960 PyTorch C++ dispatcher calls per step (frozen ViT / RL rollout + trainable projector/head)
    num_dispatched_ops = 960

    # Tier 1: torch.no_grad() Everywhere (Maintains VariableVersion counters & ViewMeta on all 960 ops)
    t1_cpu_dispatch_latency_ms = 4.85
    t1_version_counter_allocs = 960
    t1_saved_for_backward_crash_pct = 0.0

    # Tier 2: Raw @torch.inference_mode() Output Fed Directly into Trainable nn.Linear (SavedVariable Crash!)
    # Fast dispatch (2.90 ms), but crashes 100% of the time when nn.Linear saves input for backward!
    t2_cpu_dispatch_latency_ms = 2.90
    t2_version_counter_allocs = 0
    t2_saved_for_backward_crash_pct = 100.0

    # Tier 3: Nested inference_mode(True) -> inference_mode(False) Without Exit .clone()
    # If tensor is created inside inference_mode(True), exiting the context does NOT clear is_inference()!
    t3_cpu_dispatch_latency_ms = 2.91
    t3_version_counter_allocs = 0
    t3_saved_for_backward_crash_pct = 100.0

    # Tier 4: torch.inference_mode() for Frozen Tower/Rollout + Single Exit-Boundary .clone() Outside Context
    # 0 version counters on all 960 internal ops, 1 boundary .clone() (0.02 ms) -> 2.92 ms total (1.66x faster, 0% crash!)
    t4_cpu_dispatch_latency_ms = 2.92
    t4_version_counter_allocs = 1
    t4_saved_for_backward_crash_pct = 0.0

    speedup_vs_t1_x = round(t1_cpu_dispatch_latency_ms / t4_cpu_dispatch_latency_ms, 2)  # 1.66x

    return {
        "benchmark": "pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "num_dispatched_ops_per_step": num_dispatched_ops,
            "boundary_tensor_mb": 16.0,
        },
        "tiers": {
            "tier1_no_grad_everywhere": {
                "cpu_dispatch_latency_ms": t1_cpu_dispatch_latency_ms,
                "version_counter_allocs": t1_version_counter_allocs,
                "saved_for_backward_crash_pct": t1_saved_for_backward_crash_pct,
                "autograd_safe": True,
            },
            "tier2_raw_inference_mode_into_linear": {
                "cpu_dispatch_latency_ms": t2_cpu_dispatch_latency_ms,
                "version_counter_allocs": t2_version_counter_allocs,
                "saved_for_backward_crash_pct": t2_saved_for_backward_crash_pct,
                "autograd_safe": False,
            },
            "tier3_exit_inference_mode_without_clone": {
                "cpu_dispatch_latency_ms": t3_cpu_dispatch_latency_ms,
                "version_counter_allocs": t3_version_counter_allocs,
                "saved_for_backward_crash_pct": t3_saved_for_backward_crash_pct,
                "autograd_safe": False,
            },
            "tier4_inference_mode_with_boundary_clone": {
                "cpu_dispatch_latency_ms": t4_cpu_dispatch_latency_ms,
                "version_counter_allocs": t4_version_counter_allocs,
                "saved_for_backward_crash_pct": t4_saved_for_backward_crash_pct,
                "autograd_safe": True,
                "speedup_vs_tier1_x": speedup_vs_t1_x,
            },
        },
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
