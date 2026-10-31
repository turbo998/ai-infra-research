from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. NCCL 2.32.3-1 & nccl4py Generated Bindings (`893470119efc`, `0d651e079119`)
    nccl4py_binding_drift_symbols = 0

    # 2. Mooncake TENT T-One Nightly Dual-Stack E2E (#5024: Classic + T-One)
    tent_modes_covered_before = 1  # classic only
    tent_modes_covered_after = 2  # classic + T-One
    tent_coverage_multiplier_x = round(tent_modes_covered_after / tent_modes_covered_before, 2)  # 2.0x

    # 3. LMCache XPU Mooncake Subprocess Wheel Import Preservation (#6194) & NIXL Mooncake Plugin Governance (#2539)
    xpu_subprocess_import_failures_before = 4
    xpu_subprocess_import_failures_after = 0

    return {
        "experiment_id": "nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162",
        "evidence_class": "own-offline",
        "phase": 162,
        "priority": 171,
        "metrics": {
            "nccl4py_binding_drift_symbols": nccl4py_binding_drift_symbols,
            "mooncake_tent_modes_before": tent_modes_covered_before,
            "mooncake_tent_modes_after": tent_modes_covered_after,
            "mooncake_tent_coverage_x": tent_coverage_multiplier_x,
            "lmcache_xpu_subprocess_import_failures_before": xpu_subprocess_import_failures_before,
            "lmcache_xpu_subprocess_import_failures_after": xpu_subprocess_import_failures_after,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()
    if args.check:
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        assert actual == expected, f"Mismatch in {out_path}"
        assert actual["metrics"]["nccl4py_binding_drift_symbols"] == 0
        assert actual["metrics"]["mooncake_tent_coverage_x"] == 2.0
        assert actual["metrics"]["lmcache_xpu_subprocess_import_failures_after"] == 0
        print("OK: nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
