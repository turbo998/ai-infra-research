"""Unit tests for Round 370 Enterprise GPU Observability, SLO Alerting & FinOps Governance Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_gpu_observability_dcgm_otel_genai_finops_selection.run import (
    evaluate_enterprise_gpu_observability_dcgm_otel_genai_finops,
)


class TestEnterpriseGpuObservabilityDcgmOtelGenaiFinopsSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_gpu_observability_dcgm_otel_genai_finops()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_gpu_observability_dcgm_otel_genai_finops_selection",
        )
        self.assertEqual(data["round"], 370)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_fullstack_4layer_otel_genai_langfuse_finops_hub"]["incident_mttr_minutes"],
            tiers["tier1_legacy_host_and_coarse_nvidia_smi_gpu_util"]["incident_mttr_minutes"],
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_mttr_speedup_x"],
            50.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_gpu_observability_dcgm_otel_genai_finops_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_gpu_observability_dcgm_otel_genai_finops()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
