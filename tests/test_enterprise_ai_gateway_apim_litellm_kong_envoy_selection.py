"""Unit tests for Enterprise AI Gateway Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_ai_gateway_apim_litellm_kong_envoy_selection.run import (
    evaluate_enterprise_ai_gateway_apim_litellm_kong_envoy,
)


class TestEnterpriseAiGatewayApimLitellmKongEnvoySelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_ai_gateway_apim_litellm_kong_envoy()
        self.assertEqual(data["experiment_id"], "enterprise_ai_gateway_apim_litellm_kong_envoy_selection")
        self.assertEqual(data["round"], 362)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier3_cloud_native_envoy_higress_kong_ai_gateway"]["p99_gateway_overhead_ms"],
            tiers["tier2_litellm_proxy_python_agile_gateway"]["p99_gateway_overhead_ms"],
        )
        self.assertGreater(
            tiers["tier4_azure_apim_ai_gateway_enterprise_hub"]["effective_availability_pct"],
            99.95,
        )
        self.assertGreater(
            data["summary"]["tier4_tco_savings_vs_tier1_unmanaged_pct"],
            40.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_ai_gateway_apim_litellm_kong_envoy_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_ai_gateway_apim_litellm_kong_envoy()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
