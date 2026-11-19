from __future__ import annotations

import unittest
from experiments.enterprise_structured_output_json_schema_constrained_decoding_selection.run import build_payload


class TestEnterpriseStructuredOutputJsonSchemaConstrainedDecodingSelection(unittest.TestCase):
    def test_tier4_100pct_json_validity_highest_tool_accuracy_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertEqual(tier4["strict_json_schema_first_pass_validity_pct"], 100.0)
        self.assertGreater(tier4["tool_selection_accuracy_at_250_tools_pct"], 96.0)
        self.assertGreater(
            tier4["tool_selection_accuracy_at_250_tools_pct"],
            tier1["tool_selection_accuracy_at_250_tools_pct"] + 40.0,
        )
        self.assertLess(
            tier4["p95_end_to_end_generation_latency_ms"],
            tier2["p95_end_to_end_generation_latency_ms"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_api_and_downstream_parse_failure_tco_usd"],
            tier1["monthly_api_and_downstream_parse_failure_tco_usd"] * 0.20,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
