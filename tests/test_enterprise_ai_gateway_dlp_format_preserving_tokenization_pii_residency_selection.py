from __future__ import annotations

import unittest
from experiments.enterprise_ai_gateway_dlp_format_preserving_tokenization_pii_residency_selection.run import build_payload


class TestEnterpriseAiGatewayDlpFormatPreservingTokenizationPiiResidencySelection(unittest.TestCase):
    def test_tier4_zero_pii_exposure_low_tool_error_and_sub_15ms_latency(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["external_cloud_pii_and_secret_exposure_pct"], 0.0)
        self.assertLessEqual(tier4["multi_entity_coreference_and_tool_call_error_pct"], 0.5)
        self.assertLessEqual(tier4["inline_dlp_gateway_added_p99_latency_ms"], 15.0)
        self.assertEqual(tier4["cross_border_gdpr_pipl_residency_violation_pct"], 0.0)
        self.assertLess(
            tier4["monthly_compliance_breach_and_rework_tco_usd"],
            tier1["monthly_compliance_breach_and_rework_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
