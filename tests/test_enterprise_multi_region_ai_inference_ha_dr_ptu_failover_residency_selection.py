from __future__ import annotations

import unittest
from experiments.enterprise_multi_region_ai_inference_ha_dr_ptu_failover_residency_selection.run import build_payload


class TestEnterpriseMultiRegionAiInferenceHaDrPtuFailoverResidencySelection(unittest.TestCase):
    def test_tier4_four_nines_availability_subsecond_rto_zero_gdpr_violation_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["effective_annual_service_availability_pct"], 99.99)
        self.assertLess(tier4["regional_brownout_recovery_rto_sec"], 1.0)
        self.assertEqual(tier4["gdpr_data_residency_boundary_violation_pct"], 0.0)
        self.assertGreater(tier3["gdpr_data_residency_boundary_violation_pct"], 30.0)
        self.assertLess(
            tier4["monthly_multi_region_ha_compute_tco_usd"],
            tier2["monthly_multi_region_ha_compute_tco_usd"] * 0.50,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
