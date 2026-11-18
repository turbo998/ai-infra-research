from __future__ import annotations

import unittest
from experiments.enterprise_construction_epc_bim_lidar_progress_fidic_claim_agent_selection.run import build_payload


class TestEnterpriseConstructionEpcBimLidarProgressFidicClaimAgentSelection(unittest.TestCase):
    def test_tier4_zero_overbilling_rework_and_fidic_timebar(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["subcontractor_overbilling_leakage_pct"], 0.0)
        self.assertLessEqual(tier4["mep_structural_field_rework_rate_pct"], 0.5)
        self.assertEqual(tier4["fidic_time_bar_claim_forfeiture_pct"], 0.0)
        self.assertLessEqual(tier4["deviation_detection_lag_days"], 0.5)
        self.assertLess(tier4["monthly_rework_and_overpayment_loss_usd"], tier1["monthly_rework_and_overpayment_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
