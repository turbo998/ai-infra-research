from __future__ import annotations

import unittest
from experiments.enterprise_healthcare_rcm_icd10_medical_coding_denial_appeals_agent_selection.run import build_payload


class TestEnterpriseHealthcareRcmIcd10MedicalCodingDenialAppealsAgentSelection(unittest.TestCase):
    def test_tier4_denial_reduction_zero_ncci_violation_and_drg_cdi(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["initial_claim_denial_rate_pct"], 1.5)
        self.assertEqual(tier4["ncci_excludes1_edit_violation_pct"], 0.0)
        self.assertLessEqual(tier4["missed_cc_mcc_drg_undercoding_pct"], 0.5)
        self.assertLessEqual(tier4["dnfb_coding_turnaround_days"], 0.2)
        self.assertLess(tier4["monthly_denial_writeoff_and_coding_cost_usd"], tier1["monthly_denial_writeoff_and_coding_cost_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
