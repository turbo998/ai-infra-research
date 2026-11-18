from __future__ import annotations

import unittest
from experiments.enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection.run import build_payload


class TestEnterpriseLegalEdiscoveryPrivilegeReviewTarCalRedactionAgentSelection(unittest.TestCase):
    def test_tier4_recall_zero_privilege_waiver_and_redaction_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["responsive_document_recall_pct"], 95.0)
        self.assertEqual(tier4["inadvertent_privilege_waiver_escape_pct"], 0.0)
        self.assertEqual(tier4["redaction_text_layer_leak_rate_pct"], 0.0)
        self.assertGreaterEqual(tier4["human_review_volume_reduction_pct"], 80.0)
        self.assertLess(tier4["monthly_ediscovery_review_and_sanction_cost_usd"], tier1["monthly_ediscovery_review_and_sanction_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
