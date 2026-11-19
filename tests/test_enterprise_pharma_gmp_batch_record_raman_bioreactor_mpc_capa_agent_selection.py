from __future__ import annotations

import unittest
from experiments.enterprise_pharma_gmp_batch_record_raman_bioreactor_mpc_capa_agent_selection.run import build_payload


class TestEnterprisePharmaGmpBatchRecordRamanBioreactorMpcCapaAgentSelection(unittest.TestCase):
    def test_tier4_release_hold_yield_loss_and_zero_alcoa_escape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["qa_batch_release_hold_days"], 0.8)
        self.assertLessEqual(tier4["bioreactor_titer_yield_loss_and_batch_scrap_pct"], 1.0)
        self.assertLessEqual(tier4["repeat_deviation_and_human_error_recurrence_pct"], 2.5)
        self.assertEqual(tier4["alcoa_plus_audit_trail_and_cqa_escape_pct"], 0.0)
        self.assertLess(
            tier4["monthly_batch_hold_scrap_and_qa_cost_usd"],
            tier1["monthly_batch_hold_scrap_and_qa_cost_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
