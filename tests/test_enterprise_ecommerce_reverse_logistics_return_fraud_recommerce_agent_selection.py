from __future__ import annotations

import unittest
from experiments.enterprise_ecommerce_reverse_logistics_return_fraud_recommerce_agent_selection.run import build_payload


class TestEnterpriseEcommerceReverseLogisticsReturnFraudRecommerceAgentSelection(unittest.TestCase):
    def test_tier4_value_recovery_zero_fraud_and_vip_protection(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["net_value_recovery_rate_pct"], 80.0)
        self.assertEqual(tier4["empty_box_and_switch_fraud_escape_pct"], 0.0)
        self.assertEqual(tier4["vip_customer_false_ban_churn_pct"], 0.0)
        self.assertLessEqual(tier4["return_to_resale_shelf_cycle_days"], 3.0)
        self.assertLess(tier4["monthly_reverse_logistics_and_fraud_loss_usd"], tier1["monthly_reverse_logistics_and_fraud_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
