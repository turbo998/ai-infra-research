from __future__ import annotations

import unittest
from experiments.enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection.run import build_payload


class TestEnterpriseCpgTradePromotionOptimizationLiftDeductionsMilpSelection(unittest.TestCase):
    def test_tier4_negative_roi_deduction_leakage_and_margin_mape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["value_destroying_negative_roi_promo_share_pct"], 4.0)
        self.assertLessEqual(tier4["invalid_retailer_deduction_leakage_pct"], 0.3)
        self.assertLessEqual(tier4["promo_induced_out_of_stock_osa_loss_pct"], 1.0)
        self.assertLessEqual(tier4["net_incremental_margin_forecast_mape_pct"], 2.5)
        self.assertLess(
            tier4["monthly_wasted_trade_spend_and_deduction_loss_usd"],
            tier1["monthly_wasted_trade_spend_and_deduction_loss_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
