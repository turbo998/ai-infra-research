from __future__ import annotations

import unittest
from experiments.enterprise_treasury_liquidity_sweeping_fx_hedging_iso20022_agent_selection.run import build_payload


class TestEnterpriseTreasuryLiquiditySweepingFxHedgingIso20022AgentSelection(unittest.TestCase):
    def test_tier4_cash_mape_netting_bps_and_zero_overdraft(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["idle_cash_buffer_drag_pct"], 2.5)
        self.assertLessEqual(tier4["thirty_day_cash_forecast_mape_pct"], 3.0)
        self.assertLessEqual(tier4["gross_fx_spread_and_wire_fee_bps"], 4.0)
        self.assertEqual(tier4["intraday_overdraft_cutoff_breach_pct"], 0.0)
        self.assertLess(tier4["monthly_fx_interest_and_overdraft_loss_usd"], tier1["monthly_fx_interest_and_overdraft_loss_usd"] * 0.10)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
