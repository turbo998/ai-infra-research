from __future__ import annotations

import unittest
from experiments.enterprise_fpa_financial_planning_variance_pvm_monte_carlo_agent_selection.run import build_payload


class TestEnterpriseFpaFinancialPlanningVariancePvmMonteCarloAgentSelection(unittest.TestCase):
    def test_tier4_zero_reconciliation_error_zero_pvm_residual_and_low_mape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["monthly_close_and_variance_cycle_days"], 0.5)
        self.assertEqual(tier4["ebitda_intercompany_reconciliation_error_pct"], 0.0)
        self.assertLessEqual(tier4["rolling_12m_cash_forecast_mape_pct"], 2.5)
        self.assertEqual(tier4["unexplained_pvm_fx_residual_pct"], 0.0)
        self.assertLess(
            tier4["monthly_fpa_ops_and_idle_working_capital_cost_usd"],
            tier1["monthly_fpa_ops_and_idle_working_capital_cost_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
