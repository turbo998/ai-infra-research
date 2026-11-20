from __future__ import annotations

import unittest
from experiments.enterprise_agent_tool_compensation_saga_idempotency_rollback_selection.run import build_payload


class TestEnterpriseAgentToolCompensationSagaIdempotencyRollbackSelection(unittest.TestCase):
    def test_tier4_eliminates_duplicate_writes_and_dirty_residue_with_fast_compensation(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["duplicate_charge_or_double_write_rate_pct"], 0.0)
        self.assertEqual(tier4["partial_failure_dirty_state_residue_pct"], 0.0)
        self.assertGreater(tier4["automated_rollback_recovery_rate_pct"], 99.5)
        self.assertGreater(
            tier4["automated_rollback_recovery_rate_pct"],
            tier2["automated_rollback_recovery_rate_pct"] + 45.0,
        )
        self.assertLess(
            tier4["saga_compensation_p95_latency_ms"],
            tier2["saga_compensation_p95_latency_ms"] * 0.05,
        )
        self.assertLess(
            tier4["monthly_400k_transaction_reconciliation_tco_usd"],
            tier1["monthly_400k_transaction_reconciliation_tco_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
