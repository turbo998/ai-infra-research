from __future__ import annotations

import unittest
from experiments.enterprise_multi_cloud_ai_disaster_recovery_active_active_gateway_failover_selection.run import (
    build_payload,
)


class TestEnterpriseMultiCloudAiDisasterRecoveryActiveActiveGatewayFailoverSelection(unittest.TestCase):
    def test_tier4_achieves_four_nines_sla_subsecond_failover_and_zero_idle_waste(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["effective_service_availability_sla_pct"], 99.99)
        self.assertLess(tier4["brownout_failover_recovery_p99_latency_ms"], 1000.0)
        self.assertLess(
            tier4["brownout_failover_recovery_p99_latency_ms"],
            tier1["brownout_failover_recovery_p99_latency_ms"] * 0.05,
        )
        self.assertGreater(
            tier4["cross_model_schema_and_quality_parity_pct"],
            tier3["cross_model_schema_and_quality_parity_pct"] + 20.0,
        )
        self.assertEqual(tier4["idle_dr_standby_capacity_waste_pct"], 0.0)
        self.assertLess(
            tier4["monthly_80b_token_ha_platform_tco_usd"],
            tier2["monthly_80b_token_ha_platform_tco_usd"] * 0.45,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
