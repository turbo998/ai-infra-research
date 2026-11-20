from __future__ import annotations

import unittest
from experiments.enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection.run import build_payload


class TestEnterprisePrivateLlmApplianceGpuServerSizingH200L40sSelection(unittest.TestCase):
    def test_tier4_maximizes_concurrent_32k_users_and_minimizes_cost_per_user(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["concurrent_32k_context_users_supported"], 160)
        self.assertGreater(
            tier4["concurrent_32k_context_users_supported"],
            tier2["concurrent_32k_context_users_supported"] * 1.5,
        )
        self.assertLess(tier4["decode_inter_token_latency_tpot_p95_ms"], 15.0)
        self.assertLess(
            tier4["three_year_cost_per_concurrent_user_usd"],
            tier1["three_year_cost_per_concurrent_user_usd"] * 0.25,
        )
        self.assertLess(
            tier4["three_year_cost_per_concurrent_user_usd"],
            tier3["three_year_cost_per_concurrent_user_usd"] * 0.40,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
