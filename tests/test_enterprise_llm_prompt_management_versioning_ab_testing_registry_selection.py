from __future__ import annotations

import unittest
from experiments.enterprise_llm_prompt_management_versioning_ab_testing_registry_selection.run import build_payload


class TestEnterpriseLlmPromptManagementVersioningAbTestingRegistrySelection(unittest.TestCase):
    def test_tier4_achieves_3sec_rollback_zero_network_overhead_and_high_cache_preservation(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["prompt_rollback_mttr_minutes"], 0.05)
        self.assertLess(
            tier4["prompt_fetch_p99_overhead_ms"],
            tier2["prompt_fetch_p99_overhead_ms"] * 0.01,
        )
        self.assertGreater(
            tier4["prefix_cache_hit_preservation_on_edit_pct"],
            tier1["prefix_cache_hit_preservation_on_edit_pct"] + 65.0,
        )
        self.assertLess(
            tier4["model_migration_rewrite_effort_days"],
            tier3["model_migration_rewrite_effort_days"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_prompt_ops_and_incident_tco_usd"],
            tier1["monthly_prompt_ops_and_incident_tco_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
