from __future__ import annotations

import unittest
from experiments.enterprise_model_registry_oci_artifact_p2p_distribution_selection.run import build_payload


class TestEnterpriseModelRegistryOciArtifactP2pDistributionSelection(unittest.TestCase):
    def test_tier4_fastest_32node_rollout_zero_rce_and_lowest_egress_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["scale_out_32_pods_140gb_model_ready_sec"], 30.0)
        self.assertLess(
            tier4["scale_out_32_pods_140gb_model_ready_sec"],
            tier3["scale_out_32_pods_140gb_model_ready_sec"] * 0.10,
        )
        self.assertEqual(tier4["supply_chain_pickle_rce_and_tamper_risk_pct"], 0.0)
        self.assertEqual(tier4["external_nat_and_cross_az_egress_gb_per_rollout"], 140.0)
        self.assertLess(
            tier4["monthly_storage_egress_and_idle_gpu_wait_tco_usd"],
            tier1["monthly_storage_egress_and_idle_gpu_wait_tco_usd"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_storage_egress_and_idle_gpu_wait_tco_usd"],
            tier2["monthly_storage_egress_and_idle_gpu_wait_tco_usd"] * 0.20,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
