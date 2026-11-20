from __future__ import annotations

import unittest
from experiments.enterprise_ai_landing_zone_hub_spoke_private_link_egress_firewall_selection.run import build_payload


class TestEnterpriseAiLandingZoneHubSpokePrivateLinkEgressFirewallSelection(unittest.TestCase):
    def test_tier4_achieves_zero_exfiltration_zero_dns_outage_and_lowest_hop_latency(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["data_exfiltration_attack_surface_pct"], 0.0)
        self.assertEqual(tier4["dns_misconfiguration_outage_rate_pct"], 0.0)
        self.assertLess(
            tier4["network_hop_p99_overhead_ms"],
            tier3["network_hop_p99_overhead_ms"] * 0.05,
        )
        self.assertLessEqual(
            tier4["cross_subscription_private_endpoint_sprawl_count"],
            tier2["cross_subscription_private_endpoint_sprawl_count"] * 0.10,
        )
        self.assertLess(
            tier4["monthly_30_spoke_network_and_secops_tco_usd"],
            tier1["monthly_30_spoke_network_and_secops_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
