from __future__ import annotations

import unittest
from experiments.enterprise_ai_gateway_authentication_entra_id_mtls_api_key_rotation_selection.run import build_payload


class TestEnterpriseAiGatewayAuthenticationEntraIdMtlsApiKeyRotationSelection(unittest.TestCase):
    def test_tier4_eliminates_rotation_outages_and_minimizes_auth_latency_and_blast_radius(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["key_rotation_service_outage_rate_pct"], 0.0)
        self.assertLess(
            tier4["credential_compromise_blast_radius_hours"],
            tier2["credential_compromise_blast_radius_hours"] * 0.02,
        )
        self.assertLess(
            tier4["auth_verification_p99_overhead_ms"],
            tier3["auth_verification_p99_overhead_ms"] * 0.01,
        )
        self.assertEqual(tier4["caller_identity_non_repudiation_coverage_pct"], 100.0)
        self.assertLess(
            tier4["monthly_security_ops_and_leak_tco_usd"],
            tier1["monthly_security_ops_and_leak_tco_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
