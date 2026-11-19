from __future__ import annotations

import unittest
from experiments.enterprise_ai_agent_identity_delegated_auth_obo_tool_permission_selection.run import build_payload


class TestEnterpriseAiAgentIdentityDelegatedAuthOboToolPermissionSelection(unittest.TestCase):
    def test_tier4_zero_confused_deputy_zero_cross_user_leak_and_full_audit(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["confused_deputy_privilege_escalation_escape_pct"], 0.0)
        self.assertEqual(tier4["unauthorized_cross_user_data_access_pct"], 0.0)
        self.assertEqual(tier4["destructive_write_action_blast_radius_pct"], 0.0)
        self.assertEqual(tier4["per_user_audit_trail_attribution_pct"], 100.0)
        self.assertLess(
            tier4["monthly_security_incident_and_governance_tco_usd"],
            tier1["monthly_security_incident_and_governance_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
