from __future__ import annotations

import unittest
from experiments.enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection.run import build_payload


class TestEnterpriseAiDataGovernancePurviewUnityCatalogLineageDlpSelection(unittest.TestCase):
    def test_tier4_achieves_zero_exposure_15min_deletion_and_100pct_lineage(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["sensitive_data_over_exposure_incident_rate_pct"], 0.0)
        self.assertLessEqual(tier4["gdpr_deletion_propagation_sla_hours"], 0.25)
        self.assertEqual(tier4["end_to_end_chunk_to_model_lineage_coverage_pct"], 100.0)
        self.assertGreater(
            tier4["end_to_end_chunk_to_model_lineage_coverage_pct"],
            tier3["end_to_end_chunk_to_model_lineage_coverage_pct"] + 30.0,
        )
        self.assertLess(
            tier4["monthly_governance_and_remediation_tco_usd"],
            tier1["monthly_governance_and_remediation_tco_usd"] * 0.20,
        )
        self.assertLess(
            tier4["manual_compliance_audit_prep_hours_per_quarter"],
            tier2["manual_compliance_audit_prep_hours_per_quarter"] * 0.05,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
