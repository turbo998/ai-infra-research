from __future__ import annotations

import unittest
from experiments.enterprise_tax_vat_sales_use_nexus_einvoicing_compliance_agent_selection.run import build_payload


class TestEnterpriseTaxVatSalesUseNexusEinvoicingComplianceAgentSelection(unittest.TestCase):
    def test_tier4_zero_audit_error_zero_einvoice_rejection_and_zero_cert_escape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["sales_and_vat_tax_audit_assessment_error_pct"], 0.0)
        self.assertLessEqual(tier4["ap_vendor_use_tax_over_and_under_payment_pct"], 0.3)
        self.assertEqual(tier4["peppol_vida_einvoice_clearance_rejection_pct"], 0.0)
        self.assertEqual(tier4["expired_or_invalid_exemption_cert_escape_pct"], 0.0)
        self.assertLess(
            tier4["monthly_tax_penalty_overpayment_and_ops_cost_usd"],
            tier1["monthly_tax_penalty_overpayment_and_ops_cost_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
