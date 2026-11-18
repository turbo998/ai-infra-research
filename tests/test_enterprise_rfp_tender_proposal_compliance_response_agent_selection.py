import unittest
from experiments.enterprise_rfp_tender_proposal_compliance_response_agent_selection.run import build_payload


class TestEnterpriseRfpTenderProposalComplianceResponseAgentSelection(unittest.TestCase):
    def test_tier4_eliminates_star_clause_disqualifications_and_overpromises(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_presales_copy_paste_and_midnight_spreadsheet_check"]
        t2 = tiers["tier2_prompt_only_llm_auto_yes_compliance_and_unfiltered_rag"]
        t4 = tiers["tier4_star_clause_ast_plus_ga_version_truth_graph_plus_cert_validity_gate_agent"]

        self.assertGreater(t4["technical_compliance_matrix_accuracy_pct"], 97.0)
        self.assertEqual(t4["mandatory_star_clause_disqualification_or_overpromise_rate_pct"], 0.0)
        self.assertGreater(t2["mandatory_star_clause_disqualification_or_overpromise_rate_pct"], 12.0)
        self.assertLessEqual(t4["rfp_first_draft_turnaround_hours"], 2.0)
        self.assertLess(t4["monthly_presales_labor_and_bid_loss_cost_usd"], t1["monthly_presales_labor_and_bid_loss_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
