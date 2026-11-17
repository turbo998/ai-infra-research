import unittest
from experiments.enterprise_legal_contract_review_redlining_selection.run import build_Payload


class TestEnterpriseLegalContractReviewRedliningSelection(unittest.TestCase):
    def test_tier4_achieves_highest_risk_recall_and_docx_fidelity(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_512_chunk_rag_chat"]
        t2 = tiers["tier2_128k_monolithic_prompt_full_rewrite"]
        t4 = tiers["tier4_legal_ast_xref_pinning_playbook_minimal_openxml_diff"]

        self.assertGreater(t4["risk_clause_recall_pct"], 98.0)
        self.assertLess(t4["unverifiable_edit_error_pct"], 0.5)
        self.assertGreater(t4["docx_track_changes_fidelity_pct"], 99.0)
        self.assertLess(t4["monthly_total_cost_usd"], t2["monthly_total_cost_usd"] * 0.2)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
