import unittest
from experiments.enterprise_healthcare_clinical_scribe_fhir_selection.run import build_Payload


class TestEnterpriseHealthcareClinicalScribeFhirSelection(unittest.TestCase):
    def test_tier4_achieves_highest_clinical_safety_and_coding_accuracy(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_generic_asr_free_text_soap"]
        t2 = tiers["tier2_monolithic_frontier_audio_llm"]
        t4 = tiers["tier4_rxnorm_asr_fhir_r4_icd10_tree_evidence_linked"]

        self.assertGreater(t4["medication_negation_accuracy_pct"], 99.5)
        self.assertGreater(t4["icd10_cpt_coding_audit_pass_pct"], t1["icd10_cpt_coding_audit_pass_pct"] + 35.0)
        self.assertLess(t4["physician_review_minutes_per_encounter"], 1.0)
        self.assertLess(t4["monthly_total_cost_usd"], t2["monthly_total_cost_usd"] * 0.1)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
