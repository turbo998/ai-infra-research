import unittest
from experiments.enterprise_contact_center_qa_compliance_selection.run import build_Payload


class TestEnterpriseContactCenterQaComplianceSelection(unittest.TestCase):
    def test_tier4_achieves_highest_recall_and_lowest_cost(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_2pct_sampling_keyword_spotting"]
        t2 = tiers["tier2_raw_audio_frontier_multimodal_llm"]
        t4 = tiers["tier4_stereo_asr_acoustic_prosody_8b_15pct_70b_cascade"]

        self.assertGreater(t4["violation_recall_pct"], 95.0)
        self.assertGreater(t4["speaker_attribution_accuracy_pct"], 99.0)
        self.assertLess(t4["monthly_total_cost_usd"], t1["monthly_total_cost_usd"] * 0.2)
        self.assertLess(t4["monthly_total_cost_usd"], t2["monthly_total_cost_usd"] * 0.05)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
