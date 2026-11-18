import unittest
from experiments.enterprise_marketing_creative_brand_compliance_adtech_agent_selection.run import build_payload


class TestEnterpriseMarketingCreativeBrandComplianceAdtechAgentSelection(unittest.TestCase):
    def test_tier4_eliminates_logo_distortion_and_ftc_violations(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_agency_photoshoot_and_manual_legal_review"]
        t2 = tiers["tier2_unconstrained_text_to_image_and_freeform_copy_llm"]
        t4 = tiers["tier4_locked_3d_foreground_iclight_plus_ftc_claim_matrix_c2pa_bandit_agent"]

        self.assertGreater(t4["brand_visual_and_copy_compliance_pass_rate_pct"], 98.5)
        self.assertEqual(t4["ftc_ad_law_or_logo_hallucination_violation_rate_pct"], 0.0)
        self.assertGreater(t2["ftc_ad_law_or_logo_hallucination_violation_rate_pct"], 12.0)
        self.assertLessEqual(t4["campaign_concept_to_launch_hours"], 1.0)
        self.assertLess(t4["monthly_creative_agency_and_compliance_cost_usd"], t1["monthly_creative_agency_and_compliance_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
