from __future__ import annotations

import unittest
from experiments.enterprise_multimodal_video_image_understanding_indexing_rag_selection.run import build_payload


class TestEnterpriseMultimodalVideoImageUnderstandingIndexingRagSelection(unittest.TestCase):
    def test_tier4_highest_visual_and_temporal_recall_at_fraction_of_1fps_vlm_cost(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["visual_and_screen_share_event_recall_pct"], 96.0)
        self.assertGreater(
            tier4["visual_and_screen_share_event_recall_pct"],
            tier1["visual_and_screen_share_event_recall_pct"] + 75.0,
        )
        self.assertGreater(tier4["temporal_timestamp_localization_iou_pct"], 94.0)
        self.assertLess(
            tier4["cost_per_1000_hours_video_indexed_usd"],
            tier2["cost_per_1000_hours_video_indexed_usd"] * 0.08,
        )
        self.assertLess(
            tier4["monthly_50k_hour_video_platform_tco_usd"],
            tier2["monthly_50k_hour_video_platform_tco_usd"] * 0.08,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
