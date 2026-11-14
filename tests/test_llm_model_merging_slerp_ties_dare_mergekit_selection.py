import unittest
from experiments.llm_model_merging_slerp_ties_dare_mergekit_selection.run import (
    evaluate_model_merging_tiers,
)


class TestModelMergingSelection(unittest.TestCase):
    def test_tiers_and_retention(self) -> None:
        data = evaluate_model_merging_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_linear_weight_averaging"]
        t4 = tiers["tier4_dare_ties_drop_rescale_layer_adaptive"]
        self.assertGreater(t1["sign_conflict_rate_pct"], 30.0)
        self.assertEqual(t4["sign_conflict_rate_pct"], 0.0)
        self.assertGreaterEqual(t4["retention_vs_specialist_pct"], 99.0)


if __name__ == "__main__":
    unittest.main()
