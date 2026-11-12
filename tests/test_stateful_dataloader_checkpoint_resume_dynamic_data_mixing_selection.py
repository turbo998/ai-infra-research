import unittest
from experiments.stateful_dataloader_checkpoint_resume_dynamic_data_mixing_selection.run import (
    simulate_stateful_dataloader_mixing_selection,
)


class TestStatefulDataLoaderMixingSelection(unittest.TestCase):
    def test_feistel_bijection_and_o1_resume_speedup(self) -> None:
        data = simulate_stateful_dataloader_mixing_selection()
        summary = data["summary"]
        self.assertTrue(summary["feistel_bijection_verified"])
        self.assertGreater(summary["resume_speedup_ratio"], 5000.0)
        self.assertGreater(summary["cluster_cost_saved_vs_skip_usd"], 200000.0)

    def test_s5_eliminates_duplicates_and_boosts_accuracy(self) -> None:
        data = simulate_stateful_dataloader_mixing_selection()
        by_id = {s["id"]: s for s in data["scenarios"]}
        s1 = by_id["S1_Stateless_DataLoader_Reset"]
        s2 = by_id["S2_Naive_FastForward_Loop_Skip"]
        s5 = by_id["S5_Stateful_O1_Seek_Dynamic_DoReMi_Mux"]

        self.assertEqual(s5["duplicate_token_pct"], 0.0)
        self.assertLess(s5["resume_latency_sec"], 1.0)
        self.assertGreater(s2["resume_latency_sec"], 1800.0)
        self.assertGreater(s5["downstream_math_code_acc_pct"], s1["downstream_math_code_acc_pct"] + 10.0)


if __name__ == "__main__":
    unittest.main()
