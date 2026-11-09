from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.ai_training_dataloader_selection.run import RESULTS_PATH, compute_model


class TestAiTrainingDataloaderSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_dataloader_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_posix_small_files"]
        t2 = tiers["tier2_webdataset_tar_cpu_decode"]
        t3 = tiers["tier3_mosaic_mds_litdata_streaming"]
        t4 = tiers["tier4_dali_gpu_decode_gds"]

        self.assertEqual(t1["data_stall_pct"], 64.5)
        self.assertEqual(t1["samples_per_sec_gpu"], 340)
        self.assertEqual(t2["samples_per_sec_gpu"], 980)
        self.assertEqual(t3["mid_epoch_resume_sec"], 0.4)
        self.assertEqual(t4["data_stall_pct"], 1.8)
        self.assertEqual(t4["samples_per_sec_gpu"], 1435)
        self.assertEqual(t4["throughput_gain_x"], 4.22)
        self.assertEqual(t4["monthly_savings_vs_tier1_usd"], 215846.40)


if __name__ == "__main__":
    unittest.main()
