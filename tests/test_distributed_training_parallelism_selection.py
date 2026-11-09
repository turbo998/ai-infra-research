from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.distributed_training_parallelism_selection.run import RESULTS_PATH, compute_model


class TestDistributedTrainingParallelismSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_training_parallelism_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_ddp_zero1_cpu_offload"]
        t2 = tiers["tier2_full_zero3_fsdp_global"]
        t3 = tiers["tier3_hybrid_fsdp_hfsdp"]
        t4 = tiers["tier4_megatron_3d_dualpipe"]

        self.assertEqual(t1["unsharded_hbm_gb"], 244.20)
        self.assertFalse(t1["fits_80gb_without_cpu_offload"])
        self.assertEqual(t2["peak_hbm_gb"], 46.80)
        self.assertEqual(t2["mfu_pct"], 36.4)
        self.assertEqual(t3["peak_hbm_gb"], 58.40)
        self.assertEqual(t3["mfu_pct"], 54.0)
        self.assertEqual(t3["speedup_vs_zero3_x"], 1.48)
        self.assertEqual(t4["peak_hbm_gb"], 51.01)
        self.assertEqual(t4["mfu_pct"], 66.8)
        self.assertEqual(t4["speedup_vs_zero3_x"], 1.83)
        self.assertEqual(t4["cost_savings_vs_tier1_pct"], 70.2)
        self.assertEqual(t4["cost_savings_vs_zero3_pct"], 45.4)


if __name__ == "__main__":
    unittest.main()
