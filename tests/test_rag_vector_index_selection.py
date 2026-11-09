from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.rag_vector_index_selection.run import RESULTS_PATH, compute_model


class TestRagVectorIndexSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_vector_index_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_inmem_fp32_hnsw"]
        t2 = tiers["tier2_quantized_sq8_rabitq_hnsw"]
        t3 = tiers["tier3_nvme_ssd_diskann_vamana"]
        t4 = tiers["tier4_gpu_cuvs_cagra"]

        self.assertEqual(t1["dram_footprint_gb"], 890.9)
        self.assertEqual(t2["dram_footprint_gb"], 235.6)
        self.assertEqual(t2["monthly_cost_savings_pct"], 71.9)
        self.assertEqual(t3["dram_footprint_gb"], 48.0)
        self.assertEqual(t3["dram_reduction_x"], 18.56)
        self.assertEqual(t3["monthly_cost_savings_pct"], 91.0)
        self.assertEqual(t4["build_time_hr"], 0.92)
        self.assertEqual(t4["build_speedup_x"], 15.76)
        self.assertEqual(t4["qps"], 22400)


if __name__ == "__main__":
    unittest.main()
