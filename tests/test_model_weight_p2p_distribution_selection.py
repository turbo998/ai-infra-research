from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.model_weight_p2p_distribution_selection.run import RESULTS_PATH, compute_model


class TestModelWeightP2pDistributionSelection(unittest.TestCase):
    def test_results_json_matches_deterministic_model(self) -> None:
        self.assertTrue(RESULTS_PATH.exists(), f"Missing {RESULTS_PATH}")
        committed = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        computed = compute_model()
        self.assertEqual(committed, computed)

    def test_provenance_and_p2p_distribution_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertEqual(prov["provenance_type"], "synthetic")
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(prov["local_cpu_harness_measurement"], "measured")

        tiers = data["tiers"]
        t1 = tiers["tier1_centralized_oci_s3_pull"]
        t2 = tiers["tier2_sharded_registry_cdn"]
        t3 = tiers["tier3_dragonfly_p2p_tcp_tree"]
        t4 = tiers["tier4_rdma_gds_nccl_broadcast"]

        self.assertEqual(t1["rollout_p99_sec"], 2180.0)
        self.assertFalse(t1["sla_15s_pass"])
        self.assertEqual(t2["rollout_p99_sec"], 440.0)
        self.assertEqual(t3["rollout_p99_sec"], 42.0)
        self.assertEqual(t3["origin_egress_reduction_pct"], 98.4)
        self.assertEqual(t4["rollout_p99_sec"], 4.2)
        self.assertTrue(t4["sla_15s_pass"])
        self.assertEqual(t4["speedup_vs_tier1_x"], 519.05)
        self.assertEqual(t4["cost_savings_pct"], 99.8)


if __name__ == "__main__":
    unittest.main()
