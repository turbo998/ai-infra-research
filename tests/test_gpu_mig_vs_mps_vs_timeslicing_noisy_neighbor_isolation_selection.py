import unittest
from experiments.gpu_mig_vs_mps_vs_timeslicing_noisy_neighbor_isolation_selection.run import (
    evaluate_gpu_mig_mps_timeslicing_tiers,
)


class TestGpuMigMpsTimeslicingSelection(unittest.TestCase):
    def test_gpu_sharing_tiers(self) -> None:
        data = evaluate_gpu_mig_mps_timeslicing_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_k8s_gpu_time_slicing_oversubscription"]
        t2 = tiers["tier2_whole_gpu_cuda_mps_shared_context"]
        t4 = tiers["tier4_hierarchical_mig_cross_tenant_plus_intra_slice_mps"]
        self.assertEqual(t1["xid31_fault_blast_radius_pct"], 100.0)
        self.assertEqual(t2["xid31_fault_blast_radius_pct"], 100.0)
        self.assertEqual(t4["xid31_fault_blast_radius_pct"], 0.0)
        self.assertLess(t4["noisy_neighbor_p99_latency_ms"], 15.0)


if __name__ == "__main__":
    unittest.main()
