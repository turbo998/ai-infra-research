import unittest
from experiments.ai_server_cpu_numa_pinning_irq_omp_topology_selection.run import (
    evaluate_cpu_numa_irq_tiers,
)


class TestCpuNumaIrqSelection(unittest.TestCase):
    def test_tiers_and_dma(self) -> None:
        data = evaluate_cpu_numa_irq_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(data["workload"]["unbound_omp_threads"], 2048)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unbound_numa_default_omp_explosion"]
        t4 = tiers["tier4_k8s_single_numa_static_plus_irq_housekeeping_isolation"]
        self.assertLess(t1["h2d_pinned_dma_gbs"], 25.0)
        self.assertGreaterEqual(t4["h2d_pinned_dma_gbs"], 54.0)
        self.assertLessEqual(t4["training_step_jitter_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
