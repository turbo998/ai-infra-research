import subprocess
import sys
import unittest
from pathlib import Path

from experiments.gpu_sharing_isolation_selection.run import RESULTS_PATH, compute_model


class TestGpuSharingIsolationSelection(unittest.TestCase):
    def test_deterministic_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "gpu_sharing_isolation_selection" / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        self.assertIn("OK:", res.stdout)
        self.assertTrue(RESULTS_PATH.exists())

    def test_provenance_and_isolation_invariants(self) -> None:
        data = compute_model()
        prov = data["provenance"]
        self.assertFalse(prov["independently_reproduced"])
        self.assertEqual(prov["gpu_cluster_hardware_measured"], "not_measured")
        self.assertEqual(prov["local_cpu_validator_measured"], "measured")

        base = data["baseline_parameters"]
        self.assertAlmostEqual(base["single_gpu_monthly_cost_usd"], 1825.00, places=2)
        self.assertAlmostEqual(base["dedicated_monthly_cost_7_tenants_usd"], 12775.00, places=2)
        self.assertAlmostEqual(base["monthly_tco_savings_7_to_1_usd"], 10950.00, places=2)
        self.assertAlmostEqual(base["tco_reduction_7_to_1_pct"], 85.7, places=1)

        tiers = data["tiers"]
        t1 = tiers["tier1_os_k8s_time_slicing"]
        t2 = tiers["tier2_cuda_mps_spatial_sharing"]
        t3 = tiers["tier3_nvidia_hardware_mig"]
        t4 = tiers["tier4_green_ctx_and_multi_lora"]

        # Fault blast radius invariant: Time-Slicing and MPS share fatal GPU fault domain (100%), MIG is silicon-isolated (0%)
        self.assertEqual(t1["fault_blast_radius_pct"], 100.0)
        self.assertEqual(t2["fault_blast_radius_pct"], 100.0)
        self.assertEqual(t3["fault_blast_radius_pct"], 0.0)

        # Tail latency ordering under multi-tenant noisy neighbor: MIG (25.0ms) < MPS (34.5ms) < Time-Slicing (82.0ms)
        self.assertLess(t3["p99_latency_ms"], t2["p99_latency_ms"])
        self.assertLess(t2["p99_latency_ms"], t1["p99_latency_ms"])

        # Throughput ordering: MPS (1840 req/s) > MIG (1620 req/s) > Time-Slicing (1180 req/s)
        self.assertGreater(t2["aggregate_throughput_req_per_s"], t3["aggregate_throughput_req_per_s"])
        self.assertGreater(t3["aggregate_throughput_req_per_s"], t1["aggregate_throughput_req_per_s"])

        # Green Contexts & Multi-LoRA invariants
        self.assertAlmostEqual(t4["green_ctx_decode_p99_reduction_x"], 2.76, places=2)
        lora = t4["lora_consolidation_scenario"]
        self.assertEqual(lora["dedicated_gpus_without_multi_lora"], 28)
        self.assertEqual(lora["shared_gpus_with_multi_lora"], 2)
        self.assertAlmostEqual(lora["gpu_count_reduction_x"], 14.0, places=1)
        self.assertAlmostEqual(lora["monthly_savings_usd"], 47450.00, places=2)


if __name__ == "__main__":
    unittest.main()
