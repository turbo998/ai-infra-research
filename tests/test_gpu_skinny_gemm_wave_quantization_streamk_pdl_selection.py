import unittest
from experiments.gpu_skinny_gemm_wave_quantization_streamk_pdl_selection.run import (
    evaluate_skinny_gemm_pdl_tiers,
)


class TestSkinnyGemmStreamkPdlSelection(unittest.TestCase):
    def test_tiers_and_wave_quantization(self) -> None:
        data = evaluate_skinny_gemm_pdl_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertAlmostEqual(data["workload"]["dp_sm_efficiency_pct"], 51.5, places=1)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_standard_data_parallel_gemm_tail_wave"]
        t4 = tiers["tier4_stream_k_plus_pdl_programmatic_dependent_launch"]
        self.assertLess(t1["sm_wave_utilization_pct"], 55.0)
        self.assertGreaterEqual(t4["sm_wave_utilization_pct"], 96.0)
        self.assertLessEqual(t4["inter_kernel_launch_gap_us"], 0.5)


if __name__ == "__main__":
    unittest.main()
