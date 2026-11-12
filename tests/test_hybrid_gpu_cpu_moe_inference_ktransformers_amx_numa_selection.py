import unittest
from experiments.hybrid_gpu_cpu_moe_inference_ktransformers_amx_numa_selection.run import (
    simulate_ktransformers_amx_numa_selection,
)


class TestHybridGpuCpuMoEKTransformersSelection(unittest.TestCase):
    def test_pcie_payload_and_capex_reduction(self) -> None:
        data = simulate_ktransformers_amx_numa_selection()
        summary = data["summary"]
        self.assertGreater(summary["pcie_payload_reduction_ratio"], 9000.0)
        self.assertGreater(summary["decode_speedup_vs_weight_swap"], 14.0)
        self.assertGreater(summary["capex_savings_vs_8xh800_pct"], 85.0)

    def test_s4_amx_numa_outperforms_s1_and_s2(self) -> None:
        data = simulate_ktransformers_amx_numa_selection()
        by_id = {s["id"]: s for s in data["scenarios"]}
        s1 = by_id["S1_Naive_PCIe_Expert_Weight_Swap"]
        s2 = by_id["S2_Default_CPU_MoE_Unpinned_NUMA"]
        s3 = by_id["S3_KTransformers_AVX512_NUMA_Striped"]
        s4 = by_id["S4_KTransformers_AMX_NUMA_HotExpert_Overlap"]

        self.assertGreater(s3["decode_tok_s"], s2["decode_tok_s"] * 3.0)
        self.assertGreater(s4["decode_tok_s"], s3["decode_tok_s"] * 1.5)
        self.assertGreater(s4["prefill_tok_s"], s2["prefill_tok_s"] * 5.0)
        self.assertLess(s4["pcie_traffic_per_token_mb"], s1["pcie_traffic_per_token_mb"] * 0.001)


if __name__ == "__main__":
    unittest.main()
