import unittest
from experiments.quarot_spinquant_hadamard_rotation_w4a4_kv_quantization_selection.run import (
    simulate_quarot_spinquant_selection,
)


class TestQuaRotSpinQuantSelection(unittest.TestCase):
    def test_orthogonal_invariance_and_kurtosis_drop(self) -> None:
        data = simulate_quarot_spinquant_selection()
        inv = data["summary"]["invariance_verification"]
        self.assertEqual(inv["dot_product_abs_error"], 0.0)
        self.assertLess(inv["rotated_max_abs"], inv["raw_max_abs"])
        self.assertGreater(data["summary"]["kurtosis_reduction_pct"], 99.0)

    def test_spinquant_outperforms_naive_w4a4_and_awq_throughput(self) -> None:
        data = simulate_quarot_spinquant_selection()
        by_id = {s["id"]: s for s in data["scenarios"]}
        fp16 = by_id["S1_FP16_Baseline"]
        naive = by_id["S2_Naive_SmoothQuant_W4A4_KV4"]
        awq = by_id["S3_AWQ_W4A16_FP8KV_Hybrid"]
        quarot = by_id["S4_QuaRot_Hadamard_W4A4_KV4"]
        spinquant = by_id["S5_SpinQuant_Cayley_W4A4_KV4"]

        self.assertLess(spinquant["wikitext2_ppl"], naive["wikitext2_ppl"])
        self.assertLess(spinquant["wikitext2_ppl"], quarot["wikitext2_ppl"])
        self.assertLessEqual(spinquant["ppl_delta"], 0.15)
        self.assertGreater(spinquant["prefill_throughput_tok_s"], awq["prefill_throughput_tok_s"] * 2.0)
        self.assertLess(spinquant["total_vram_gb"], fp16["total_vram_gb"] * 0.30)


if __name__ == "__main__":
    unittest.main()
