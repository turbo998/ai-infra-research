import unittest
from experiments.llm_moe_grouped_gemm_m_alignment_tma_padding_cutlass_selection.run import evaluate_moe_grouped_gemm_m_alignment


class TestMoeGroupedGemmMAlignment(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_moe_grouped_gemm_m_alignment()
        self.assertEqual(res["round"], 294)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_deepgemm_adaptive_m64_m128_contiguous_aligned_permute_zero_tma_update"]["achieved_fp8_tflops"],
            tiers["tier1_unaligned_ragged_me_scalar_fallback_or_cublas_loop"]["achieved_fp8_tflops"] * 6.0,
        )
        self.assertEqual(
            tiers["tier4_deepgemm_adaptive_m64_m128_contiguous_aligned_permute_zero_tma_update"]["tma_descriptor_overhead_us"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
