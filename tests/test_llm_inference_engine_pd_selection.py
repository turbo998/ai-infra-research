from __future__ import annotations

import unittest

from experiments.llm_inference_engine_pd_selection.run import build_results, main


class TestLLMInferenceEnginePDSelection(unittest.TestCase):
    def test_deterministic_metrics_match_faq_claims(self) -> None:
        results = build_results()
        self.assertEqual(results["provenance"], "own-offline")

        roofline = results["roofline_physics"]
        self.assertEqual(roofline["ridge_point_flop_per_byte"], 205.5)
        self.assertEqual(roofline["prefill_4k_flop_per_byte"], 512.0)
        self.assertEqual(roofline["decode_bs32_flop_per_byte"], 31.2)
        self.assertEqual(roofline["prefill_to_ridge_ratio"], 2.49)
        self.assertEqual(roofline["decode_to_ridge_ratio"], 0.15)

        prefix = results["prefix_cache_and_engine_optimizations"]
        self.assertEqual(prefix["uncached_ttft_ms"], 186.4)
        self.assertEqual(prefix["sglang_radix_ttft_ms"], 54.0)
        self.assertEqual(prefix["ttft_speedup_x"], 3.45)
        self.assertEqual(prefix["effective_qps_gain_x"], 2.85)
        self.assertEqual(prefix["vllm_multimodal_encoder_speedup_x"], 3.2)

        pd = results["colocated_vs_pd_disaggregation"]
        self.assertEqual(pd["colocated_chunked_p99_tpot_ms"], 68.4)
        self.assertEqual(pd["disaggregated_decode_p99_tpot_ms"], 18.2)
        self.assertEqual(pd["kv_rdma_transfer_ms"], 22.8)
        self.assertEqual(pd["tpot_tail_reduction_x"], 3.76)
        self.assertEqual(pd["goodput_gain_x"], 2.38)

        tco = results["heterogeneous_pd_tco"]
        self.assertEqual(tco["tco_reduction_pct"], 34.2)

    def test_cli_check_passes(self) -> None:
        self.assertEqual(main([]), 0)
        self.assertEqual(main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
