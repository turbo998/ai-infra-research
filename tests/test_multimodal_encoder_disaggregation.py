import json
import unittest

from experiments.multimodal_encoder_disaggregation.run import (
    INPUT_PATH,
    RESULTS_JSON_PATH,
    evaluate_multimodal_epd,
)


class TestMultimodalEncoderDisaggregation(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        self.results = evaluate_multimodal_epd(self.config)

    def test_sealed_results_match(self) -> None:
        sealed = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.results, sealed)

    def test_epd_eliminates_colocated_tpot_interference(self) -> None:
        s = self.results["summary"]
        self.assertGreater(s["p99_tpot_reduction_ratio"], 3.4)
        self.assertGreater(s["throughput_gain_ratio"], 2.5)
        self.assertAlmostEqual(s["multi_turn_encoder_cache_hit_rate"], 0.92, places=2)

    def test_cache_capacity_monotonic_savings(self) -> None:
        sweep = self.results["cache_capacity_sweep"]
        saved = [row["saved_vit_tflops_per_req"] for row in sweep]
        self.assertEqual(saved, sorted(saved))
        self.assertGreater(saved[-1], 38.0)


if __name__ == "__main__":
    unittest.main()
