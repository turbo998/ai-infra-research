import json
import unittest
from pathlib import Path

from experiments.cuda_graph_piecewise_capture.run import (
    INPUT_PATH,
    RESULTS_JSON_PATH,
    evaluate_cuda_graph_capture,
)


class TestCudaGraphPiecewiseCapture(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        self.results = evaluate_cuda_graph_capture(self.config)

    def test_sealed_results_match(self) -> None:
        sealed = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.results, sealed)

    def test_memory_pool_compression_and_cpu_launch_wall(self) -> None:
        s = self.results["summary"]
        self.assertGreater(s["memory_pool_compression_ratio"], 50.0)
        self.assertLess(s["shared_global_graph_pool_gb"], 1.5)
        self.assertGreater(s["b200_bs1_piecewise_speedup"], 1.8)

    def test_piecewise_beats_full_graph_on_unaligned_batches(self) -> None:
        s = self.results["summary"]
        self.assertLess(
            s["h100_bs33_padded_to_40_piecewise_tpot_ms"],
            s["h100_bs33_full_graph_tpot_ms"],
        )
        self.assertGreater(s["b200_bs257_piecewise_vs_full_delta_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
