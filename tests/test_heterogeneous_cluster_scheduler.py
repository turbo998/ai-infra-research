import json
import unittest

from experiments.heterogeneous_cluster_scheduler import run as het_mod


class TestHeterogeneousClusterScheduler(unittest.TestCase):
    def test_deterministic_evaluation_and_stored_results(self) -> None:
        cfg = json.loads(het_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = het_mod.evaluate(cfg)
        het_mod.verify_results(res)
        stored = json.loads(het_mod.RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(res, stored)

    def test_asymmetric_pp_layout_beats_uniform(self) -> None:
        cfg = json.loads(het_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = het_mod.evaluate(cfg)
        s = res["summary"]
        self.assertGreaterEqual(s["pp_throughput_speedup"], 1.45)
        self.assertLess(s["asymmetric_optimal_waste_pct"], 15.0)
        self.assertGreater(s["uniform_waste_pct"], 35.0)

    def test_cross_gen_kv_transcode_speedup(self) -> None:
        cfg = json.loads(het_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = het_mod.evaluate(cfg)
        s = res["summary"]
        self.assertGreater(s["kv_128k_speedup_vs_bf16"], 3.5)
        self.assertGreater(s["kv_128k_speedup_vs_fp8"], 2.0)


if __name__ == "__main__":
    unittest.main()
