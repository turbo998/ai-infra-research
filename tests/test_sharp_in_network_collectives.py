import json
import unittest
from pathlib import Path

from experiments.sharp_in_network_collectives import run as sharp_mod


class TestSharpInNetworkCollectives(unittest.TestCase):
    def test_deterministic_evaluation_and_stored_results(self) -> None:
        cfg = json.loads(sharp_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = sharp_mod.evaluate(cfg)
        sharp_mod.verify_results(res)
        stored = json.loads(sharp_mod.RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(res, stored)

    def test_sharp_scaling_and_group_exhaustion(self) -> None:
        cfg = json.loads(sharp_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = sharp_mod.evaluate(cfg)
        s = res["summary"]
        self.assertGreater(s["scale_16384_sharp_speedup_vs_ring"], 3.0)
        self.assertLess(s["jobs_80_managed_nvls_tree_p99_ms"], s["jobs_80_unmanaged_p99_ms"])

    def test_bitwise_reproducibility_bounds(self) -> None:
        cfg = json.loads(sharp_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = sharp_mod.evaluate(cfg)
        bw = res["bitwise_reproducibility"]
        self.assertGreater(bw["bf16_tree_bitwise_mismatch_rate_pct"], 85.0)
        self.assertLess(bw["fp32_switch_accum_bf16_mismatch_rate_pct"], 10.0)
        self.assertEqual(bw["fixed_arity_deterministic_tree_mismatch_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
