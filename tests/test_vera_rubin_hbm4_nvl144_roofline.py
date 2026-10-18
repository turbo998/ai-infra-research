from __future__ import annotations

import unittest

from experiments.vera_rubin_hbm4_nvl144_roofline import run as exp


class VeraRubinHbm4Nvl144RooflineTest(unittest.TestCase):
    def test_generate_results_and_check(self) -> None:
        data = exp.generate_results()
        self.assertEqual(data["experiment_id"], "vera_rubin_hbm4_nvl144_roofline")
        self.assertEqual(data["evidence_status"], "own-offline")
        sm = data["summary_metrics"]
        self.assertLess(sm["vr200_1t_moe_tpot_ms"], sm["gb200_1t_moe_tpot_ms"])
        self.assertLess(sm["gb200_1t_moe_tpot_ms"], sm["h100_1t_moe_tpot_ms"])
        self.assertGreater(sm["vr200_tpot_speedup_vs_h100"], 15.0)
        self.assertGreater(sm["vr200_cost_reduction_vs_h100_x"], 9.5)
        self.assertEqual(exp.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
