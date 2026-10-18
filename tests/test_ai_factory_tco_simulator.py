import json
import unittest

from experiments.ai_factory_tco_simulator import run as tco_mod


class TestAiFactoryTcoSimulator(unittest.TestCase):
    def test_deterministic_evaluation_and_stored_results(self) -> None:
        cfg = json.loads(tco_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = tco_mod.evaluate(cfg)
        tco_mod.verify_results(res)
        stored = json.loads(tco_mod.RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(res, stored)

    def test_pareto_progression_across_generations(self) -> None:
        cfg = json.loads(tco_mod.INPUT_PATH.read_text(encoding="utf-8"))
        res = tco_mod.evaluate(cfg)
        s = res["summary"]
        self.assertGreaterEqual(s["gen3_vs_gen1_training_cost_efficiency_ratio"], 7.0)
        self.assertGreaterEqual(s["gen3_vs_gen1_token_cost_reduction_factor"], 11.0)
        self.assertLess(s["gen3_usd_per_1m_tokens"], s["het_usd_per_1m_tokens"])
        self.assertLess(s["het_usd_per_1m_tokens"], s["gen2_usd_per_1m_tokens"])


if __name__ == "__main__":
    unittest.main()
