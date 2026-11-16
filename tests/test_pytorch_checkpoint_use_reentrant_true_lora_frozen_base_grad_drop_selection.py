import unittest
from experiments.pytorch_checkpoint_use_reentrant_true_lora_frozen_base_grad_drop_selection.run import (
    evaluate_checkpoint_reentrant_tiers,
)


class TestCheckpointUseReentrantSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_checkpoint_reentrant_tiers()
        self.assertEqual(res["reentrant_true_lora_grad_survival_pct"], 0.0)
        self.assertEqual(res["reentrant_false_lora_grad_survival_pct"], 100.0)
        self.assertAlmostEqual(res["overhead_reduction_x"], 7.5, places=1)
        self.assertAlmostEqual(res["activation_saved_pct"], 72.0, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["lora_gradient_survival_pct"], 0.0)
        self.assertEqual(t4["lora_gradient_survival_pct"], 100.0)
        self.assertTrue(t4["torch_compile_compatible"])
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
