import unittest
from experiments.pytorch_register_buffer_persistent_false_rope_overwrite_checkpoint_bloat_selection.run import evaluate


class TestPyTorchRegisterBufferPersistentFalseRoPEOverwriteCheckpointBloatSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_persistent_true_inv_freq_overwrite"]
        t2 = res["tiers"]["tier2_persistent_true_cached_masks_bloat"]
        t3 = res["tiers"]["tier3_plain_python_attribute_untracked"]
        t4 = res["tiers"]["tier4_persistent_false_shared_rotary"]

        self.assertAlmostEqual(t1["eval_ppl_64k"], 184.60, places=2)
        self.assertAlmostEqual(t2["checkpoint_bloat_gb"], 18.40, places=2)
        self.assertAlmostEqual(t3["device_move_crash_pct"], 100.0, places=1)
        self.assertAlmostEqual(t4["eval_ppl_64k"], 5.42, places=2)
        self.assertAlmostEqual(t4["checkpoint_bloat_gb"], 0.00, places=2)
        self.assertAlmostEqual(t4["ppl_reduction_x"], 34.1, places=1)
        self.assertAlmostEqual(t4["saved_ckpt_gb"], 18.40, places=2)


if __name__ == "__main__":
    unittest.main()
